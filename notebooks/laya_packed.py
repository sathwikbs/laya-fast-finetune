"""Laya in a one-pass-per-record ("packed") layout: checks, profiling, training, evaluation.

Standard Laya encodes every (question, state) pair as its own sequence, so a state with five
questions is encoded five times. Here one record is one sequence:

    [CLS] q1 [SEP] [MASK] opt ... [SEP] ... [CLS] qn [SEP] [MASK] opt ... [SEP] state [SEP]

Attention: a question's tokens see their own segment and the state, never another question;
state tokens see everything. Position ids put every question directly before the state, so each
allowed query/key pair keeps the relative offset it has in a standard sequence (RoPE and the
+/-64 local window depend only on that offset). The decision head runs on the packed sequence
with the same mask. Each question adds its type embedding to its own tokens and the state gets
the mean of the record's question types, so a one-question record is exactly standard Laya.

Loss, optimizer, LR schedule, sigma schedule and temperature calibration are the upstream
script's. The RL advantage is normalised over pairs of questions, as the upstream micro-batch
of 2 does (an odd question out of a record is normalised alone).

    python notebooks/laya_packed.py check
    python notebooks/laya_packed.py profile
    python notebooks/laya_packed.py train --epochs 1 --output-dir ./laya_packed_e1
    python notebooks/laya_packed.py eval --output-dir ./laya_packed_e1
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from safetensors.torch import load_file
from transformers import AutoTokenizer

import laya_finetune_typed_decisions_mps as base
from laya.common import build_head, build_model, build_sequence, proper_reward, render_options, QTYPES

LAYA = Path(__file__).resolve().parent.parent
TRAIN_PARQUET = LAYA / "typed-decisions/all/train-00000-of-00001.parquet"
STEP_QUESTIONS = 32


# ------------------------------------------------------------------ data
def load_rows(tokenizer, cfg):
    """Training rows built exactly like base.prepare_items, plus each item's head length."""
    rows, flat = [], []
    for r in pd.read_parquet(TRAIN_PARQUET).itertuples(index=False):
        state, questions, gold = json.loads(r.state), json.loads(r.questions), json.loads(r.gold)
        entries = []
        for qid, question in questions.items():
            if qid not in gold:
                continue
            item = base.build_training_item(tokenizer, cfg, state, question, gold[qid])
            if item is None:
                continue
            q = {"t": question["type"], "ins": question["instructions"], "crit": question.get("criteria", {})}
            head = build_head(tokenizer, q, cfg["head_max_len"])[0]
            assert item["ids"][:len(head)] == head, "item does not start with its head"
            item = dict(item, H=len(head), index=len(flat))
            entries.append(item)
            flat.append(item)
        rows.append(entries)
    return rows, flat


def pack(entries):
    """One packed record from standard items of the same state (head first, then state)."""
    states = [e["ids"][e["H"]:] for e in entries]
    state = max(states, key=len)
    for s in states:   # per-question truncations of one state are prefixes of the longest
        assert s[:-1] == state[:len(s) - 1] and s[-1] == state[-1], "state parts differ"
    p = max(e["H"] for e in entries)
    ids, pos, seg, questions = [], [], [], []
    for i, e in enumerate(entries):
        off = len(ids)
        assert all(m < e["H"] for m in e["markers"]), "marker outside the head"
        ids += e["ids"][:e["H"]]
        pos += range(p - e["H"], p)
        seg += [i] * e["H"]
        questions.append({"markers": [off + m for m in e["markers"]], "cls": off,
                          "qtype": e["qtype"], "target": e.get("target"), "item": e})
    ids += state
    pos += range(p, p + len(state))
    seg += [-1] * len(state)
    return {"ids": ids, "pos": pos, "seg": seg, "questions": questions}


def compact(record, keep_items=False):
    """The record as numpy arrays (4 bytes per token instead of a Python int each), without the
    per-question standard items unless asked for (they hold a full copy of the state each)."""
    out = {"ids": np.asarray(record["ids"], dtype=np.int32), "pos": np.asarray(record["pos"], dtype=np.int32),
           "seg": np.asarray(record["seg"], dtype=np.int16), "questions": []}
    for q in record["questions"]:
        c = {"markers": np.asarray(q["markers"], dtype=np.int32), "cls": int(q["cls"]), "qtype": int(q["qtype"]),
             "target": None if q["target"] is None else np.asarray(q["target"], dtype=np.float32)}
        if keep_items:
            c["item"] = q["item"]
        out["questions"].append(c)
    return out


def record_bytes(records):
    return sum(r["ids"].nbytes + r["pos"].nbytes + r["seg"].nbytes
               + sum(q["markers"].nbytes + (0 if q["target"] is None else q["target"].nbytes) for q in r["questions"])
               for r in records)


def split_records(rows, n_items, calib_max=400, keep_items=False):
    """The upstream calibration split (by item), then one compact packed record per row and split."""
    order = list(range(n_items))
    random.Random(20260922).shuffle(order)
    calib = set(order[:min(calib_max, n_items // 10)])
    train_recs, calib_recs = [], []
    for entries in rows:
        tr = [e for e in entries if e["index"] not in calib]
        ca = [e for e in entries if e["index"] in calib]
        if tr:
            train_recs.append(compact(pack(tr), keep_items))
        if ca:
            calib_recs.append(compact(pack(ca), keep_items))
    return train_recs, calib_recs


# ------------------------------------------------------------------ model
def collate_packed(records, pad_id, window, device, state_sees_questions=True):
    b, length = len(records), max(len(r["ids"]) for r in records)
    ids = torch.full((b, length), pad_id, dtype=torch.long)
    pos = torch.zeros((b, length), dtype=torch.long)
    seg = torch.full((b, length), -2, dtype=torch.long)          # -2 padding, -1 state, i question i
    for i, r in enumerate(records):
        n = len(r["ids"])
        ids[i, :n] = torch.as_tensor(np.asarray(r["ids"], dtype=np.int64))
        pos[i, :n] = torch.as_tensor(np.asarray(r["pos"], dtype=np.int64))
        seg[i, :n] = torch.as_tensor(np.asarray(r["seg"], dtype=np.int64))
    real, is_state = seg != -2, seg == -1
    allowed = (seg[:, :, None] == seg[:, None, :]) | is_state[:, None, :]
    if state_sees_questions:
        allowed = allowed | is_state[:, :, None]
    allowed = allowed & real[:, None, :]
    pad_rows = ~real
    allowed = allowed | (pad_rows[:, :, None] & torch.eye(length, dtype=torch.bool)[None])   # finite pad rows
    near = (pos[:, :, None] - pos[:, None, :]).abs() <= window

    qs = [(i, q) for i, r in enumerate(records) for q in r["questions"]]
    kmax = max(len(q["markers"]) for _, q in qs)
    q_rec = torch.tensor([i for i, _ in qs])
    q_markers = torch.zeros((len(qs), kmax), dtype=torch.long)
    q_mask = torch.zeros((len(qs), kmax), dtype=torch.bool)
    q_target = torch.zeros((len(qs), kmax), dtype=torch.float32)
    for j, (_, q) in enumerate(qs):
        k = len(q["markers"])
        q_markers[j, :k] = torch.as_tensor(np.asarray(q["markers"], dtype=np.int64))
        q_mask[j, :k] = True
        if q["target"] is not None:
            q_target[j, :len(q["target"])] = torch.as_tensor(np.asarray(q["target"], dtype=np.float32))
    type_w = torch.zeros((b, length, 3))
    for i, r in enumerate(records):
        counts = torch.zeros(3)
        for qi, q in enumerate(r["questions"]):
            type_w[i, seg[i] == qi, q["qtype"]] = 1.0
            counts[q["qtype"]] += 1
        type_w[i, is_state[i]] = counts / counts.sum()
    to = lambda t: t.to(device)  # noqa: E731
    return {"ids": to(ids), "pos": to(pos), "full": to(allowed[:, None]), "slide": to((allowed & near)[:, None]),
            "type_w": to(type_w), "q_rec": to(q_rec), "q_markers": to(q_markers), "q_mask": to(q_mask),
            "q_target": to(q_target), "q_cls": to(torch.tensor([q["cls"] for _, q in qs])),
            "q_qtype": to(torch.tensor([q["qtype"] for _, q in qs])), "n_questions": len(qs)}


def head_layer(layer, h, allowed):
    """nn.TransformerEncoderLayer (norm_first) with a per-record boolean mask (True = may attend)."""
    x = layer.norm1(h)
    h = h + layer.dropout1(layer.self_attn(x, x, x, attn_mask=~allowed, need_weights=False)[0])
    return h + layer._ff_block(layer.norm2(h))


def packed_forward(model, batch):
    """DecisionModel.forward on packed records: (logits [Q, K], act [Q, n_act])."""
    enc = model.encoder
    h = enc.embeddings(input_ids=batch["ids"])
    rope = {t: enc.rotary_emb(h, batch["pos"], t) for t in set(enc.config.layer_types)}
    masks = {"full_attention": batch["full"], "sliding_attention": batch["slide"]}
    for layer in enc.layers:
        h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=rope[layer.attention_type])
    h = enc.final_norm(h)
    h = h + batch["type_w"] @ model.type_emb.weight
    if model.head is not None:
        for layer in model.head.layers:
            if model.head_checkpointing and model.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(head_layer, layer, h, batch["full"], use_reentrant=False)
            else:
                h = head_layer(layer, h, batch["full"])
    m = h[batch["q_rec"][:, None], batch["q_markers"]]
    marker_mask = batch["q_mask"]
    # From here on, the tail of DecisionModel.forward (laya/common.py), per question.
    logits = model.scorer(m).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, -1e4)
    p = torch.softmax(logits.detach(), -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    if p.size(-1) >= 2:
        top2 = p.topk(2, -1).values
    else:
        top1 = p.topk(1, -1).values
        top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    pooled = h[batch["q_rec"], batch["q_cls"]].float()
    act_logits = model.act_head(torch.cat([pooled, feats], -1))
    return logits, act_logits


def question_loss(logits, act, mask, target, qtype, sigma, pairs):
    """Sum over questions of the upstream CE + RL loss (RL advantage normalised per pair)."""
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((4,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    noisy = logits.detach().unsqueeze(0) + eps
    probabilities = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(probabilities, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        advantage = reward - reward.mean(0, keepdim=True)
        for group in pairs:
            advantage[:, group] = advantage[:, group] / (advantage[:, group].std() + 1e-6)
    logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    rl = -(advantage * logp).mean(0)
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    return (rl + ce).sum() + 0.0 * act.float().sum()


def pairs_for(n, device):
    return [torch.arange(i, min(i + 2, n), device=device) for i in range(0, n, 2)]


def load_model(model_dir, device, checkpointing=False):
    """The upstream script's model setup (base.train)."""
    with open(Path(model_dir) / "rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg.update({"max_tokens_per_batch": 2048, "max_len": 1024, "head_max_len": 256})
    if checkpointing:
        cfg["gradient_checkpointing"] = True
    tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    model = build_model(cfg, encoder_dir=Path(model_dir) / "encoder")
    model.load_state_dict(load_file(str(Path(model_dir) / "model.safetensors")), strict=True)
    model.float()
    if checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device).train()
    return cfg, tok, model


# ------------------------------------------------------------------ modes
def mode_check(device):
    base_cfg = json.load(open(LAYA / "laya_base/rl_agent_config.json"))
    cfg, tok, model = load_model(LAYA / "laya_base", device)
    rows, flat = load_rows(tok, base_cfg)
    stored = torch.load(LAYA / "train_items.pt", weights_only=False)
    same = len(stored) == len(flat) and all(
        a["ids"] == b["ids"] and a["markers"] == b["markers"] and a["target"] == b["target"]
        and a["qtype"] == b["qtype"] for a, b in zip(stored, flat))
    print(f"check A, items rebuilt from the parquet == train_items.pt: {same} ({len(flat)} items, {len(rows)} rows)",
          flush=True)
    assert same
    window = model.encoder.config.sliding_window
    model.eval()
    worst_l = worst_a = 0.0
    with torch.no_grad():
        for entries in rows[:40]:
            e = entries[0]
            b = collate_packed([pack([e])], tok.pad_token_id, window, device)
            lp, ap = packed_forward(model, b)
            ids, att, mpos, mmask, _, qt = (t.to(device) for t in base.collate([e], tok.pad_token_id))
            ls, as_ = model(ids, att, mpos, mmask, qt)
            worst_l = max(worst_l, (lp - ls).abs()[mmask].max().item())
            worst_a = max(worst_a, (ap - as_).abs().max().item() / as_.abs().max().item())
    print(f"check B, one-question record == standard Laya: max|dlogit| {worst_l:.2e}, act rel diff {worst_a:.2e}",
          flush=True)
    worst = 0.0
    with torch.no_grad():
        for ci, entries in enumerate(rows[:40]):
            if len(entries) < 2:
                continue
            # Same question type: the state's type embedding is the mean of the record's question
            # types, so a different-type swap legitimately reaches question 1 through the state.
            donor = next(e for es in rows[ci + 1:] for e in es if e["qtype"] == entries[1]["qtype"])
            changed = list(entries)
            changed[1] = dict(donor, ids=donor["ids"][:donor["H"]] + entries[1]["ids"][entries[1]["H"]:])
            a = packed_forward(model, collate_packed([pack(entries)], tok.pad_token_id, window, device, False))[0]
            c = packed_forward(model, collate_packed([pack(changed)], tok.pad_token_id, window, device, False))[0]
            worst = max(worst, (a[0] - c[0]).abs()[: len(entries[0]["markers"])].max().item())
    print(f"check C, questions are isolated (state blind to questions, replace question 2, question 1 "
          f"unchanged): max|dlogit| {worst:.2e}", flush=True)
    worst = 0
    for entries in rows[:200]:
        a = collate_packed([pack(entries)], tok.pad_token_id, window, "cpu")
        c = collate_packed([compact(pack(entries))], tok.pad_token_id, window, "cpu")
        worst = max(worst, sum(int(not torch.equal(a[k], c[k])) for k in a if torch.is_tensor(a[k])))
    print(f"check E, compact records give identical batches: {worst == 0} (200 records)", flush=True)
    assert worst == 0
    model.train()
    train_recs, calib_recs = split_records(rows, len(flat), keep_items=True)
    b = collate_packed(train_recs[:1], tok.pad_token_id, window, device)
    logits, act = packed_forward(model, b)
    loss = question_loss(logits, act, b["q_mask"], b["q_target"], b["q_qtype"], 0.4, pairs_for(b["n_questions"], device))
    loss.backward()
    g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    model.zero_grad(set_to_none=True)
    print(f"check D, training micro-batch: loss {loss.item():.4f}, grad norm {g:.4f}, finite "
          f"{bool(np.isfinite(loss.item()) and np.isfinite(g))}", flush=True)
    n_tr = sum(len(r["questions"]) for r in train_recs)
    n_ca = sum(len(r["questions"]) for r in calib_recs)
    print(f"records: train {len(train_recs)} ({n_tr} questions), calibration {len(calib_recs)} ({n_ca} questions); "
          f"tokens per epoch: packed {sum(len(r['ids']) for r in train_recs):,} vs standard "
          f"{sum(len(q['item']['ids']) for r in train_recs for q in r['questions']):,}", flush=True)


def mode_profile(device, n_records=60, warmup=8, checkpointing=False):
    base_cfg = json.load(open(LAYA / "laya_base/rl_agent_config.json"))
    cfg, tok, model = load_model(LAYA / "laya_base", device, checkpointing)
    rows, flat = load_rows(tok, base_cfg)
    train_recs, _ = split_records(rows, len(flat), keep_items=True)
    sample = random.Random(5).sample(train_recs, warmup + n_records)
    window = model.encoder.config.sliding_window
    model.train()
    results = {}
    for layout in ("standard", "packed"):
        peak = 0

        def run(recs):
            nonlocal peak
            n_q = 0
            for r in recs:
                if layout == "packed":
                    b = collate_packed([r], tok.pad_token_id, window, device)
                    logits, act = packed_forward(model, b)
                    loss = question_loss(logits, act, b["q_mask"], b["q_target"], b["q_qtype"], 0.4,
                                         pairs_for(b["n_questions"], device))
                    loss.backward()
                else:
                    items = [q["item"] for q in r["questions"]]
                    for i in range(0, len(items), 2):
                        chunk = items[i:i + 2]
                        ids, att, mpos, mmask, target, qt = (t.to(device) for t in base.collate(chunk, tok.pad_token_id))
                        logits, act = model(ids, att, mpos, mmask, qt)
                        loss = question_loss(logits.float(), act, mmask, target, qt, 0.4, pairs_for(len(chunk), device))
                        loss.backward()
                loss.item()
                peak = max(peak, torch.mps.driver_allocated_memory())
                model.zero_grad(set_to_none=True)
                n_q += len(r["questions"])
            return n_q

        run(sample[:warmup])
        torch.mps.synchronize()
        t0 = time.perf_counter()
        n_q = run(sample[warmup:])
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        results[layout] = n_q / dt
        print(f"PROFILE {layout:<8} {n_q} questions in {dt:.1f}s = {n_q / dt:.2f} questions/s, "
              f"peak {peak / 2**30:.1f} GB (+{2 * 4 * sum(p.numel() for p in model.parameters()) / 2**30:.1f} GB "
              f"AdamW state in training), checkpointing={checkpointing}", flush=True)
    print(f"PROFILE speedup packed/standard: {results['packed'] / results['standard']:.2f}x", flush=True)


def plan_steps(records, epoch):
    random.Random(42 + epoch).shuffle(records)
    steps, current, n = [], [], 0
    for r in records:
        q = len(r["questions"])
        if current and n + q - STEP_QUESTIONS > STEP_QUESTIONS - n:
            steps.append(current)
            current, n = [], 0
        current.append(r)
        n += q
        if n >= STEP_QUESTIONS:
            steps.append(current)
            current, n = [], 0
    if current:
        steps.append(current)
    return steps


def mode_train(device, epochs, output_dir, checkpointing):
    base_cfg = json.load(open(LAYA / "laya_base/rl_agent_config.json"))
    cfg, tok, model = load_model(LAYA / "laya_base", device, checkpointing)
    rows, flat = load_rows(tok, base_cfg)
    train_recs, calib_recs = split_records(rows, len(flat))
    del rows, flat          # only the compact packed records are kept for training
    gc.collect()
    print(f"packed records in memory: {record_bytes(train_recs + calib_recs) / 2**20:.1f} MB", flush=True)
    window = model.encoder.config.sliding_window
    encoder_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": 2.5e-5}, {"params": head_params, "lr": 1e-4}],
                                  weight_decay=0.01)
    plans = [plan_steps(train_recs, e) for e in range(epochs)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sum(len(p) for p in plans), eta_min=1e-6)
    q_per_step = [sum(len(r["questions"]) for r in s) for p in plans for s in p]
    print(f"Device {device}; packed layout; checkpointing={checkpointing}; {len(train_recs)} records / "
          f"{sum(len(r['questions']) for r in train_recs)} questions; {len(calib_recs)} calibration records; "
          f"{len(q_per_step)} steps, questions/step mean {np.mean(q_per_step):.1f} (min {min(q_per_step)}, "
          f"max {max(q_per_step)})", flush=True)
    model.train()
    for epoch in range(epochs):
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, epochs - 1)
        t0, loss_sum, n_q, n_mb = time.perf_counter(), 0.0, 0, 0
        for step in plans[epoch]:
            denom = sum(len(r["questions"]) for r in step)
            for r in step:
                b = collate_packed([r], tok.pad_token_id, window, device)
                logits, act = packed_forward(model, b)
                loss = question_loss(logits, act, b["q_mask"], b["q_target"], b["q_qtype"], sigma,
                                     pairs_for(b["n_questions"], device)) / denom
                loss.backward()
                loss_sum += loss.item() * denom
                n_q += b["n_questions"]
                n_mb += 1
                if n_mb % 200 == 0:
                    print(f"epoch {epoch + 1}/{epochs}, record {n_mb}, loss={loss_sum / n_q:.4f}, "
                          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        print(f"Epoch {epoch + 1}/{epochs} complete; avg_loss={loss_sum / n_q:.4f}; "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        base.save_checkpoint(model, tok, cfg, output_dir, epoch + 1)

    model.eval()
    samples = [[] for _ in range(3)]
    with torch.no_grad():
        for i in range(0, len(calib_recs), 4):
            recs = calib_recs[i:i + 4]
            b = collate_packed(recs, tok.pad_token_id, window, device)
            logits, _ = packed_forward(model, b)
            qs = [q for r in recs for q in r["questions"]]
            for j, q in enumerate(qs):
                samples[q["qtype"]].append((logits[j, :len(q["markers"])].cpu(), q["target"]))
    temperatures = [base.fit_temperature(group) if group else 1.2 for group in samples]
    base.save_checkpoint(model, tok, cfg, output_dir, epochs, final=True)
    cfg.update({"fine_tuned": True, "model_name": "laya-packed", "layout": "packed", "temperature": temperatures})
    cfg.pop("temperature_by_options", None)
    for path in (Path(output_dir), Path(output_dir) / "checkpoint_latest"):
        with open(path / "rl_agent_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"Model saved to {output_dir}; temperatures {temperatures}", flush=True)


def mode_eval(device, output_dir):
    sys.path.insert(0, str(LAYA / "research/scripts"))
    import bench_local as bl
    ag = bl.laya.load(str(output_dir), device=str(device))
    ag.model.eval()
    model, tok = ag.model, ag.tok
    max_len, hml = ag.cfg.get("max_len", 512), ag.cfg.get("head_max_len", 192)
    window = model.encoder.config.sliding_window
    cases, gold, wfs = bl.build_typed_decisions()
    idx, lgs = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for ci, (state, questions) in enumerate(cases):
            entries, slots = [], []
            for qid, qdef in questions.items():
                q = bl.to_internal(qdef)
                try:
                    ids, mk = build_sequence(tok, state, q, max_len, hml)
                except Exception:
                    idx.append((ci, qid, QTYPES[q["t"]], 0)); slots.append(None); continue
                if len(mk) != len(render_options(q)):
                    idx.append((ci, qid, QTYPES[q["t"]], 0)); slots.append(None); continue
                head = build_head(tok, q, hml)[0]
                entries.append({"ids": ids, "markers": mk, "qtype": QTYPES[q["t"]], "H": len(head)})
                idx.append((ci, qid, QTYPES[q["t"]], len(mk))); slots.append(len(entries) - 1)
            out = None
            if entries:
                b = collate_packed([pack(entries)], tok.pad_token_id, window, device)
                out = packed_forward(model, b)[0].float().cpu().numpy()
            lgs += [None if s is None else out[s, :len(entries[s]["markers"])] for s in slots]
    secs = time.perf_counter() - t0
    # metric code verbatim from bench_local.run_part_b
    rows, soft, brier_s, mae = [], [], [], []
    for (ci, qid, qt, k), z in zip(idx, lgs):
        g = gold[ci][qid]
        if z is None:
            rows.append((g["idx"], None)); continue
        p = bl.softmax_t(z, bl.temp_for(ag, qt, k))
        rows.append((g["idx"], p))
        gp = np.asarray(g["soft"], float)
        if gp.sum() > 0:
            gp = gp / gp.sum()
            pp = p[:len(gp)] if len(p) >= len(gp) else np.pad(p, (0, len(gp) - len(p)))
            pp = pp / max(pp.sum(), 1e-12)
            soft.append(float((pp * gp).sum())); brier_s.append(float(((pp - gp) ** 2).sum()))
        if "gold_score" in g:
            exp = float((np.arange(len(p)) * p).sum())
            mae.append(abs(exp - g["gold_score"]))
    m = bl.metrics(rows)
    print(f"EVAL packed {output_dir}: n {m['n']} acc {m['accuracy']:.4f} soft {np.mean(soft):.4f} "
          f"brier {np.mean(brier_s):.4f} ECE {m['ece']:.4f} MAE {np.mean(mae):.4f} "
          f"dropped {sum(z is None for z in lgs)} ({secs:.0f}s)", flush=True)
    json.dump({"metrics": m, "soft_accuracy": float(np.mean(soft)), "brier_vs_soft": float(np.mean(brier_s)),
               "score_mae": float(np.mean(mae))}, open(Path(output_dir) / "eval_typed_decisions.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "profile", "train", "eval"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--output-dir", default=str(LAYA / "laya_packed_e1"))
    ap.add_argument("--checkpointing", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    torch.set_float32_matmul_precision("high")
    if a.mode == "check":
        mode_check(device)
    elif a.mode == "profile":
        mode_profile(device, checkpointing=a.checkpointing)
    elif a.mode == "train":
        mode_train(device, a.epochs, a.output_dir, a.checkpointing)
    else:
        mode_eval(device, a.output_dir)


if __name__ == "__main__":
    main()
