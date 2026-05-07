"""Associative recall (MQAR): a synthetic task that *requires* retrieval.

Layout per sequence: [k1 v1 k2 v2 ... kn vn]  followed by  [kq1 vq1 kq2 vq2 ...].
Loss is taken at every query-key position (predict the matching value).

Intent: isolate C3 (exact retrieval) as the actual training objective, so we
can compare clew-dense vs clew-aria on a task that *forces* induction heads.

v0 status: INCONCLUSIVE. Our dense baseline plateaus at ~30% accuracy on
n_pairs=4 MQAR across many configs (up to 2.68M params, 8000 steps). The
Mamba paper's published dense baseline reaches 100% on the same task, so
something in our setup (init, residual scaling, hparams, or a subtle bug)
is preventing induction heads from forming. Until dense solves it, ARIA's
number on this task is uninformative.

Right follow-up: replicate the published Mamba MQAR config verbatim
(arch + init + hparams) to get a working dense baseline, *then* plug
ARIA in. Keeping this file in the repo as a starting point for that work.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clew.config import Config
from clew.train import make_model


N_KEYS = 32
N_VALS = 32
VOCAB_SIZE = N_KEYS + N_VALS  # 64


def gen_batch(batch_size: int, n_pairs: int, rng: np.random.Generator, n_queries: int = None):
    """Build a batch of MQAR (Multi-Query Associative Recall) sequences.

    First, lay down `n_pairs` distinct (key, value) pairs as a kv-table prefix:
        [k1 v1 k2 v2 ... kn vn]
    Then append `n_queries` queries (default = n_pairs), each of which is one of
    the keys followed by its expected value:
        [... kq1 vq1 kq2 vq2 ...]

    Loss is taken at every query-key position (predict the matching value).
    This gives n_queries gradient signals per sequence — enough for induction
    heads to form — while still requiring true retrieval.

    Returns (x, y_query, query_positions).
    """
    if n_queries is None:
        n_queries = n_pairs
    seq_len = 2 * (n_pairs + n_queries)
    x = np.zeros((batch_size, seq_len), dtype=np.int64)
    y = np.full((batch_size, seq_len), -100, dtype=np.int64)
    for b in range(batch_size):
        keys = rng.choice(N_KEYS, size=n_pairs, replace=False)
        vals = rng.integers(0, N_VALS, size=n_pairs) + N_KEYS
        # kv-table prefix
        for i in range(n_pairs):
            x[b, 2 * i] = keys[i]
            x[b, 2 * i + 1] = vals[i]
        # queries (uniformly sampled keys from the table, with replacement)
        qis = rng.integers(0, n_pairs, size=n_queries)
        for j, qi in enumerate(qis):
            kpos = 2 * n_pairs + 2 * j
            x[b, kpos] = keys[qi]
            x[b, kpos + 1] = vals[qi]
            y[b, kpos] = vals[qi]  # predict val from key position
    query_positions = list(range(2 * n_pairs, seq_len, 2))
    return torch.from_numpy(x), torch.from_numpy(y), query_positions


def evaluate(model, n_pairs, device, n_batches=10, batch_size=64):
    """Return (loss, accuracy) at the query position."""
    model.eval()
    rng = np.random.default_rng(7)
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, qps = gen_batch(batch_size, n_pairs, rng)
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            # eval over all query positions
            qp_t = torch.tensor(qps, device=device)
            preds = logits[:, qp_t, :].argmax(-1)  # (B, Q)
            tgts = y[:, qp_t]  # (B, Q)
            mask = tgts != -100
            correct += (preds == tgts).masked_select(mask).sum().item()
            total += mask.sum().item()
            l = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
            )
            losses.append(l.item())
    model.train()
    return float(np.mean(losses)), correct / total


def train_one(kind: str, n_pairs: int = 30, steps: int = 2000, batch_size: int = 64,
              device: str = None, lr: float = 3e-4, log_every: int = 100):
    """Train a fresh model on associative recall and return its eval log."""
    device = device or ("cpu" if kind == "aria" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    seq_len = 4 * n_pairs  # n_pairs kv-table + n_queries query-pairs
    cfg = Config(
        block_size=seq_len,
        batch_size=batch_size,
        n_layer=4,
        n_head=4,
        d_model=128,
        max_steps=steps,
        learning_rate=lr,
        warmup_steps=100,
        device=device,
        lsh_bits=16,
        asc_k_max=16,
        asc_n_probes=4,
        aux_router_weight=0.0,  # let the task itself drive the hash
        aux_router_until_step=0,
    )
    cfg.vocab_size = VOCAB_SIZE
    torch.manual_seed(0)
    model = make_model(cfg, kind).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{kind}] params={n_params/1e6:.2f}M  device={device}  seq_len={seq_len}  n_pairs={n_pairs}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    log = []
    t0 = time.time()
    for step in range(steps):
        x, y, _qps = gen_batch(batch_size, n_pairs, rng)
        x, y = x.to(device), y.to(device)
        logits, _, aux = model(x)
        # MQAR: loss at every query position (n_queries per sequence)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            ev_loss, ev_acc = evaluate(model, n_pairs, device, n_batches=4, batch_size=batch_size)
            elapsed = time.time() - t0
            log.append({"step": step, "loss": loss.item(), "eval_loss": ev_loss, "eval_acc": ev_acc, "elapsed": elapsed})
            print(f"[{kind}] step {step:4d}  train-loss {loss.item():.4f}  eval-loss {ev_loss:.4f}  eval-acc {ev_acc:.2%}  {elapsed:.0f}s")
    final_loss, final_acc = evaluate(model, n_pairs, device, n_batches=20, batch_size=batch_size)
    print(f"[{kind}] FINAL  loss {final_loss:.4f}  acc {final_acc:.2%}")
    return {"kind": kind, "n_pairs": n_pairs, "log": log, "final_loss": final_loss, "final_acc": final_acc}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"== associative recall  n_pairs={args.n_pairs}  steps={args.steps}  batch_size={args.batch_size}")
    print(f"   chance accuracy = 1/{N_VALS} = {1/N_VALS:.2%}")
    print()

    results = []
    for kind in ("dense", "aria"):
        r = train_one(kind, n_pairs=args.n_pairs, steps=args.steps, batch_size=args.batch_size)
        results.append(r)
        print()

    # write report
    Path("runs").mkdir(exist_ok=True)
    import json
    Path("runs/assoc_recall.json").write_text(json.dumps(results, indent=2))

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, color in zip(results, ("#1a1a2e", "#c44536")):
        steps = [e["step"] for e in r["log"]]
        accs = [e["eval_acc"] for e in r["log"]]
        ax.plot(steps, accs, marker="o", color=color, label=f"{r['kind']} (final {r['final_acc']:.0%})")
    ax.axhline(1 / N_VALS, color="gray", linestyle=":", label=f"chance ({1/N_VALS:.0%})")
    ax.set_xlabel("step")
    ax.set_ylabel("query-position accuracy")
    ax.set_title(f"Associative recall: dense vs aria  (n_pairs={args.n_pairs}, T={2*(args.n_pairs+1)})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/assoc_recall.png", dpi=120)
    print("saved plots/assoc_recall.png")
    print()
    print("== summary ==")
    for r in results:
        print(f"  {r['kind']:5s}  final acc {r['final_acc']:.2%}  loss {r['final_loss']:.4f}")


if __name__ == "__main__":
    run()
