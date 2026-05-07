"""Zoology MQAR replication.

Verbatim port of the Multi-Query Associative Recall data generator from
HazyResearch/zoology (Arora, Eyuboglu et al., "Zoology: Measuring and
improving recall in efficient language models"). Source:
https://github.com/HazyResearch/zoology/blob/main/zoology/data/multiquery_ar.py

We use this canonical generator to validate our dense baseline against
the published 100% MQAR result. Once dense reaches 100% on this task,
we plug in clew-aria for a fair C3 comparison.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    random_non_queries: bool = True,
):
    """Verbatim port of zoology.data.multiquery_ar.multiquery_ar."""
    assert input_seq_len % 2 == 0
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 + num_kv_pairs * 2 <= input_seq_len

    rng = np.random.default_rng(seed)
    context_size = num_kv_pairs * 2

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys = np.stack([rng.choice(key_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)])
    values = np.stack([rng.choice(value_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)])

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack([rng.choice(np.arange(space, dtype=int), size=num_kv_pairs, replace=False, p=p) for _ in range(num_examples)])

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs = examples[:, :-1]
    labels = labels[:, 1:]

    if random_non_queries:
        mask = inputs == 0
        replacements = rng.integers(0, vocab_size, size=inputs.shape)
        inputs = np.where(mask, replacements, inputs)

    return torch.from_numpy(inputs), torch.from_numpy(labels)


def evaluate(model, args, device, n_examples=512):
    model.eval()
    x, y = make_mqar(
        vocab_size=args.vocab_size,
        num_examples=n_examples,
        input_seq_len=args.seq_len,
        seed=999,
        num_kv_pairs=args.num_kv_pairs,
    )
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits, _ = model(x, y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        preds = logits.argmax(-1)
        mask = y != -100
        acc = (preds[mask] == y[mask]).float().mean().item()
    model.train()
    return float(loss.item()), float(acc)


def evaluate_clew(model, args, device, n_examples=512):
    """Eval that uses our GPT shell which returns (logits, loss, aux)."""
    model.eval()
    x, y = make_mqar(
        vocab_size=args.vocab_size,
        num_examples=n_examples,
        input_seq_len=args.seq_len,
        seed=999,
        num_kv_pairs=args.num_kv_pairs,
    )
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits, _loss, _aux = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        preds = logits.argmax(-1)
        mask = y != -100
        acc = (preds[mask] == y[mask]).float().mean().item()
    model.train()
    return float(loss.item()), float(acc)


def train_nanogpt(args):
    """Train nanoGPT (reference) on Zoology MQAR. If this hits ~100%, task is well-formed."""
    import sys
    sys.path.insert(0, "eval")
    from _nanogpt_ref import GPT, GPTConfig

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = GPTConfig(
        block_size=args.seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.d_model,
        dropout=0.0,
        bias=False,
    )
    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    print(f"[nanogpt] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}")
    opt = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95), device_type=device
    )
    rng = np.random.default_rng(0)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        x, y = make_mqar(
            vocab_size=args.vocab_size,
            num_examples=args.batch_size,
            input_seq_len=args.seq_len,
            seed=int(rng.integers(0, 2**31)),
            num_kv_pairs=args.num_kv_pairs,
        )
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            ev_loss, ev_acc = evaluate(model, args, device, n_examples=512)
            elapsed = time.time() - t0
            log.append({"step": step, "loss": loss.item(), "eval_loss": ev_loss, "eval_acc": ev_acc, "elapsed": elapsed})
            print(f"[nanogpt] step {step:5d}  train-loss {loss.item():.4f}  eval-loss {ev_loss:.4f}  eval-acc {ev_acc:.2%}  {elapsed:.0f}s", flush=True)
    return log


def train_clew(kind: str, args):
    """Train clew-dense or clew-aria on Zoology MQAR using our codebase."""
    from clew.config import Config
    from clew.train import make_model

    # ARIA's Python loop runs better on CPU; dense prefers MPS
    device = "cpu" if kind == "aria" else ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config(
        block_size=args.seq_len,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        device=device,
        warmup_steps=100,
        learning_rate=args.lr,
        max_steps=args.steps,
        lsh_bits=4,
        asc_k_max=16,
        asc_n_probes=4,
        aux_router_weight=0.5,
        aux_router_until_step=4000,
    )
    cfg.vocab_size = args.vocab_size
    torch.manual_seed(0)
    model = make_model(cfg, kind).to(device)
    print(f"[{kind}] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        x, y = make_mqar(
            vocab_size=args.vocab_size,
            num_examples=args.batch_size,
            input_seq_len=args.seq_len,
            seed=int(rng.integers(0, 2**31)),
            num_kv_pairs=args.num_kv_pairs,
        )
        x, y = x.to(device), y.to(device)
        logits, _, aux = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            ev_loss, ev_acc = evaluate_clew(model, args, device, n_examples=256)
            elapsed = time.time() - t0
            log.append({"step": step, "loss": loss.item(), "eval_loss": ev_loss, "eval_acc": ev_acc, "elapsed": elapsed})
            print(f"[{kind}] step {step:5d}  train-loss {loss.item():.4f}  eval-loss {ev_loss:.4f}  eval-acc {ev_acc:.2%}  {elapsed:.0f}s", flush=True)
    return log


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--num-kv-pairs", type=int, default=8)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--model", choices=["nanogpt", "dense", "aria"], default="nanogpt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"== Zoology MQAR  model={args.model}  vocab={args.vocab_size}  T={args.seq_len}  num_kv_pairs={args.num_kv_pairs}")
    if args.model == "nanogpt":
        log = train_nanogpt(args)
    else:
        log = train_clew(args.model, args)
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/zoology_mqar_{args.model}.json").write_text(json.dumps(log, indent=2))
    print(f"final acc: {log[-1]['eval_acc']:.2%}")
