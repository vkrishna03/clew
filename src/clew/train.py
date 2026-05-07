"""Shared training loop for dense and aria."""
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .model_dense import GPT, DenseAttention


def get_batch(split, cfg: Config, device):
    data_dir = Path(cfg.data_dir)
    f = data_dir / ("train.bin" if split == "train" else "val.bin")
    data = np.memmap(f, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, cfg: Config):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, cfg, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(split, cfg, device)
            _, loss, _ = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def make_model(cfg: Config, kind: str):
    if kind == "dense":
        return GPT(cfg, attn_cls=DenseAttention)
    elif kind == "aria":
        from .model_aria import AriaAttention
        return GPT(cfg, attn_cls=AriaAttention)
    else:
        raise ValueError(kind)


def train(cfg: Config, kind: str):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # load vocab
    with open(Path(cfg.data_dir) / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    cfg.vocab_size = meta["vocab_size"]

    device = torch.device(cfg.device if torch.backends.mps.is_available() or cfg.device == "cpu" else "cpu")
    model = make_model(cfg, kind).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{kind}] params: {n_params/1e6:.2f}M  device: {device}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    out_dir = Path(cfg.out_dir) / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    model.train()
    for step in range(cfg.max_steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch("train", cfg, device)
        logits, loss, aux = model(x, y)
        # aux router weight schedule
        if kind == "aria" and step < cfg.aux_router_until_step:
            total = loss + cfg.aux_router_weight * aux
        else:
            total = loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 50 == 0 or step == cfg.max_steps - 1:
            elapsed = time.time() - t0
            print(f"[{kind}] step {step:4d}  loss {loss.item():.4f}  aux {float(aux):.4f}  lr {lr:.2e}  {elapsed:.1f}s")

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            ev = estimate_loss(model, cfg, device)
            print(f"[{kind}] eval step {step:4d}  train {ev['train']:.4f}  val {ev['val']:.4f}")
            log.append({"step": step, "train": ev["train"], "val": ev["val"], "loss": loss.item()})

    # save
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_dir / "ckpt.pt")
    with open(out_dir / "log.pkl", "wb") as f:
        pickle.dump(log, f)
    print(f"[{kind}] done in {time.time()-t0:.1f}s. saved to {out_dir}")
    return model, log
