"""Sweep sequence length for both models. Log peak memory + time. Plot."""
import gc
import json
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clew.config import Config
from clew.train import make_model

LENGTHS = [256, 512, 1024, 2048, 4096, 8192]


def measure(kind, T, B=2, device="mps"):
    cfg = Config(block_size=T, batch_size=B, device=device)
    cfg.vocab_size = 65
    if device == "mps":
        torch.mps.empty_cache()
    gc.collect()
    model = make_model(cfg, kind).to(device)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    if device == "mps":
        torch.mps.empty_cache()
        # PyTorch doesn't expose peak memory on MPS reliably; use driver_allocated_memory
        before = torch.mps.driver_allocated_memory()
    t0 = time.time()
    try:
        logits, loss, aux = model(x, x)
        loss.backward()
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        peak = (torch.mps.driver_allocated_memory() - before) / 1e6 if device == "mps" else 0.0
        # also track end-of-step allocated
        after = torch.mps.driver_allocated_memory() / 1e6 if device == "mps" else 0.0
        ok = True
    except (RuntimeError, MemoryError) as e:
        dt = float("nan")
        peak = float("nan")
        after = float("nan")
        ok = False
        print(f"  {kind} T={T} OOM: {e}")
    finally:
        del model, x
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    return {"kind": kind, "T": T, "ok": ok, "time_s": dt, "delta_mb": peak, "alloc_mb": after}


def run():
    results = []
    for T in LENGTHS:
        for kind in ("dense", "aria"):
            print(f"measuring {kind} T={T}")
            r = measure(kind, T)
            print(f"  -> {r}")
            results.append(r)
    out = Path("runs/memory_scan.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (("dense", "o"), ("aria", "s")):
        Ts = [r["T"] for r in results if r["kind"] == kind and r["ok"]]
        mb = [r["alloc_mb"] for r in results if r["kind"] == kind and r["ok"]]
        ax.plot(Ts, mb, marker=marker, label=kind)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("MPS allocated memory (MB)")
    ax.set_title("Memory vs sequence length")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/memory.png", dpi=120)
    print("saved plots/memory.png")


if __name__ == "__main__":
    run()
