"""Needle-in-haystack evaluation.

NOTE: a 1M-param char-level Shakespeare model trained for 1000 steps cannot
'understand' a planted English fact. So we use a *char-level needle*:
embed a unique short marker substring at a random position, then test
whether the model's hash-attention selects positions overlapping the
needle when queried at the end. This measures the underlying retrieval
mechanism (which is what the trifecta cares about), independent of the
language head.
"""
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from clew.config import Config
from clew.train import make_model

LENGTHS = [256, 512, 1024, 2048, 4096]


def _load_aria(device):
    ck = torch.load("runs/aria/ckpt.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    cfg.device = device
    model = make_model(cfg, "aria").to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def retrieval_at(model, cfg, T, n_trials=20, needle_len=8, device="mps"):
    """Plant a marker substring; check whether ARIA's candidate set at the last
    position contains positions inside the planted needle for at least one head.

    Context = real (in-distribution) Shakespeare bytes so the learned hash
    behaves as it does at training time.
    """
    V = cfg.vocab_size
    # load Shakespeare token stream
    train = np.memmap(Path(cfg.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(n_trials):
        start = rng.integers(0, len(train) - T - 1)
        ctx = train[start : start + T].astype(np.int64).copy()
        # plant a needle: a fixed token pattern at a random position
        np_idx = rng.integers(needle_len, T - needle_len)
        needle = np.array([7, 11, 13, 17, 19, 23, 29, 31][:needle_len], dtype=np.int64)
        ctx[np_idx : np_idx + needle_len] = needle
        # query: same needle at the end (so attention should retrieve the planted needle)
        ctx[-needle_len:] = needle

        x = torch.from_numpy(ctx).unsqueeze(0).to(device)

        # Forward through the first ARIA attention block manually to inspect
        # candidate sets for the last query position.
        # We'll use just one block to keep this straightforward.
        # cfg.block_size is the trained context; if T > block_size, wrap pos emb.
        T_eff = x.shape[1]
        pos = torch.arange(T_eff, device=device) % cfg.block_size
        h = model.tok_emb(x) + model.pos_emb(pos)
        block = model.blocks[0]
        # mimic AriaAttention.forward but capture cand
        x_in = block.ln1(h)
        attn = block.attn
        B, Tt, C = x_in.shape
        H, D = attn.n_head, attn.d_head
        q, k, v = attn.qkv(x_in).split(C, dim=-1)
        q = q.view(B, Tt, H, D).transpose(1, 2)
        k = k.view(B, Tt, H, D).transpose(1, 2)
        codes_q = torch.stack([attn.hashes[i].codes_int(q[:, i]) for i in range(H)], 1)
        codes_k = torch.stack([attn.hashes[i].codes_int(k[:, i]) for i in range(H)], 1)
        cand = attn._candidate_indices(codes_q, codes_k)  # (B,H,T,K)
        last_cands = cand[0, :, -1, :].cpu().numpy()  # (H, K)

        # success: any head retrieves any position inside the planted needle
        needle_positions = set(range(np_idx, np_idx + needle_len))
        any_hit = False
        for h_idx in range(H):
            for c in last_cands[h_idx]:
                if int(c) in needle_positions:
                    any_hit = True
                    break
            if any_hit:
                break
        if any_hit:
            hits += 1
    return hits / n_trials


def run():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg = _load_aria(device)
    print(f"loaded aria, block_size={cfg.block_size}, n_head={cfg.n_head}")
    results = []
    for T in LENGTHS:
        if T > cfg.block_size * 32:
            print(f"skip T={T} (too far past trained context)")
            continue
        acc = retrieval_at(model, cfg, T, n_trials=20, device=device)
        print(f"T={T:>6d}  retrieval-acc={acc:.2%}")
        results.append({"T": T, "acc": acc})
    Path("runs").mkdir(exist_ok=True)
    Path("runs/needle.json").write_text(json.dumps(results, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts = [r["T"] for r in results]
    accs = [r["acc"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(t) for t in Ts], accs, color="#c44536")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("ARIADNE: needle retrieval (hash-bucket hit rate)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/niah.png", dpi=120)
    print("saved plots/niah.png")


if __name__ == "__main__":
    run()
