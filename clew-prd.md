# PRD: Clew

**Building and validating the subquadratic trifecta on an 8GB M3 MacBook**

> A Karpathy-style "let's build it from scratch" implementation of the **ARIADNE** attention architecture (codenamed **Clew**), run side-by-side against a dense-attention baseline on tinyshakespeare, with a long-context stress test that's the actual point of the exercise.

---

## Naming

- **ARIADNE** — the formal architecture name (used in the paper, citations, conference talks). Stands for *Associative Retrieval via Indexed Anchored Differentiable Neural Embeddings*.
- **Clew** — the project / repo / CLI name. In Greek myth, the *clew* is the ball of thread Ariadne gives Theseus to find his way out of the Labyrinth. It's also the etymological origin of the English word "clue" — fitting for a system that helps a model find the right tokens in a sea of past context.
  - Pronounced "clue."
  - Repo: `clew`. Domain candidate: `clew.dev`. CLI: `clew train`, `clew eval`.
  - Rejected alternatives: *Aria* (collides with HTML ARIA accessibility standards — unsearchable), *Skein* (collides with Skein cryptographic hash function), *Theseus* (collides with TheseusOS), *Daedalus* (collides with Cardano wallet).

---

## 0. TL;DR (For Anyone Skimming)

We're going to build **two miniature GPTs from scratch** in a single Python repo:

1. **`clew-dense`** — a faithful nanoGPT clone. Standard quadratic attention. The control.
2. **`clew-aria`** — same model, same training loop, same data. Only the attention mechanism is swapped for ARIADNE's hash-cascade lookup.

Both train on Shakespeare. Both produce text. We measure three things:

- **Quality.** Do they reach the same training loss and produce comparable text?
- **Memory.** Where does dense OOM? Where does Clew keep running?
- **Recall.** On a needle-in-a-haystack test, can Clew actually retrieve specific facts from arbitrarily long contexts, the way the theory predicts?

If `clew-aria` matches dense quality at short contexts AND keeps running at sequence lengths where dense crashes AND retrieves correctly — that's a strong public signal that the trifecta is real. If it fails any of those three, we know exactly which assumption broke and we've added a real data point to a debate currently dominated by closed-source claims.

The whole thing fits on an 8GB M3, runs in hours not weeks, and the codebase is small enough that a researcher or coding agent can read it end-to-end.

---

## 1. Problem & Why It Matters

### 1.1 The technical problem (for researchers)

Long-context language models hit three fundamental walls and current solutions only escape two of them at a time:

- **C1 — Subquadratic scaling, end to end, no hidden N²** anywhere (including in the router or indexer)
- **C2 — Content-dependent token selection** (the model decides what to look at, not a fixed pattern)
- **C3 — Exact retrieval from arbitrary positions** (no compression of the past into a fixed-size state)

Dense attention has C2+C3 but fails C1. Mamba/RWKV/linear-attention have C1+C2 but fail C3. Sliding-window / fixed-pattern sparse have C1+C3 but fail C2. Modern learned-sparse (NSA, MoBA, DSA) appears to have all three but actually fails C1 because the router itself is N². No published architecture has all three at frontier scale.

ARIADNE proposes that a **learned hierarchical hash-index over keys (the Associative Skip-Cascade)** lets a query find its top-k relevant past tokens in O(log N), enabling exact attention over only that bounded set. If the recall holds and training is stable, all three properties can be satisfied simultaneously.

### 1.2 The market problem (for CEO / investor)

Subquadratic Inc.'s SubQ launch (May 2026) claims a 12M-token model that achieves the trifecta. They have not released a paper or weights. The community is split between "biggest breakthrough since the Transformer" and "AI Theranos." DeepSeek V3.2's DSA shows sparse attention works in production but only escapes two of three (its router is still quadratic). MiniMax recently abandoned pure linear attention for M2.5, returning to dense — a strong negative signal that fixed-state compression breaks at scale.

**Clew** produces an open, falsifiable, single-file reference implementation of the architecture family that solves the trifecta. Useful as:

- A research artifact that validates or kills the design
- A teaching tool for the next year of long-context work
- Grounded credibility on the topic — we've actually built one and have data, not opinions
- Foundation for a serious v1 implementation if v0 results are positive

### 1.3 Audience for this document

| Section | CEO / Investor | Tech Lead / Researcher | Coding Agent / Implementer |
|---|---|---|---|
| §0–2 | ✅ | ✅ | ✅ |
| §3 Goals & Success | ✅ | ✅ | ✅ |
| §4 Scope | ✅ | ✅ | ✅ |
| §5 Architecture | skim | ✅ | ✅ |
| §6 Implementation Plan | skip | ✅ | ✅ |
| §7 Test Plan | skim | ✅ | ✅ |
| §8 Hardware Budget | ✅ | ✅ | ✅ |
| §9 Risks | ✅ | ✅ | skim |
| §10 Roadmap | ✅ | ✅ | skim |
| §11 Glossary | ✅ | skim | skim |
| §12 Open Questions | skip | ✅ | ✅ |
| §13 References | skip | ✅ | ✅ |
| §14 Public Eval Platforms | skim | ✅ | ✅ |
| §15 HTML Landing Page Spec | ✅ | ✅ | ✅ |
| §16 CEO Cheatsheet | ✅ | skip | skip |

---

## 2. Background

If you haven't already read the ARIADNE paper draft, the **two-line summary**:

Build a learned, hierarchical, content-addressable index of keys as the model reads. When a query needs to look something up, walk down the index in O(log N) hops to find a small candidate set, then run normal exact attention over just those candidates. The full keys and values are never compressed; the index is the only thing being approximated.

**Why this is plausibly different from past failed attempts (Reformer, Memorizing Transformers):**

- The hash function is learned end-to-end, not random
- The index is multi-resolution (skip-cascade), not flat
- The base layer keeps bit-exact KV pairs — no compression
- Cross-layer index sharing reduces overhead by a large constant factor
- We have lessons from 4+ years of subsequent retrieval-augmented work to incorporate

**Why this might still fail:** hash drift during training, GPU-unfriendly memory access patterns, training instability, the auxiliary router loss not converging. We list and design tests for each of these.

---

## 3. Goals & Success Criteria

### 3.1 What success looks like

This is research code, not a product. "Success" means **conclusive evidence one way or the other**.

#### Hard requirements

| ID | Requirement | Measurement |
|---|---|---|
| H1 | Both models train to convergence on tinyshakespeare without crashing | Train loss < 1.5 char-level cross-entropy after 5000 steps |
| H2 | clew-aria matches clew-dense val loss within +0.05 nats at 256-token context | Held-out val cross-entropy |
| H3 | Generated text from clew-aria is qualitatively comparable to clew-dense | Side-by-side samples, human read |
| H4 | Both fit and train on an 8GB M3 in under 6 hours each | Wall-clock + Activity Monitor |
| H5 | Total repo is < 1500 lines of Python, single-file friendly | `wc -l` |

#### Stretch goals (where the actual research signal lives)

| ID | Goal | Measurement | Stretch level |
|---|---|---|---|
| S1 | clew-dense OOMs at some sequence length L_dense; clew-aria keeps training at 4×L_dense | Crash point + memory profile | **Critical for the demo** |
| S2 | Needle-in-haystack: Clew retrieves a planted fact at 32k tokens with >90% accuracy | Custom synthetic eval | **Critical for the demo** |
| S3 | Needle-in-haystack at 128k tokens with >80% accuracy | Same eval | High |
| S4 | LSH miss rate < 5% on standard ANN benchmarks (with our learned hash) | Recall@k vs ground-truth top-k | High |
| S5 | clew-aria wall-clock faster than clew-dense at 8k+ token contexts | Tokens/sec | Medium |
| S6 | Reproducible: anyone can clone and re-run on their own M3 in one command | `make all` works | High |

### 3.2 Explicit non-goals

- **Not training a frontier model.** This is a 1–10M-parameter educational implementation.
- **Not building production infrastructure.** No FlashAttention-class custom kernels. No distributed training.
- **Not aiming for SOTA on any standard benchmark.** Shakespeare is not MMLU.
- **Not claiming to disprove or replicate SubQ.** SubQ is closed; we have no idea what they're doing internally.

---

## 4. Scope

### 4.1 In scope (v0)

- Two single-file PyTorch model implementations (dense + ARIADNE)
- Shared training loop, data loader, tokenizer
- Learned-LSH module
- Simple flat-bucket index with multi-probe (the simplified ASC for v0)
- Auxiliary router loss
- Three eval scripts: train/val loss curves, memory profiler, needle-in-haystack
- README with reproducibility instructions
- Plots: loss curves, memory vs seq_len, recall vs seq_len
- Static HTML landing page (see §15)

### 4.2 Out of scope (v0)

- HNSW proximity graph (deferred to v0.5)
- Cross-layer index sharing (deferred to v0.5)
- Triton or Metal-Performance-Shaders custom kernels
- BPE tokenization (char-level keeps things simple, matches Karpathy)
- Multi-GPU anything
- Inference-time KV cache optimization
- Go port of the ASC

### 4.3 Punted to v1+

- Go-based ASC index for production speed
- Real long-context dataset (PG19, code, etc.)
- Larger models (100M+ params, won't fit on 8GB)
- Custom CUDA / Metal kernels
- Distributed training

---

## 5. Architecture

### 5.1 The two models, side by side

Both are nanoGPT-shaped:

```
Input tokens (B, T)
      ↓
Token + Position Embeddings (B, T, d_model)
      ↓
[N transformer blocks]:
  ├─ LayerNorm
  ├─ Attention  ← ONLY THIS DIFFERS
  ├─ Residual
  ├─ LayerNorm
  ├─ MLP (4× expansion, GELU)
  └─ Residual
      ↓
Final LayerNorm
      ↓
LM Head (tied weights with embedding)
      ↓
Logits (B, T, vocab_size)
```

Only the attention block differs. Everything else (data, optimizer, init, LR schedule, training loop) is identical, controlled by the same config.

### 5.2 Dense attention (the baseline)

Standard scaled dot-product. Causal mask. Multi-head. `F.scaled_dot_product_attention` for speed.

```python
# Q · Kᵀ → softmax → · V, with causal mask. Quadratic in T.
attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 5.3 ARIADNE attention (the contender)

Three stages. Key insight: **exact attention over a small candidate set, where the candidate set is chosen by content.**

```
For each query q at position t:
  1. Hash:    h_q = phi(q)        # learned LSH
  2. Lookup:  S = ASC.descend(h_q) # O(log N) walk down the index, returns ~k=64 candidate positions
  3. Attend:  out = softmax(q · K[S]ᵀ / √d) · V[S]   # exact attention over just S
```

#### 5.3.1 The Associative Skip-Cascade (ASC), v0 simplified

For v0 we use **flat LSH buckets with multi-probe**, not the full HNSW. Closer to Reformer architecturally, but with two key differences:

- **Learned** hash function (not random projection)
- **Multi-probe** descent that checks neighboring buckets

```python
class ASC:
    buckets: dict[int, list[int]]  # bucket_id → [token positions]
    keys: torch.Tensor             # (T, d) all keys, append-only, bit-exact
    values: torch.Tensor           # (T, d) all values, append-only, bit-exact

    def insert(self, t, k_t, v_t):
        bucket_id = self.phi(k_t)
        self.buckets[bucket_id].append(t)
        self.keys[t] = k_t
        self.values[t] = v_t

    def lookup(self, q, k_max=64, n_probes=4):
        h = self.phi(q)
        candidates = []
        for h_probe in [h] + flip_random_bits(h, n_probes - 1):
            candidates.extend(self.buckets[h_probe])
        candidates = list(set(candidates))[-k_max:]  # respect causal: t' < t
        return candidates
```

**Why this v0 simplification is enough:** if learned LSH alone gets us reasonable recall on Shakespeare-scale, the architecture is plausible. If it doesn't, no amount of HNSW polish will save it. The skip-cascade is a constant-factor improvement, not an algorithmic enabler.

**Why this is fully sublinear:** insertion is O(1) amortized, lookup is O(n_probes · k_max), no scan over T anywhere.

#### 5.3.2 Learned hash function `phi`

```python
class LearnedLSH(nn.Module):
    def __init__(self, d_model, n_bits=64):
        super().__init__()
        self.W = nn.Linear(d_model, n_bits, bias=False)
        self.tau = 0.5  # annealed during training

    def forward(self, x):
        # Forward: hard sign. Backward: tanh surrogate (straight-through).
        logits = self.W(x) / self.tau
        if self.training:
            soft = torch.tanh(logits)
            hard = (logits > 0).float() * 2 - 1
            return hard + (soft - soft.detach())   # STE
        else:
            return (logits > 0).int()
```

#### 5.3.3 Auxiliary router loss

The model needs the hash to put truly-attended-to keys in the same (or nearby) buckets. We add a teacher loss during the first few thousand steps: compute dense attention on 5% of batches (cheap at small T), use it as a signal to align hash codes. Detach the dense computation so gradients flow only through `phi`.

### 5.4 Why we're confident this fits in 8GB

For T = 8192 tokens, d_model = 192, 6 layers, 6 heads:

- Model params: ~10M → 40MB at fp32
- Activations during training:
  - Dense: O(B · L · H · T²) → ~38 GB at B=4. **OOMs hard.**
  - Clew: O(B · L · H · T · k) with k=64 → ~300 MB. **Comfortable.**
- KV cache: 2 · 6 · 8192 · 192 · 4 bytes ≈ 75 MB. Fine.
- Index buckets: ~1MB. Fine.

This is exactly the demo: dense crashes at 8k context, Clew keeps going. We measure the actual crash point empirically.

---

## 6. Implementation Plan

### 6.1 Tooling: `uv` package manager

We use **`uv`** (Astral, Rust-based, ~10× faster than pip) for environment and dependency management. Same team as `ruff`. If you don't have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

Project setup:

```bash
git clone https://github.com/<you>/clew.git && cd clew
uv sync                          # creates .venv, installs from pyproject.toml + uv.lock
uv run python data/prepare.py   # downloads tinyshakespeare, tokenizes
uv run clew train --model dense  # train the baseline
uv run clew train --model aria   # train ARIADNE
uv run clew eval needle          # run NIAH benchmark
uv run clew eval memory          # memory scan
```

`pyproject.toml`:

```toml
[project]
name = "clew"
version = "0.1.0"
description = "ARIADNE: subquadratic trifecta attention, from scratch"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "numpy",
    "matplotlib",
    "tqdm",
    "tiktoken",         # only if BPE in v0.5
    "pytest",
]

[project.scripts]
clew = "clew.cli:main"

[tool.uv]
dev-dependencies = ["pytest", "ruff"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

No `requirements.txt`. `uv.lock` is committed for reproducibility.

### 6.2 Repo layout

```
clew/
├── README.md                    # how to run, expected outputs
├── Makefile                     # one-command reproducibility
├── pyproject.toml               # uv-managed
├── uv.lock                      # locked deps
├── index.html                   # landing page (see §15)
├── data/
│   ├── prepare.py               # download tinyshakespeare, tokenize
│   ├── input.txt                # raw shakespeare (gitignored)
│   └── meta.pkl                 # vocab
├── src/clew/
│   ├── __init__.py
│   ├── cli.py                   # the `clew` entry point
│   ├── config.py                # @dataclass Config — single source of truth
│   ├── model_dense.py           # ~200 lines, nanoGPT-faithful baseline
│   ├── model_aria.py            # ~400 lines, ARIADNE attention
│   ├── lsh.py                   # ~80 lines, learned hash + STE
│   ├── asc.py                   # ~150 lines, bucket index + lookup
│   ├── train.py                 # shared training loop
│   └── sample.py                # generation
├── eval/
│   ├── needle_haystack.py       # synthetic NIAH benchmark
│   ├── memory_scan.py           # increase seq_len until OOM
│   ├── recall_at_k.py           # LSH recall vs dense top-k ground truth
│   └── ruler_lite.py            # mini-RULER protocol on our model (see §14)
├── tests/
│   ├── test_lsh.py
│   ├── test_asc.py
│   └── test_models.py
├── plots/                       # generated
└── runs/                        # generated, gitignored
```

Total target: **~1200 lines** of Python.

### 6.3 Build order

Each phase produces a runnable, testable artifact. **Don't move on until the previous phase passes its acceptance check.**

**Phase 0 — Skeleton & data (~2h)**
1. `uv init clew`, set up `pyproject.toml`
2. `data/prepare.py` — download tinyshakespeare, char-level tokenize, train/val split
3. `src/clew/config.py` — config dataclass
4. ✅ `uv run python data/prepare.py` produces `train.bin`, `val.bin`, `meta.pkl`

**Phase 1 — Dense baseline (~3h)**
5. `model_dense.py` — copy nanoGPT, simplify, MPS-ready
6. `train.py` — unified loop
7. Train 5000 steps, T=256, on M3 MPS
8. `sample.py` — generate 500 chars
9. ✅ Train loss < 1.5, generations look Shakespeare-y. **Save as gold standard.**

**Phase 2 — LSH primitive (~2h)**
10. `lsh.py` — LearnedLSH with STE
11. Unit test: synthetic vectors, train hash to cluster nearby vectors, measure recall@10
12. ✅ Recall@10 > 80% after 1000 hash-training steps

**Phase 3 — ASC index (~3h)**
13. `asc.py` — bucket dict, insert, multi-probe lookup
14. Unit test: insert 10k vectors, query for known nearest
15. ✅ Recall@k=64 with n_probes=4 > 70% on synthetic data

**Phase 4 — ARIADNE attention block (~4h)**
16. `model_aria.py` — attention replaced
17. Per-layer per-head ASC during forward pass; reset between batches
18. Train 5000 steps, T=256, identical config to Phase 1
19. ✅ Train loss within 0.05 of dense; generations comparable. **Main milestone.**

**Phase 5 — Memory scan (~2h)**
20. `eval/memory_scan.py` — sweep T ∈ [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, ...]
21. Plot memory(T) for both models
22. ✅ Dense OOMs by T=16k, Clew keeps going

**Phase 6 — Needle-in-haystack (~3h)**
23. `eval/needle_haystack.py` — plant facts at random positions, ask at end
24. Test at T = 1k, 4k, 16k, 64k where applicable
25. ✅ Clew ≥ 90% @ 16k, ≥ 80% @ 64k

**Phase 7 — Polish (~3h)**
26. README with reproduction instructions
27. Final plots
28. `index.html` landing page (see §15)
29. ✅ Stranger can clone, run `make all`, reproduce in < 6h on M3

**Total estimated time: ~22h focused work.** Coding agent: faster. Human with debugging: 3–5 days.

### 6.4 Critical implementation details

#### MPS gotchas on M3
- Use `torch.float16` cautiously — MPS support is uneven. Default to fp32, switch to bf16 only after correctness is verified.
- Some ops fall back to CPU silently. Profile to confirm MPS is being used.
- `F.scaled_dot_product_attention` works on MPS in recent PyTorch — verify version.

#### The ASC index during training
- It's not a `nn.Module` — regenerated each forward pass from current keys
- Insertion happens *during* the forward pass, in causal order
- Lookup at position t can only see positions < t — enforce explicitly
- Buckets stay as Python dicts on CPU. Only keys/values tensors live on MPS.
- This is slow in v0. That's fine. We're validating correctness.

#### Gradient flow
- Through `phi(q)` via STE
- Through `K[S]` and `V[S]` (these are sliced from real K, V tensors — autograd handles it)
- NOT through the choice of S (set membership). Auxiliary router loss compensates.

#### Hash drift
- Hash updates each step but the bucket index from the previous batch is stale
- v0: **rebuild the index from scratch every forward pass.** Slow but correct.
- v0.5: rebuild every K batches.
- v1: amortized incremental updates.

#### Random seeds
- Set seeds for: torch, numpy, python random, MPS
- LSH initialization is sensitive — report 3-seed averages.

#### Don't optimize prematurely
- Python dicts for buckets in v0
- `torch.gather` for candidate selection
- No custom kernels
- Profile only after Phase 6 acceptance

---

## 7. Test Plan

### 7.1 Unit tests

```python
# tests/test_lsh.py
def test_lsh_recall_after_training()      # trained LSH clusters nearby vectors

# tests/test_asc.py
def test_asc_insert_lookup()              # inserted vectors are findable
def test_asc_causal_mask()                # position t cannot see positions ≥ t

# tests/test_models.py
def test_dense_forward_shapes()
def test_aria_forward_shapes()
def test_aria_grad_flow()                 # gradients reach all params including phi
```

### 7.2 Integration tests

| Test | Setup | Pass criterion |
|---|---|---|
| Dense overfits a single batch | T=64, batch of 4, 200 steps | Loss < 0.1 |
| ARIADNE overfits a single batch | Same | Loss < 0.15 (looser; LSH adds noise) |
| Both produce coherent text after 5k steps | T=256, full Shakespeare | Human-readable Shakespeare-flavored output |

### 7.3 The actual experiments

#### E1 — Loss parity at short context
Train both, same config, T=256, 5000 steps, 3 seeds. Plot train + val loss. Pass: ARIADNE final val loss ≤ dense + 0.05 nats.

#### E2 — Memory scan
For T ∈ [256, 1k, 4k, 16k, 64k, 256k, 1M]: one forward+backward step on each. Log peak MPS memory, peak Python memory, time. Plot log(memory) vs log(T). Expected: dense quadratic, Clew near-linear.

#### E3 — Needle in haystack
100 needles per length: "The secret password is XYZ123" embedded in random Shakespeare context. Question at end. Score: exact-match. Pass: Clew ≥ 90% @ 16k, ≥ 80% @ 64k.

#### E4 — Ablations
- LSH bits m ∈ {16, 32, 64, 128}
- Multi-probe r ∈ {1, 2, 4, 8}
- Auxiliary router loss on/off
- Hash rebuild frequency ∈ {every step, every 10, every 100}

#### E5 — Recall@k vs dense ground truth
At T=2048, 100 random queries. Dense top-64 = ground truth. Compare against ARIADNE candidate set. Pass: Recall@64 ≥ 75%.

### 7.4 What ships in the README

- Loss curves (E1)
- Memory plot (E2)
- NIAH bar chart (E3)
- Sample text from both models, side by side
- One-line summary: *"Clew matches dense at short context within X nats, runs at Y× longer sequences before OOM, and retrieves needles at length Z with accuracy W%."*

---

## 8. Hardware & Performance Budget

**Target machine:** M3 MacBook Air or Pro, 8GB unified memory, macOS 14+.

### 8.1 Memory budget (8GB total)

| Consumer | Budget |
|---|---|
| OS + browser + IDE | ~3.5 GB |
| Python runtime + PyTorch | ~1 GB |
| Model parameters (10M @ fp32) | 40 MB |
| Optimizer state (AdamW: 2× params) | 80 MB |
| **Activations (varies by T)** | **The interesting variable** |
| KV cache during training | ~100 MB at T=8k |
| ASC index (Python dicts, CPU) | ~50 MB at T=8k |
| Headroom for spikes | ~500 MB |
| **Available for activations** | **~3 GB** |

3 GB activation budget determines OOM threshold. Dense at 8k tokens, B=4, 6 layers, 6 heads is ~38 GB of attention scores — OOMs well before that. Empirical OOM point likely T=4k–8k for dense, T=256k–1M for Clew.

### 8.2 Time budget

- Phase 1 dense training, 5000 steps, T=256: ~30 min on M3
- Phase 4 Clew training, same: ~60–90 min (Python dict overhead)
- Phase 5 memory scan: ~1h (lots of OOM-and-retry)
- Phase 6 NIAH: ~30 min
- **Total compute: ~4h.** Wall-clock with debugging: ~3 days.

### 8.3 Different machines

- **M1/M2 MacBook 8GB+:** identical, slightly slower
- **M3/M4 Pro/Max with 16GB+:** dense survives to ~16k
- **Linux + CUDA GPU:** swap `mps` for `cuda`; everything else same
- **Windows + CPU:** works but 10–20× slower

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Clew doesn't converge (LSH too noisy at small scale) | Medium | High | Validate LSH module on isolated synthetic problem first |
| Hash drift makes training unstable | Medium | High | Rebuild index every forward pass in v0 |
| MPS bugs in PyTorch | Medium | Medium | CPU fallback flag; verify on smaller config |
| Memory scan unconvincing (dense doesn't OOM as expected) | Low | Medium | Push to T=1M; even then asymptotic gap is the story |
| NIAH accuracy low (model too small to learn retrieval at all) | Medium | High | Stronger inductive bias via auxiliary loss; or 20M params |
| Implementation takes 3× longer than estimated | High | Low | Phase-gated; partial results still useful |
| Results inconclusive | Medium | Medium | Honest "what worked / what didn't" writeup is still valuable |
| Someone else publishes first | Low | Low | Good. Trifecta needs more attempts, not fewer. |

---

## 10. Roadmap Beyond v0

If v0 results are positive:

**v0.5 (1–2 weeks)**
- HNSW proximity graph
- Cross-layer index sharing
- 50M params
- BPE tokenization
- PG19 or code dataset

**v1 (1–2 months)**
- **Port ASC to Go** as sidecar service. Graph-heavy + pointer-heavy → Go's strength. Python keeps model + training; Go owns the index. Communication via shared memory or gRPC.
- Custom Metal kernel for candidate-attention step
- Train 100M+ model
- arxiv submission
- Submit to public long-context leaderboards (see §14)

**v2 (open-ended)**
- Custom CUDA kernels (FlashAriadne)
- Distributed training
- Frontier-scale: 1B+ params, 10M+ context
- Real codebase QA benchmark
- Open weights

---

## 11. Glossary

- **Attention** — operation that lets each word "look at" every other word for context
- **Quadratic** — cost ∝ N². The thing we're escaping.
- **Sublinear / subquadratic** — cost grows slower than N. Logarithmic is excellent.
- **Context window** — how many tokens a model can read at once
- **KV cache** — stored "memory" of past tokens during generation
- **LSH (Locality-Sensitive Hashing)** — hash function where similar inputs go to the same bucket
- **HNSW** — Hierarchical Navigable Small World; standard data structure in vector databases (Pinecone, Weaviate)
- **Dense attention** — standard transformer attention; quadratic
- **Sparse attention** — only computes a subset of attention scores
- **Needle in a haystack (NIAH)** — benchmark where a fact is hidden in a long document
- **MPS** — Metal Performance Shaders, Apple's GPU compute framework; PyTorch uses this on M-series
- **STE (Straight-Through Estimator)** — backprop trick for non-differentiable ops like sign() or argmax
- **uv** — Rust-based Python package manager from Astral, replaces pip + virtualenv

---

## 12. Open Questions for the Implementer

These are genuinely open in v0 — the answer comes from running the experiments:

1. Right `n_bits` for LSH? Probably 32–64 for Shakespeare scale; sweep to confirm.
2. How aggressive should the auxiliary router loss be? Coefficient sweep needed.
3. Hash shared across heads or per-head? Per-head more expressive, shared cheaper.
4. Does Shakespeare-scale need hierarchy (HNSW) or do flat buckets suffice? Likely flat is enough at T ≤ 64k.
5. How does Clew compare to a Reformer-style baseline?
6. Does the model "use" the long context, or just not crash? NIAH is the discriminator.

---

## 13. Reference Materials

For the implementer:

- **nanoGPT** (Karpathy): https://github.com/karpathy/nanoGPT
- **Karpathy's "Let's build GPT" video** — the cultural reference for code style
- **uv docs**: https://docs.astral.sh/uv/
- **Reformer paper** (Kitaev et al., 2020, arXiv:2001.04451) — closest prior work on LSH attention
- **Mamba paper** (Gu & Dao, 2023, arXiv:2312.00752)
- **NSA paper** (Yuan et al., 2025, arXiv:2502.11089)
- **MoBA** (Lu et al., 2025, arXiv:2502.13189)
- **DeepSeek V3.2 paper** (arXiv:2512.02556)
- **HNSW paper** (Malkov & Yashunin, 2018, arXiv:1603.09320)
- **Modern Hopfield Networks** (Ramsauer et al., 2020, arXiv:2008.02217)
- **The ARIADNE paper draft** (this conversation)
- **tinyshakespeare**: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

---

## 14. Public Eval Platforms & Benchmarks

This is the section that lets us compare Clew against the rest of the world. We won't submit a 10M-param Shakespeare model to any public leaderboard — it would lose badly. But we **can implement the same eval *protocols*** locally and compare numbers. Public benchmarks give us: (a) the methodology to copy, (b) reference numbers from frontier models for context, (c) a clear path to v1 submission once we have a real-scale model.

### 14.1 Long-context benchmarks (most relevant)

| Benchmark | What it tests | Where | Use for Clew |
|---|---|---|---|
| **RULER** (NVIDIA) | 13 synthetic tasks: NIAH variants, multi-key, multi-value, multi-query, variable tracking, common/frequent words, QA. Up to 128K. | github.com/NVIDIA/RULER | **The gold standard.** Implement a "RULER-lite" subset (4 tasks) at our scale. |
| **LongBench v2** (THUDM) | 503 multi-task examples, 8k–2M words, EN+ZH | github.com/THUDM/LongBench | Skip for v0 (too hard for tiny model); revisit at v1. |
| **InfiniteBench** (OpenBMB) | >100k tokens; math, code, dialogue | github.com/OpenBMB/InfiniteBench | v1 target. |
| **NIAH (original)** | Greg Kamradt's "Pressure Testing GPT-4-128K" | needleinahaystack.dev | The simplest, most viral. **Implement first.** |
| **MRCR** | Multi-Round Coreference Resolution; tests retrieval across long dialogs. SubQ cites this. | HF datasets | Stretch v0; v1 priority. |
| **BABILong** | Long-context bAbI; reasoning over up to 10M tokens | github.com/booydar/babilong | Cool target, may be too hard for 10M params. |
| **HELMET** (Princeton) | Holistic Evaluation of Long-context Models | github.com/princeton-nlp/HELMET | v1. |
| **L-Eval** | 18 long-doc tasks | github.com/OpenLMLab/LEval | v1. |
| **LooGLE** | Long Generic Eval | github.com/bigai-nlco/LooGLE | v1. |

### 14.2 General eval frameworks

- **lm-evaluation-harness** (EleutherAI, github.com/EleutherAI/lm-evaluation-harness) — de facto standard; supports many benchmarks. Wrap our model in their interface for v0.5+.
- **OpenCompass** (github.com/open-compass/opencompass) — broader, Chinese-led, comprehensive.
- **HuggingFace `evaluate`** library — programmatic access to standard metrics.

### 14.3 Public leaderboards

- **HuggingFace Open LLM Leaderboard** (huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — general; our model is too small to make sense here.
- **HuggingFace Long-Context Leaderboard** — specifically what we'd want to submit to at v1.
- **LiveBench** (livebench.ai) — fresh test sets to avoid contamination; good for "is this real?" sanity check.
- **Aider Polyglot** — code editing benchmark; relevant for v1 codebase-QA story.
- **Vellum's long-context comparisons** (vellum.ai/llm-leaderboard) — public side-by-sides.

### 14.4 Strategy for v0

1. **Implement NIAH locally** as primary metric (Phase 6). Score is comparable in spirit to RULER's "single needle" task.
2. **Implement a "RULER-lite" subset** in `eval/ruler_lite.py`: single-needle NIAH, multi-key NIAH, variable tracking, common-word extraction. Four tasks instead of thirteen. This gets us a credible eval protocol the research community recognizes.
3. **Compare to published numbers** in the README. Not "we beat Claude on RULER" — instead, "here's how a 10M-param Shakespeare model does on the same protocol; here are reference numbers from frontier models for context."
4. **Save lm-evaluation-harness integration for v0.5** when we have BPE and a real-scale model.

### 14.5 Strategy for v1

Once we have a 100M+ param model on a real dataset:

1. Submit to RULER (full 13 tasks)
2. Submit to LongBench v2
3. Submit to HELMET
4. Open a HuggingFace model card with eval results
5. Compare against DSA, NSA, MoBA, Mamba, Jamba on the same benchmarks
6. arxiv preprint with full evals

### 14.6 What gets reported

Even at v0 scale, we report:

- NIAH accuracy at 1k / 4k / 16k / 64k
- RULER-lite scores at the same lengths
- Memory and throughput at each length
- Recall@k vs dense top-k ground truth (this is unique — most benchmarks only test downstream task success, not the underlying retrieval mechanism)
- Honest comparison with published numbers from real models on the same protocols

---

## 15. HTML Landing Page Spec (`index.html`)

A single-file static landing page lives at the repo root and ships at v0. Hostable on Cloudflare Pages with zero build step.

### 15.1 Purpose

- Public face of the project. Anyone landing on `clew.dev` (or the GitHub Pages URL) gets the thesis, the results, and how to run it in under a minute.
- Doubles as a research artifact you can share on Twitter or in a paper footnote.
- Makes the project feel real to investors / non-technical readers without requiring them to read the README.

### 15.2 Audience and content priorities

| Reader | What they need |
|---|---|
| Someone who saw a tweet | The trifecta thesis, one chart that makes the point, link to repo |
| Researcher | Architecture diagram, comparison table, results, citations |
| Coding agent / engineer | "How to run" with `uv` commands, repo link |
| CEO / investor | Hero, thesis, results, status |

### 15.3 Sections (in order)

1. **Hero**
   - The word "Clew" in oversized display type
   - Tagline: *"A thread through the labyrinth. Subquadratic attention with exact retrieval, on a laptop."*
   - Tiny SVG: a labyrinth or thread spiral. Decorative, not loud.
   - Status badge: `v0.1 — May 2026` or `experimental`.

2. **The Thesis** (the trifecta)
   - Three constraints, written editorially. Numbered C1, C2, C3.
   - Brief explanation of why each matters.

3. **The Architecture Comparison Table**
   - Rows: Dense, Mamba/SSM, Sliding-window, NSA/MoBA/DSA, Reformer, **Clew**.
   - Columns: Subquadratic, Content-aware, Exact retrieval.
   - Checkmarks and crosses. Clew is the only row with three checks.
   - This is the single most-shared image. Make it good.

4. **How It Works** (the architecture)
   - Three stages: Hash → Lookup → Attend.
   - Inline SVG diagram or styled boxes.
   - One-paragraph plain-English explanation.

5. **Results**
   - Three charts side by side: loss curves, memory scan, NIAH accuracy.
   - Placeholder images at v0 launch; real PNGs (generated by the eval scripts) replace them post-experiment.
   - Two text blocks: sample generation from `clew-dense` and `clew-aria`, side by side, ~200 chars each.

6. **How to Run**
   - Code block with `uv` setup commands.
   - Copy button on the code block.
   - Link to repo.

7. **The Paper**
   - Link to the ARIADNE paper draft.
   - Citation block (BibTeX).

8. **Related Work**
   - Compact list with one-line summary each: nanoGPT, Reformer, Mamba, NSA, MoBA, DSA, SubQ, HNSW, Modern Hopfield Networks. Each links to arxiv.

9. **Footer**
   - Project author, license, contact.
   - Acknowledgement: "Built on a MacBook. Inspired by Karpathy's nanoGPT and the Subquadratic Inc. SubQ launch (May 2026)."

### 15.4 Aesthetic direction

**Editorial-academic with mythological undertones.** Should feel like a serious research artifact someone could print and tape to their lab door. Not flashy. Not generic.

- **Typography**
  - Display: **Fraunces** (variable serif, characterful) or **EB Garamond** (classic, mythological feel)
  - Body: **Newsreader** or **Source Serif Pro** (refined readable serif)
  - Mono: **JetBrains Mono** or **iA Writer Mono** (for code blocks)
  - Avoid: Inter, Roboto, Arial, Space Grotesk
- **Color palette**
  - Background: `#faf8f3` (warm cream — paper-like)
  - Text: `#1a1a2e` (deep ink, slight blue tint, not pure black)
  - Accent: `#c44536` (terracotta — Greek pottery red)
  - Muted: `#6b6b7d` (gray-purple for secondary text)
  - Line: `#d4cdb8` (subtle aged paper line)
- **Layout**
  - Single column, ~720px max width.
  - Generous line height (1.7+).
  - Asymmetric where it serves: pull-quotes, drop caps on section openers.
  - Margins like a printed book.
- **Visual details**
  - Drop cap on hero paragraph.
  - Hairline thread/line dividers between sections (could literally be a thread SVG motif).
  - Architecture diagram and comparison table are the two "set pieces" — invest design effort here.
- **Motion**
  - One subtle on-load animation — a thread drawing itself across the hero, or section reveals on scroll.
  - No bouncy effects, no fading icons, nothing that announces "I'm a website."
- **Restraint**
  - No purple gradients.
  - No cards with shadows.
  - No glassmorphism.
  - No icon clutter.

### 15.5 Technical constraints

- **Single file**: `index.html` with inline `<style>` and minimal inline `<script>`.
- **No build step**: hostable from `file://` or any static host.
- **Vanilla**: no React, no Vue, no Tailwind, no frameworks. Pure HTML+CSS+vanilla JS if needed.
- **Fonts via Google Fonts** with `<link rel="preconnect">` and `font-display: swap`.
- **Charts**: SVG inline (drawn by the eval scripts) or PNG generated by matplotlib. No client-side charting library.
- **Mobile-readable**: usable on phone. Single column already handles this.
- **Size budget**: < 50KB HTML+CSS, fonts loaded async.
- **Accessibility**: real semantic HTML, alt text on images, color contrast > 4.5:1.
- **License**: include MIT or Apache 2.0 in footer.

### 15.6 When to build

After Phase 7 (polish). The page can ship with placeholder charts at v0 launch, then the real charts get committed once the eval scripts produce them. The text content (thesis, architecture, comparison table, how-to-run) can be written before the experiments are done — those are stable.

### 15.7 Hosting

- **Cloudflare Pages** (recommended): connect to GitHub, auto-deploy on push, custom domain support, free.
- Alternative: GitHub Pages from the repo root. Slower DNS, fewer features.

---

## 16. CEO / Investor Cheatsheet

**What it is:** a small, public, runnable validation of the architecture that SubQ might have built privately.

**Why it matters:** if the trifecta (subquadratic + content-aware + exact-recall) really works at small scale, it gives us credibility, a research artifact, and a foundation for serious follow-on work. If it fails at small scale, we've added a real data point to a debate currently dominated by closed-source claims.

**What we'll see:**
- A 10-million-parameter model that writes Shakespeare as well as the standard implementation, but trains on **40× longer sequences without crashing**
- A "find this hidden fact in 64,000 words" test where our model succeeds and the standard one OOMs
- A clean public landing page (clew.dev) with the thesis, architecture, and results
- Total cost: ~$0 (laptop), ~$0 (Shakespeare is free), ~3–5 days of engineering

**What we won't have:**
- A frontier-scale model (that's a v1 conversation)
- A direct disproof or replication of SubQ (they're closed, we can't)
- A polished product

**Decision asked:** approve ~3–5 days of engineering time to produce v0. Single deliverable: a public repo + 1-page writeup of results + landing page.

**Risks:** experiment might be inconclusive. Even then the codebase, the framing, and the eval methodology are useful artifacts.

---

*Document version: v0.2 — May 2026. Codename: Clew. Architecture: ARIADNE. To be revised as implementation surfaces real constraints.*
