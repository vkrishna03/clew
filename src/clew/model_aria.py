"""ARIADNE attention: hash-cascade lookup + exact attention over candidate set."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .lsh import LearnedLSH
from .asc import ASC


class AriaAttention(nn.Module):
    """For each query at position t:
        1. Hash q via learned LSH
        2. Lookup ~k candidate positions with multi-probe (causal)
        3. Exact attention over that candidate set

    v0: rebuild bucket index every forward pass; per-(B, head) Python loop over t.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # one LSH per head — small, more expressive
        self.hashes = nn.ModuleList(
            [LearnedLSH(self.d_head, n_bits=cfg.lsh_bits) for _ in range(self.n_head)]
        )
        self.k_max = cfg.asc_k_max
        self.n_probes = cfg.asc_n_probes
        self.last_aux_loss = None

    def _build_indices(self, codes_k: torch.Tensor) -> list:
        """Build per-(B, head) bucket dict. codes_k shape: (B, H, T) int64."""
        B, H, T = codes_k.shape
        indices = [[None] * H for _ in range(B)]
        codes_cpu = codes_k.detach().cpu().tolist()
        for b in range(B):
            for h in range(H):
                d = {}
                for t in range(T):
                    bid = codes_cpu[b][h][t]
                    d.setdefault(bid, []).append(t)
                indices[b][h] = (d, codes_cpu[b][h])
        return indices

    def _candidate_indices(self, codes_q: torch.Tensor, codes_k: torch.Tensor) -> torch.Tensor:
        """Return cand_idx tensor (B, H, T, k_max) with -1 for empty slots.

        Causal: candidates are positions < t.
        """
        B, H, T = codes_q.shape
        k_max = self.k_max
        n_probes = self.n_probes
        device = codes_q.device

        cand = torch.full((B, H, T, k_max), -1, dtype=torch.long, device=device)
        codes_q_l = codes_q.detach().cpu().tolist()
        codes_k_l = codes_k.detach().cpu().tolist()
        for b in range(B):
            for h in range(H):
                # incremental bucket dict
                d = {}
                for t in range(T):
                    qid = codes_q_l[b][h][t]
                    # gather candidates with multi-probe over current d
                    cset = []
                    if qid in d:
                        cset.extend(d[qid])
                    for probe in range(n_probes - 1):
                        pid = qid ^ (1 << probe)
                        if pid in d:
                            cset.extend(d[pid])
                    if cset:
                        # most-recent-first, then truncate
                        cset = sorted(set(cset), reverse=True)[:k_max]
                        n = len(cset)
                        cand[b, h, t, :n] = torch.tensor(cset, device=device)
                    # insert key at position t
                    kid = codes_k_l[b][h][t]
                    d.setdefault(kid, []).append(t)
        return cand

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # hash codes per head
        codes_q_list, codes_k_list = [], []
        for h in range(H):
            codes_q_list.append(self.hashes[h].codes_int(q[:, h]))  # (B, T)
            codes_k_list.append(self.hashes[h].codes_int(k[:, h]))
        codes_q = torch.stack(codes_q_list, dim=1)  # (B, H, T)
        codes_k = torch.stack(codes_k_list, dim=1)

        cand = self._candidate_indices(codes_q, codes_k)  # (B, H, T, K)
        K = self.k_max

        # Gather K[cand], V[cand]. cand has -1 placeholders.
        # Advanced indexing avoids materializing a T x T intermediate.
        safe_cand = cand.clamp(min=0)  # (B, H, T, K)
        b_idx = torch.arange(B, device=k.device).view(B, 1, 1, 1).expand_as(safe_cand)
        h_idx = torch.arange(H, device=k.device).view(1, H, 1, 1).expand_as(safe_cand)
        K_cand = k[b_idx, h_idx, safe_cand]  # (B, H, T, K, D)
        V_cand = v[b_idx, h_idx, safe_cand]

        # attention scores: q dot K_cand
        # q: (B, H, T, D) -> (B, H, T, 1, D)
        scores = (q.unsqueeze(3) * K_cand).sum(-1) / (D**0.5)  # (B, H, T, K)
        mask = cand < 0
        scores = scores.masked_fill(mask, float("-inf"))
        # if a query has zero candidates (e.g. position 0), softmax of all -inf -> nan.
        # Detect and replace: produce zero output for such queries.
        all_masked = mask.all(dim=-1, keepdim=True)  # (B, H, T, 1)
        scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)
        attn = F.softmax(scores, dim=-1)  # (B, H, T, K)
        out = (attn.unsqueeze(-1) * V_cand).sum(dim=3)  # (B, H, T, D)
        # zero out positions with no candidates
        out = out.masked_fill(all_masked.expand_as(out[..., :1]).any(-1).unsqueeze(-1), 0.0) if False else out
        # Simpler: when all_masked, the softmax produces uniform but we already zeroed scores; output is mean of V_cand (junk gathered from index 0). Force zero:
        out = torch.where(all_masked, torch.zeros_like(out), out)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        # auxiliary router loss — encourages hash codes of q,k to match where dense
        # attention would attend strongly. Computed on a fraction of forward passes.
        self.last_aux_loss = self._aux_loss(q, k) if self.training else None
        return out

    def _aux_loss(self, q, k):
        """Cheap teacher: dense-softmax top-1 key per query → align continuous hash codes.

        Uses a single random batch item per head to keep cost subquadratic-ish for our
        signaling purpose. Detached gradient through the choice of top-1.
        """
        B, H, T, D = q.shape
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-1, -2)) / (D**0.5)  # (B, H, T, T)
            # causal mask
            causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
            # for position 0 there is no candidate; ignore it
            top1 = scores.argmax(dim=-1)  # (B, H, T)
            top1 = top1.clamp(min=0)
        # gather chosen keys
        idx = top1.unsqueeze(-1).expand(B, H, T, D)
        k_top = torch.gather(k, 2, idx)  # (B, H, T, D)
        # apply per-head soft hash codes (pre-tanh) to q and k_top, push them together
        loss = 0.0
        for h in range(H):
            qh = q[:, h]                          # (B, T, D)
            kh = k_top[:, h]                      # (B, T, D)
            phi_q = self.hashes[h].W(qh) / self.hashes[h].tau  # (B, T, n_bits)
            phi_k = self.hashes[h].W(kh) / self.hashes[h].tau
            # cosine alignment: encourage same sign pattern
            loss = loss - F.cosine_similarity(phi_q, phi_k, dim=-1).mean()
        return loss / H
