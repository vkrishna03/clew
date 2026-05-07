"""Associative Skip-Cascade (v0): flat LSH bucket index with multi-probe."""
from collections import defaultdict
from typing import List
import torch

from .lsh import LearnedLSH


class ASC:
    """Per-(layer,head) bucket index over keys at positions 0..t-1."""

    def __init__(self, lsh: LearnedLSH, k_max: int = 64, n_probes: int = 4):
        self.lsh = lsh
        self.k_max = k_max
        self.n_probes = n_probes
        self.buckets: dict = defaultdict(list)

    def reset(self):
        self.buckets = defaultdict(list)

    @torch.no_grad()
    def insert_batch(self, ids: torch.Tensor, positions: torch.Tensor):
        """Insert keys with their integer bucket ids at given positions."""
        ids_list = ids.tolist()
        pos_list = positions.tolist()
        for bid, p in zip(ids_list, pos_list):
            self.buckets[bid].append(p)

    @torch.no_grad()
    def lookup(self, q_id: int, t_query: int) -> List[int]:
        """Return up to k_max candidate positions strictly less than t_query.

        Multi-probe: flip the lowest n_probes-1 bits.
        """
        candidates = set()
        candidates.update(self.buckets.get(q_id, ()))
        for b in range(self.n_probes - 1):
            probed = q_id ^ (1 << b)
            candidates.update(self.buckets.get(probed, ()))
        # causal filter
        out = [p for p in candidates if p < t_query]
        if len(out) > self.k_max:
            # Recent-first: prefer closer positions
            out.sort(reverse=True)
            out = out[: self.k_max]
        return out
