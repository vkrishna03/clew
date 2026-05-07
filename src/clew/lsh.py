"""Learned LSH with straight-through binarization."""
import torch
import torch.nn as nn


class LearnedLSH(nn.Module):
    """Maps d-dim vectors to n_bits-bit signed binary codes via learned projection.

    forward returns:
      hard codes in {-1, +1} (with STE backward through tanh surrogate) when training,
      hard codes in {0, 1} ints when eval (used for bucket-id hashing).
    """

    def __init__(self, d_model: int, n_bits: int = 32, tau: float = 0.5):
        super().__init__()
        self.W = nn.Linear(d_model, n_bits, bias=False)
        nn.init.normal_(self.W.weight, std=1.0 / d_model**0.5)
        self.tau = tau
        self.n_bits = n_bits

    def forward(self, x):
        logits = self.W(x) / self.tau
        if self.training:
            soft = torch.tanh(logits)
            hard = (logits > 0).float() * 2 - 1
            return hard + (soft - soft.detach())  # STE
        else:
            return (logits > 0).float() * 2 - 1

    @torch.no_grad()
    def codes_int(self, x) -> torch.Tensor:
        """Return integer bucket ids of shape (..., ) packed from sign bits."""
        bits = (self.W(x) > 0).long()  # (..., n_bits)
        # pack bits into a single int64 (n_bits up to 63)
        weights = (1 << torch.arange(self.n_bits, device=x.device, dtype=torch.long))
        return (bits * weights).sum(dim=-1)
