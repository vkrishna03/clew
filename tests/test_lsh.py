import torch
from clew.lsh import LearnedLSH


def test_lsh_shapes():
    lsh = LearnedLSH(d_model=16, n_bits=8)
    x = torch.randn(4, 16)
    codes = lsh(x)
    assert codes.shape == (4, 8)
    ids = lsh.codes_int(x)
    assert ids.shape == (4,)
    assert ids.dtype == torch.long


def test_lsh_grad_flows():
    lsh = LearnedLSH(d_model=16, n_bits=8)
    lsh.train()
    x = torch.randn(4, 16, requires_grad=True)
    codes = lsh(x)
    codes.sum().backward()
    assert x.grad is not None
    assert lsh.W.weight.grad is not None


def test_lsh_clusters_after_training():
    """Train LSH so paired (anchor, positive) get the same bucket more often than random."""
    torch.manual_seed(0)
    d, n_bits, n = 16, 8, 200
    lsh = LearnedLSH(d, n_bits)
    opt = torch.optim.Adam(lsh.parameters(), lr=1e-2)
    anchors = torch.randn(n, d)
    positives = anchors + 0.05 * torch.randn(n, d)
    for _ in range(300):
        ca = lsh(anchors)
        cp = lsh(positives)
        # encourage anchor & positive to share signs
        loss = -(ca * cp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    lsh.eval()
    ids_a = lsh.codes_int(anchors)
    ids_p = lsh.codes_int(positives)
    same = (ids_a == ids_p).float().mean().item()
    assert same > 0.5, f"recall@bucket too low: {same:.2%}"
