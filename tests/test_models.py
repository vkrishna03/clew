import torch
from clew.config import Config
from clew.train import make_model


def test_dense_forward_backward():
    cfg = Config(block_size=32, batch_size=2, n_layer=2, n_head=2, d_model=32)
    cfg.vocab_size = 65
    m = make_model(cfg, "dense")
    x = torch.randint(0, 65, (2, 32))
    logits, loss, aux = m(x, x)
    assert logits.shape == (2, 32, 65)
    loss.backward()


def test_aria_forward_backward():
    cfg = Config(block_size=32, batch_size=2, n_layer=2, n_head=2, d_model=32, lsh_bits=8)
    cfg.vocab_size = 65
    m = make_model(cfg, "aria")
    m.train()
    x = torch.randint(0, 65, (2, 32))
    logits, loss, aux = m(x, x)
    assert logits.shape == (2, 32, 65)
    (loss + 0.1 * aux).backward()
    # all params should have grad
    for n, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {n}"
