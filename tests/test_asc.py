import torch
from clew.lsh import LearnedLSH
from clew.asc import ASC


def test_asc_insert_lookup_causal():
    lsh = LearnedLSH(d_model=8, n_bits=6)
    asc = ASC(lsh, k_max=8, n_probes=2)
    keys = torch.randn(20, 8)
    ids = lsh.codes_int(keys)
    asc.insert_batch(ids, torch.arange(20))
    # query at t=10 must only see positions < 10
    cands = asc.lookup(int(ids[10].item()), t_query=10)
    assert all(c < 10 for c in cands)


def test_asc_finds_inserted():
    lsh = LearnedLSH(d_model=8, n_bits=6)
    asc = ASC(lsh, k_max=16, n_probes=1)
    keys = torch.randn(5, 8)
    ids = lsh.codes_int(keys)
    asc.insert_batch(ids, torch.arange(5))
    # exact same key -> same bucket -> retrieved
    q_id = int(ids[2].item())
    cands = asc.lookup(q_id, t_query=5)
    assert 2 in cands
