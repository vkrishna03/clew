"""Generate text from a checkpoint."""
import pickle
from pathlib import Path
import torch

from .config import Config
from .train import make_model


def sample(kind: str, prompt: str = "\n", max_new_tokens: int = 300, temperature: float = 0.8, top_k: int = 40):
    out_dir = Path("runs") / kind
    ck = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    with open(Path(cfg.data_dir) / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    device = torch.device(cfg.device if torch.backends.mps.is_available() else "cpu")
    model = make_model(cfg, kind).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = "".join(itos[int(t)] for t in out[0].tolist())
    return text
