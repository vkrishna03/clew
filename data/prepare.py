"""Download tinyshakespeare, char-tokenize, save train/val splits."""
from pathlib import Path
import pickle
import numpy as np
import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent


def main():
    raw = DATA_DIR / "input.txt"
    if not raw.exists():
        print(f"downloading {URL}")
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        raw.write_text(r.text)
    text = raw.read_text()
    print(f"length: {len(text):,} chars")

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    print(f"vocab_size: {vocab_size}")

    n = len(text)
    train_text = text[: int(0.9 * n)]
    val_text = text[int(0.9 * n) :]

    def encode(s):
        return np.array([stoi[c] for c in s], dtype=np.uint16)

    train_ids = encode(train_text)
    val_ids = encode(val_text)
    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")

    with open(DATA_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)
    print(f"train: {len(train_ids):,} tokens, val: {len(val_ids):,} tokens")
    print("done.")


if __name__ == "__main__":
    main()
