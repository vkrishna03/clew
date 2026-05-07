"""Plot dense vs aria loss curves from saved logs."""
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, color in (("dense", "#1a1a2e"), ("aria", "#c44536")):
        p = Path(f"runs/{kind}/log.pkl")
        if not p.exists():
            print(f"missing {p}")
            continue
        log = pickle.load(open(p, "rb"))
        steps = [e["step"] for e in log]
        train = [e["train"] for e in log]
        val = [e["val"] for e in log]
        ax.plot(steps, train, "-", color=color, label=f"{kind} train")
        ax.plot(steps, val, "--", color=color, label=f"{kind} val", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("clew-dense vs clew-aria on tinyshakespeare")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/loss.png", dpi=120)
    print("saved plots/loss.png")


if __name__ == "__main__":
    run()
