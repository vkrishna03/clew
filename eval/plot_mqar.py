"""Plot Zoology MQAR comparison: nanoGPT vs clew-dense vs clew-aria."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    fig, ax = plt.subplots(figsize=(8, 5))
    files = [
        ("nanogpt", "runs/zoology_mqar_nanogpt.json", "#1a1a2e", "-"),
        ("clew-dense", "runs/zoology_mqar_dense.json", "#6b6b7d", "-"),
        ("clew-aria", "runs/zoology_mqar_aria.json", "#c44536", "-"),
    ]
    for name, path, color, ls in files:
        p = Path(path)
        if not p.exists():
            print(f"missing {p}")
            continue
        log = json.loads(p.read_text())
        steps = [e["step"] for e in log]
        acc = [e["eval_acc"] for e in log]
        ax.plot(steps, acc, ls, color=color, label=f"{name} (final {acc[-1]:.0%})", linewidth=1.8)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="50%")
    ax.set_xlabel("step")
    ax.set_ylabel("eval accuracy on canonical Zoology MQAR")
    ax.set_title("Associative recall: dense models phase-transition, ARIA does not")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/mqar.png", dpi=120)
    print("saved plots/mqar.png")


if __name__ == "__main__":
    run()
