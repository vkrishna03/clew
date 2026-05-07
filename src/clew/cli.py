"""clew CLI: train / sample / eval."""
import argparse
import sys

from .config import Config


def cmd_train(args):
    from .train import train
    cfg = Config()
    if args.steps:
        cfg.max_steps = args.steps
    if args.block_size:
        cfg.block_size = args.block_size
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    train(cfg, args.model)


def cmd_sample(args):
    from .sample import sample
    text = sample(args.model, prompt=args.prompt, max_new_tokens=args.tokens)
    print(text)


def cmd_eval(args):
    if args.kind == "needle":
        from eval.needle_haystack import run as r
        r()
    elif args.kind == "memory":
        from eval.memory_scan import run as r
        r()
    elif args.kind == "recall":
        from eval.recall_at_k import run as r
        r()
    else:
        raise SystemExit(f"unknown eval: {args.kind}")


def main():
    p = argparse.ArgumentParser(prog="clew")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--model", choices=["dense", "aria"], required=True)
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--block-size", type=int, default=None)
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--device", default=None)
    pt.set_defaults(fn=cmd_train)

    ps = sub.add_parser("sample")
    ps.add_argument("--model", choices=["dense", "aria"], required=True)
    ps.add_argument("--prompt", default="\n")
    ps.add_argument("--tokens", type=int, default=300)
    ps.set_defaults(fn=cmd_sample)

    pe = sub.add_parser("eval")
    pe.add_argument("kind", choices=["needle", "memory", "recall"])
    pe.set_defaults(fn=cmd_eval)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
