"""Train the surrogate ensemble.

    python scripts/train.py --train data/train.jsonl --val data/val.jsonl --out models/scout.pt
"""

import argparse
import json
import os

from tsnscout.dataset import read_samples
from tsnscout.train import train_ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="models/scout.pt")
    ap.add_argument("--report", default=None)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    train, val = read_samples(args.train), read_samples(args.val)
    print(f"train {len(train)} samples ({sum(s.label for s in train)} schedulable), val {len(val)}")
    ens, report = train_ensemble(
        train, val, n_models=args.members, epochs=args.epochs, hidden=args.hidden, rounds=args.rounds
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ens.save(args.out)
    report["train_files"] = args.train
    report_path = args.report or os.path.splitext(args.out)[0] + ".report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh)
    print(f"saved {args.out} and {report_path}")


if __name__ == "__main__":
    main()
