"""Generate instances and label them with the exact solver.

    python scripts/make_dataset.py --split train

Seeds are disjoint between splits, so no test instance is ever seen in training.
"""

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

from tsnscout.dataset import label_instance

SPLITS = {
    # name: (config, first seed, number of instances)
    "train": ("small", 0, 3500),
    "val": ("small", 1_000_000, 400),
    "test": ("small", 2_000_000, 500),
    "test_large": ("large", 3_000_000, 250),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--n", type=int, default=None, help="override the number of instances")
    ap.add_argument("--time-limit", type=float, default=5.0, help="solver time limit per routing, seconds")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, seed0, n = SPLITS[args.split]
    n = args.n or n
    out = args.out or f"data/{args.split}.jsonl"
    jobs = [(seed0 + i, cfg, args.time_limit) for i in range(n)]
    t0 = time.time()
    counts = Counter()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with Pool(args.workers) as pool, open(out, "w") as fh:
        for i, rec in enumerate(pool.imap(label_instance, jobs, chunksize=4), 1):
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            counts.update(row["status"] for row in rec["rows"])
            if i % 100 == 0 or i == n:
                print(f"[{args.split}] {i}/{n} instances  {dict(counts)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
