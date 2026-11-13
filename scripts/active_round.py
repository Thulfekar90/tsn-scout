"""One round of active learning: label what the surrogate itself proposes.

    python scripts/active_round.py --model models/scout_round0.pt --out data/active1.jsonl

A surrogate trained on hand-picked routings is then asked to rank routings chosen by a
search that maximises its own output.  Search finds the inputs where the model is most
optimistic, which are exactly where it is most likely wrong.  Solving those proposals
exactly and adding them to the training set closes that gap.  Instances are fresh (their
seeds are used nowhere else) and drawn from the small configuration only, so the larger
test networks stay unseen.
"""

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

SEED0 = 6_000_000
_STATE = {}


def _init(model_path):
    import torch

    from tsnscout.gnn import Ensemble

    torch.set_num_threads(1)
    _STATE["ens"] = Ensemble.load(model_path)


def _work(job):
    from tsnscout.generator import CONFIGS, generate
    from tsnscout.search import scout_proposals
    from tsnscout.solver import solve

    seed, k, time_limit = job
    inst = generate(seed, CONFIGS["small"])
    proposals, _ = scout_proposals(inst, _STATE["ens"], k, np.random.default_rng(seed + 1))
    rows = []
    for routing in proposals:
        res = solve(inst, routing, time_limit=time_limit)
        rows.append({"routing": routing, "status": res.status, "seconds": round(res.seconds, 4), "kind": "scout"})
    return {"instance": inst.to_dict(), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=SEED0)
    args = ap.parse_args()

    jobs = [(args.seed0 + i, args.k, args.time_limit) for i in range(args.n)]
    counts, t0 = Counter(), time.time()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool, open(args.out, "w") as fh:
        for i, rec in enumerate(pool.imap(_work, jobs), 1):
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            counts.update(row["status"] for row in rec["rows"])
            if i % 50 == 0 or i == args.n:
                print(f"[active] {i}/{args.n}  {dict(counts)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
