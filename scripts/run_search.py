"""The headline experiment: how quickly does each method reach a verified schedule?

    python scripts/run_search.py --config large --n 100 --model scout=models/scout.pt

Every method proposes up to ``--k`` routings; the exact solver checks them in order.  The
exact joint solver, which needs no proposals, runs as the reference.  Instances use seeds
that appear in no training, validation or test file.
"""

import argparse
import json
import time
from multiprocessing import get_context

import numpy as np

SEED0 = {"small": 4_000_000, "large": 5_000_000}
_STATE = {}


def _init(models):
    import torch

    from tsnscout.gnn import Ensemble

    torch.set_num_threads(1)
    _STATE["models"] = {name: Ensemble.load(path) for name, path in models.items()}


def _work(job):
    from tsnscout.generator import CONFIGS, generate
    from tsnscout.search import run_method
    from tsnscout.solver import solve_joint

    seed, cfg, methods, k, time_limit, joint_limit = job
    inst = generate(seed, CONFIGS[cfg])
    rec = {"seed": seed, "n_flows": inst.n_flows, "n_links": inst.n_links, "methods": {}}
    for method in methods:
        rng = np.random.default_rng(seed + 1)
        if method == "joint":
            res = solve_joint(inst, time_limit=joint_limit)
            rec["methods"][method] = {
                "solved": res.feasible, "calls": 1, "propose": 0.0, "trace": [[res.seconds, res.status]],
            }
            continue
        name, _, model = method.partition(":")
        out = run_method(name, inst, k, rng, ensemble=_STATE["models"].get(model or "scout"), time_limit=time_limit)
        rec["methods"][method] = {
            "solved": out.solved, "calls": out.solver_calls, "propose": out.propose_seconds,
            "trace": [[t, s] for t, s in out.trace],
        }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=sorted(SEED0))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=8, help="solver calls each proposing method may spend")
    ap.add_argument("--time-limit", type=float, default=5.0, help="solver limit per proposal, seconds")
    ap.add_argument("--joint-limit", type=float, default=60.0, help="limit of the exact joint solver, seconds")
    ap.add_argument("--model", action="append", default=[], help="name=path; used by method scout:name")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.model)
    methods = args.methods or ["shortest", "random", "balanced"] + [f"scout:{m}" for m in models] + ["joint"]
    jobs = [(SEED0[args.config] + i, args.config, methods, args.k, args.time_limit, args.joint_limit) for i in range(args.n)]
    out = args.out or f"results/search_{args.config}.json"
    t0 = time.time()
    records = []
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(models,)) as pool:
        for i, rec in enumerate(pool.imap(_work, jobs), 1):
            records.append(rec)
            if i % 10 == 0 or i == args.n:
                solved = {m: sum(r["methods"][m]["solved"] for r in records) for m in methods}
                print(f"[{args.config}] {i}/{args.n}  solved {solved}  {time.time() - t0:.0f}s", flush=True)
    meta = {"config": args.config, "k": args.k, "time_limit": args.time_limit, "joint_limit": args.joint_limit,
            "methods": methods, "models": models, "workers": args.workers}
    with open(out, "w") as fh:
        json.dump({"meta": meta, "records": records}, fh)
    print("saved", out)


if __name__ == "__main__":
    main()
