"""Command line entry point.

    tsnscout demo --config large --seed 5000057 --model models/scout.pt --plot example.png

generates one network, lets every method try to route and schedule it, and prints what
each one achieved.
"""

from __future__ import annotations

import argparse

import numpy as np


def demo(args) -> None:
    import torch

    from .baselines import balanced_routing
    from .features import InstanceEncoder
    from .generator import CONFIGS, generate
    from .gnn import Ensemble
    from .search import run_method

    torch.set_num_threads(1)
    inst = generate(args.seed, CONFIGS[args.config])
    ens = Ensemble.load(args.model)
    switches = sum(inst.is_switch)
    print(f"instance {args.seed} ({args.config}): {switches} switches, {inst.n_nodes - switches} end systems, "
          f"{inst.n_links} directed links, {inst.n_flows} flows")

    enc = InstanceEncoder(inst)
    named = {"shortest paths": inst.shortest_routing(), "load balancing": balanced_routing(inst)}
    prob, std, _ = ens.predict(enc.encode(np.array(list(named.values()))))
    print("\nsurrogate's opinion before any solving:")
    for name, p, s in zip(named, prob.tolist(), std.tolist()):
        print(f"  {name:15s} P(schedulable) = {p:5.1%}   ensemble disagreement = {s:.2f}")

    print(f"\n{'method':28s} {'verified schedule':18s} {'solver calls':>12s} {'seconds':>9s}")
    best = None
    for method, label in (("shortest", "shortest paths"), ("balanced", "load balancing"), ("scout", "surrogate-guided")):
        out = run_method(method, inst, args.k, np.random.default_rng(args.seed + 1), ens, time_limit=args.time_limit)
        print(f"{label:28s} {'yes' if out.solved else 'no':18s} {out.solver_calls:12d} {out.seconds:9.2f}")
        if out.solved and (best is None or method == "scout"):
            best = out
    if args.plot and best is not None:
        from .viz import plot_instance

        plot_instance(inst, best.routing, best.schedule, args.plot)
        print(f"\nschedule figure written to {args.plot}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tsnscout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="route and schedule one generated network with every method")
    d.add_argument("--config", default="large", choices=["small", "large"])
    d.add_argument("--seed", type=int, default=5_000_057)
    d.add_argument("--model", default="models/scout.pt")
    d.add_argument("--k", type=int, default=8, help="solver calls each method may spend")
    d.add_argument("--time-limit", type=float, default=5.0)
    d.add_argument("--plot", default=None, help="write the network and its schedule to this image file")
    d.set_defaults(func=demo)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
