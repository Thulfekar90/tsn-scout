"""Hand-written alternatives to the learned surrogate.

They matter as much as the model: a surrogate is only worth its training cost if it beats
what an engineer would try first.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .problem import Instance


def balance_objective(util: np.ndarray, power: float = 6.0) -> float:
    """Smooth stand-in for the maximum link utilisation (a p-norm)."""
    return float(np.sum(util**power))


def balanced_routing(
    inst: Instance,
    rng: Optional[np.random.Generator] = None,
    start: Optional[Sequence[int]] = None,
    power: float = 6.0,
    max_sweeps: int = 30,
) -> List[int]:
    """Load balancing by local search: move one flow at a time to the candidate route that
    lowers a p-norm of the link utilisations, until no single move helps."""
    rng = rng or np.random.default_rng(0)
    routing = list(start) if start is not None else inst.shortest_routing()
    util = np.asarray(inst.link_utilization(routing))
    for _ in range(max_sweeps):
        improved = False
        for f in rng.permutation(inst.n_flows):
            if inst.n_candidates(f) == 1:
                continue
            share = inst.flows[f].length / inst.flows[f].period
            util[inst.routes[f][routing[f]]] -= share
            best_p, best_val = routing[f], None
            for p, route in enumerate(inst.routes[f]):
                util[route] += share
                val = balance_objective(util, power)
                util[route] -= share
                if best_val is None or val < best_val - 1e-15:
                    best_p, best_val = p, val
            if best_p != routing[f]:
                improved = True
                routing[f] = best_p
            util[inst.routes[f][routing[f]]] += share
        if not improved:
            break
    return routing
