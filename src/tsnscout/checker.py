"""Independent schedule checker.

It shares no reasoning with the solver: every frame instance in the hyperperiod is painted
onto a timeline and overlaps are counted.  A schedule the solver calls valid must pass
here, which guards against a mistake in the modular arithmetic of the solver model.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .problem import Instance, Schedule


def violations(inst: Instance, routing: Sequence[int], schedule: Schedule) -> List[str]:
    """Return a list of human-readable constraint violations (empty means valid)."""
    out: List[str] = []
    for f, flow in enumerate(inst.flows):
        hops = len(inst.route(f, routing))
        for h in range(hops):
            start = schedule.get((f, h))
            if start is None:
                out.append(f"flow {f} hop {h}: no offset")
                return out
            if start < 0 or start + flow.length > flow.period:
                out.append(f"flow {f} hop {h}: window leaves its period")
        for h in range(hops - 1):
            if schedule[f, h + 1] < schedule[f, h] + flow.length + inst.hop_delay:
                out.append(f"flow {f}: hop {h + 1} starts before hop {h} has been forwarded")
        latency = schedule[f, hops - 1] + flow.length - schedule[f, 0]
        if latency > flow.deadline:
            out.append(f"flow {f}: latency {latency} exceeds deadline {flow.deadline}")

    for link, users in enumerate(inst.link_users(routing)):
        if len(users) < 2:
            continue
        hyper = int(np.lcm.reduce([inst.flows[f].period for f, _ in users]))
        busy = np.zeros(hyper, dtype=np.int32)
        for f, h in users:
            flow = inst.flows[f]
            for r in range(hyper // flow.period):
                start = schedule[f, h] + r * flow.period
                busy[start : start + flow.length] += 1
        if busy.max() > 1:
            out.append(f"link {link} {inst.links[link]}: {int((busy > 1).sum())} ticks carry two frames")
    return out


def is_valid(inst: Instance, routing: Sequence[int], schedule: Schedule) -> bool:
    return not violations(inst, routing, schedule)
