"""Exact schedule synthesis with CP-SAT.  This is the expensive oracle and the final judge.

Decision variables are the transmission offsets ``phi[f, hop]`` of every flow on every link
of its route.  Three families of constraints:

1. *Flow order.*  ``phi[f, h+1] >= phi[f, h] + length + hop_delay``.
2. *Deadline.*    ``phi[f, last] + length - phi[f, 0] <= deadline``.
3. *Link exclusivity.*  Two periodic flows ``i`` and ``j`` on the same link never overlap
   iff ``(phi_j - phi_i) mod g`` lies in ``[L_i, g - L_j]`` with ``g = gcd(T_i, T_j)``.
   The set of all start-time differences between instances of the two flows is exactly
   ``phi_j - phi_i + g*Z``, and no difference may fall in the open interval ``(-L_j, L_i)``.
   This needs one integer variable per pair instead of one interval per frame instance.

The same condition can be stated with scheduling propagators instead: every frame instance
within the link's hyperperiod becomes an interval and one NoOverlap covers the link.  The
two statements are logically equivalent but derived independently, so the test suite
checks that they never disagree.  In our measurements the interval form decides more
instances within a given time limit, so it is the default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import gcd, lcm
from typing import Dict, List, Optional, Sequence

from ortools.sat.python import cp_model

from .problem import Instance, Schedule

FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
UNKNOWN = "unknown"  # the time limit was reached without an answer


@dataclass
class SolveResult:
    status: str
    schedule: Optional[Schedule]
    seconds: float
    routing: Optional[List[int]] = None
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.status == FEASIBLE


def _run(model: cp_model.CpModel, time_limit: float, seed: int, workers: int):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    t0 = time.perf_counter()
    code = solver.Solve(model)
    seconds = time.perf_counter() - t0
    if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status = FEASIBLE
    elif code == cp_model.INFEASIBLE:
        status = INFEASIBLE
    else:
        status = UNKNOWN
    stats = {"conflicts": solver.NumConflicts(), "branches": solver.NumBranches()}
    return solver, status, seconds, stats


def _add_exclusion(model, phi_i, phi_j, fi, fj, tag, enforce=()):
    """Link exclusivity between two flows; returns False if it can never hold."""
    g = gcd(fi.period, fj.period)
    if fi.length + fj.length > g:
        if enforce:
            model.AddBoolOr([lit.Not() for lit in enforce])
            return True
        return False
    lo = -(fi.period - fi.length)  # bounds of phi_j - phi_i
    hi = fj.period - fj.length
    k = model.NewIntVar((fi.length - hi) // g - 1, (g - fj.length - lo) // g + 1, "k" + tag)
    diff = phi_j - phi_i + k * g
    c1 = model.Add(diff >= fi.length)
    c2 = model.Add(diff <= g - fj.length)
    if enforce:
        c1.OnlyEnforceIf(list(enforce))
        c2.OnlyEnforceIf(list(enforce))
    return True


MODULAR = "modular"
INTERVAL = "interval"
BOTH = "both"


def solve(
    inst: Instance,
    routing: Sequence[int],
    time_limit: float = 10.0,
    seed: int = 0,
    workers: int = 1,
    formulation: str = INTERVAL,
) -> SolveResult:
    """Find offsets for a fixed routing, or prove that none exist.

    ``formulation`` selects how link exclusivity is stated: ``"modular"`` (one integer per
    pair of flows), ``"interval"`` (one NoOverlap per link over the link's hyperperiod) or
    ``"both"``.  They are logically equivalent; the tests check that they agree.
    """
    if formulation not in (MODULAR, INTERVAL, BOTH):
        raise ValueError(f"unknown formulation {formulation!r}")
    model = cp_model.CpModel()
    phi: Dict = {}
    for f, flow in enumerate(inst.flows):
        hops = len(inst.route(f, routing))
        if inst.min_latency(f, routing[f]) > flow.deadline:
            return SolveResult(INFEASIBLE, None, 0.0, list(routing), {"presolved": 1})
        for h in range(hops):
            phi[f, h] = model.NewIntVar(0, flow.period - flow.length, f"phi_{f}_{h}")
        for h in range(hops - 1):
            model.Add(phi[f, h + 1] >= phi[f, h] + flow.length + inst.hop_delay)
        model.Add(phi[f, hops - 1] + flow.length - phi[f, 0] <= flow.deadline)

    for link, users in enumerate(inst.link_users(routing)):
        if len(users) < 2:
            continue
        for a in range(len(users)):
            fa, ha = users[a]
            for b in range(a + 1, len(users)):
                fb, hb = users[b]
                if inst.flows[fa].length + inst.flows[fb].length > gcd(inst.flows[fa].period, inst.flows[fb].period):
                    return SolveResult(INFEASIBLE, None, 0.0, list(routing), {"presolved": 1})
                if formulation in (MODULAR, BOTH):
                    _add_exclusion(
                        model, phi[fa, ha], phi[fb, hb], inst.flows[fa], inst.flows[fb], f"_{link}_{fa}_{fb}"
                    )
        if formulation in (INTERVAL, BOTH):
            hyper = 1
            for f, _ in users:
                hyper = lcm(hyper, inst.flows[f].period)
            intervals = []
            for f, h in users:
                flow = inst.flows[f]
                for r in range(hyper // flow.period):
                    intervals.append(
                        model.NewFixedSizeIntervalVar(
                            phi[f, h] + r * flow.period, flow.length, f"iv_{link}_{f}_{r}"
                        )
                    )
            model.AddNoOverlap(intervals)

    solver, status, seconds, stats = _run(model, time_limit, seed, workers)
    schedule = {key: int(solver.Value(var)) for key, var in phi.items()} if status == FEASIBLE else None
    return SolveResult(status, schedule, seconds, list(routing), stats)


def solve_joint(
    inst: Instance, time_limit: float = 60.0, seed: int = 0, workers: int = 1
) -> SolveResult:
    """Exact joint routing and scheduling: the solver also chooses each flow's route.

    This is the baseline that needs no learning at all.  It is complete, so given enough
    time it finds a routing whenever one exists; the question the experiments ask is how
    much time that takes compared with scouting routings first.
    """
    model = cp_model.CpModel()
    pick = {}
    phi = {}  # (flow, link) -> offset on that link if the chosen route uses it
    for f, flow in enumerate(inst.flows):
        lits = []
        for p, route in enumerate(inst.routes[f]):
            lit = model.NewBoolVar(f"r_{f}_{p}")
            if inst.min_latency(f, p) > flow.deadline:
                model.Add(lit == 0)
            pick[f, p] = lit
            lits.append(lit)
            for link in route:
                if (f, link) not in phi:
                    phi[f, link] = model.NewIntVar(0, flow.period - flow.length, f"phi_{f}_{link}")
            for a, b in zip(route, route[1:]):
                model.Add(phi[f, b] >= phi[f, a] + flow.length + inst.hop_delay).OnlyEnforceIf(lit)
            model.Add(phi[f, route[-1]] + flow.length - phi[f, route[0]] <= flow.deadline).OnlyEnforceIf(lit)
        model.AddExactlyOne(lits)

    uses = {}  # (flow, link) -> literal "flow traverses link"
    by_link: Dict[int, List[int]] = {}
    for f in range(inst.n_flows):
        per_link: Dict[int, List] = {}
        for p, route in enumerate(inst.routes[f]):
            for link in route:
                per_link.setdefault(link, []).append(pick[f, p])
        for link, lits in per_link.items():
            if len(lits) == len(inst.routes[f]):
                uses[f, link] = None  # on every candidate: always used
            else:
                u = model.NewBoolVar(f"u_{f}_{link}")
                model.AddMaxEquality(u, lits)
                uses[f, link] = u
            by_link.setdefault(link, []).append(f)

    for link, fs in by_link.items():
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                fa, fb = fs[a], fs[b]
                enforce = [u for u in (uses[fa, link], uses[fb, link]) if u is not None]
                ok = _add_exclusion(
                    model, phi[fa, link], phi[fb, link], inst.flows[fa], inst.flows[fb],
                    f"_{link}_{fa}_{fb}", enforce=enforce,
                )
                if not ok:
                    return SolveResult(INFEASIBLE, None, 0.0, None, {"presolved": 1})

    solver, status, seconds, stats = _run(model, time_limit, seed, workers)
    if status != FEASIBLE:
        return SolveResult(status, None, seconds, None, stats)
    routing = [
        next(p for p in range(inst.n_candidates(f)) if solver.Value(pick[f, p]))
        for f in range(inst.n_flows)
    ]
    schedule = {
        (f, h): int(solver.Value(phi[f, link]))
        for f in range(inst.n_flows)
        for h, link in enumerate(inst.route(f, routing))
    }
    return SolveResult(FEASIBLE, schedule, seconds, routing, stats)
