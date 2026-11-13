"""Predict-then-verify routing search.

Every method here has the same shape: *propose* an ordered list of routings, then hand them
to the exact solver one by one until one is proved schedulable.  Methods differ only in how
they order the list.  The solver is the judge in every case, so no method can return a
routing that does not work; a bad proposer just wastes solver calls.

* ``shortest``  one proposal: shortest paths.
* ``random``    uniformly random routings.
* ``balanced``  load balancing from several starting points.
* ``scout``     simulated annealing driven by the surrogate ensemble.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .baselines import balanced_routing
from .checker import is_valid
from .features import InstanceEncoder
from .gnn import Ensemble
from .generator import sample_routing
from .problem import Instance, Schedule, quick_infeasible
from .solver import FEASIBLE, solve


@dataclass
class ScoutConfig:
    chains: int = 40
    steps: int = 80
    rescore: int = 120  # best states of the walk that the full ensemble re-ranks
    beta: float = 1.0  # pessimism: score = mean logit - beta * ensemble std
    temp_start: float = 1.5  # annealing temperature, in logit units
    temp_end: float = 0.1
    min_distance: float = 0.04  # proposals differ in at least this share of the flows


def _diverse_top(states: np.ndarray, scores: np.ndarray, k: int, min_dist: int) -> List[int]:
    """Greedy pick of the k best states that are at least ``min_dist`` reroutes apart.
    Near-duplicates fail for the same reason, so verifying them wastes solver calls."""
    order = np.argsort(-scores)
    chosen: List[int] = []
    for i in order:
        if all((states[i] != states[j]).sum() >= min_dist for j in chosen):
            chosen.append(int(i))
            if len(chosen) == k:
                break
    return chosen


def scout_proposals(
    inst: Instance,
    ensemble: Ensemble,
    k: int,
    rng: np.random.Generator,
    cfg: ScoutConfig = ScoutConfig(),
) -> Tuple[List[List[int]], dict]:
    """Anneal ``cfg.chains`` routings in parallel under the surrogate's score, then let the
    whole ensemble re-rank the best routings seen anywhere along the way and return ``k``."""
    enc = InstanceEncoder(inst)
    F = inst.n_flows
    movable = np.flatnonzero(enc.n_cand > 1)

    starts = [inst.shortest_routing(), balanced_routing(inst, rng)]
    while len(starts) < cfg.chains:
        start = sample_routing(inst, rng, "partial" if len(starts) % 2 else "random")
        starts.append(balanced_routing(inst, rng, start=start) if len(starts) % 4 < 2 else start)
    state = np.asarray(starts, dtype=np.int64)

    # Each chain is steered by one ensemble member, so a step costs one forward pass per
    # routing rather than one per member.  The whole ensemble votes at the end.
    members = ensemble.models
    owner = np.arange(cfg.chains) % len(members)

    def walk_score(routings: np.ndarray) -> np.ndarray:
        out = np.empty(len(routings), dtype=np.float32)
        with torch.no_grad():
            for m, model in enumerate(members):
                mask = owner == m
                if mask.any():
                    out[mask] = (model(enc.encode(routings[mask]))[0] / ensemble.temperature).numpy()
        return out

    score = walk_score(state)
    seen_states, seen_scores = [state.copy()], [score.copy()]
    n_evals = len(state)
    if len(movable):
        rows = np.arange(cfg.chains)
        for temp in np.geomspace(cfg.temp_start, cfg.temp_end, cfg.steps):
            flows = rng.choice(movable, size=cfg.chains)
            shift = 1 + (rng.random(cfg.chains) * (enc.n_cand[flows] - 1)).astype(np.int64)
            cand = state.copy()
            cand[rows, flows] = (state[rows, flows] + shift) % enc.n_cand[flows]
            cand_score = walk_score(cand)
            n_evals += cfg.chains
            accept = np.log(rng.random(cfg.chains)) < (cand_score - score) / temp
            state[accept], score[accept] = cand[accept], cand_score[accept]
            seen_states.append(cand)
            seen_scores.append(cand_score)

    all_states = np.concatenate(seen_states)
    all_scores = np.concatenate(seen_scores)
    _, first = np.unique(all_states, axis=0, return_index=True)
    all_states, all_scores = all_states[first], all_scores[first]
    short = np.argsort(-all_scores)[: cfg.rescore]
    all_states = all_states[short]
    final = ensemble.predict(enc.encode(all_states), beta=cfg.beta)[2].numpy()
    n_evals += len(all_states) * len(members)
    picks = _diverse_top(all_states, final, k, max(1, int(round(cfg.min_distance * F))))
    info = {"evaluations": n_evals, "scores": [float(final[i]) for i in picks]}
    return [all_states[i].tolist() for i in picks], info


def balanced_proposals(inst: Instance, k: int, rng: np.random.Generator) -> List[List[int]]:
    out, seen = [], set()
    tries = 0
    while len(out) < k and tries < 6 * k:
        start = None if tries == 0 else sample_routing(inst, rng, "random")
        r = balanced_routing(inst, rng, start=start)
        tries += 1
        if tuple(r) not in seen:
            seen.add(tuple(r))
            out.append(r)
    return out


def random_proposals(inst: Instance, k: int, rng: np.random.Generator) -> List[List[int]]:
    return [sample_routing(inst, rng, "random") for _ in range(k)]


@dataclass
class Outcome:
    """What happened when a proposal list was verified."""

    solved: bool
    routing: Optional[List[int]] = None
    schedule: Optional[Schedule] = None
    solver_calls: int = 0
    propose_seconds: float = 0.0
    #: (wall-clock seconds since the method started, solver status) after each solver call
    trace: List[Tuple[float, str]] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.trace[-1][0] if self.trace else self.propose_seconds


def verify(
    inst: Instance,
    proposals: Sequence[Sequence[int]],
    time_limit: float = 5.0,
    propose_seconds: float = 0.0,
) -> Outcome:
    """Send proposals to the exact solver in order; stop at the first schedulable one.

    Routings that fail a sound one-line check are skipped without a solver call.  A schedule
    is only reported after the independent checker has confirmed it.
    """
    out = Outcome(False, propose_seconds=propose_seconds)
    clock = propose_seconds
    for routing in proposals:
        if quick_infeasible(inst, routing) is not None:
            continue
        res = solve(inst, routing, time_limit=time_limit)
        clock += res.seconds
        out.solver_calls += 1
        out.trace.append((clock, res.status))
        if res.status == FEASIBLE:
            if not is_valid(inst, routing, res.schedule):
                raise AssertionError("solver returned a schedule the independent checker rejects")
            out.solved, out.routing, out.schedule = True, list(routing), res.schedule
            break
    return out


def run_method(
    name: str,
    inst: Instance,
    k: int,
    rng: np.random.Generator,
    ensemble: Optional[Ensemble] = None,
    time_limit: float = 5.0,
    scout_cfg: ScoutConfig = ScoutConfig(),
) -> Outcome:
    """Propose with method ``name`` and verify up to ``k`` routings."""
    t0 = time.perf_counter()
    if name == "shortest":
        proposals = [inst.shortest_routing()]
    elif name == "random":
        proposals = random_proposals(inst, k, rng)
    elif name == "balanced":
        proposals = balanced_proposals(inst, k, rng)
    elif name == "scout":
        if ensemble is None:
            raise ValueError("the scout method needs a trained ensemble")
        proposals, _ = scout_proposals(inst, ensemble, k, rng, scout_cfg)
    else:
        raise ValueError(f"unknown method {name!r}")
    return verify(inst, proposals, time_limit, propose_seconds=time.perf_counter() - t0)
