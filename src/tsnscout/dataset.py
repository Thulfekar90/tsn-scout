"""Labelled data: (instance, routing) pairs with the exact solver's verdict.

Each instance is labelled under several routings.  Seeing the same network succeed with one
routing and fail with another is what teaches the model to rank routings, which is the job
it has during search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import numpy as np

from .baselines import balanced_routing
from .generator import CONFIGS, generate, sample_routing
from .problem import Instance
from .solver import FEASIBLE, solve

ROUTING_KINDS = ("shortest", "balanced", "random", "perturbed")


@dataclass
class Sample:
    """One labelled routing of one instance."""

    inst: Instance
    routing: List[int]
    status: str
    seconds: float
    kind: str = ""

    @property
    def label(self) -> int:
        """1 if the solver produced a schedule within its time limit, else 0.

        Timeouts count as 0: for a designer, a routing the solver cannot schedule in the
        time available is not a usable routing.  The README reports how many there are.
        """
        return int(self.status == FEASIBLE)


def propose_routings(inst: Instance, rng: np.random.Generator) -> List[tuple]:
    """Four routings per instance, from naive to near-good, so labels are informative."""
    out = [("shortest", inst.shortest_routing()), ("balanced", balanced_routing(inst, rng))]
    out.append(("random", sample_routing(inst, rng, "partial" if rng.random() < 0.5 else "random")))
    start = sample_routing(inst, rng, "random")
    near = balanced_routing(inst, rng, start=start)
    for f in rng.choice(inst.n_flows, size=int(rng.integers(0, 6)), replace=False):
        near[f] = int(rng.integers(inst.n_candidates(f)))
    out.append(("perturbed", near))
    return out


def label_instance(args) -> dict:
    """Worker: generate instance ``seed`` and solve it under each proposed routing."""
    seed, cfg_name, time_limit = args
    inst = generate(seed, CONFIGS[cfg_name])
    rng = np.random.default_rng(seed + 7_000_000)
    rows = []
    for kind, routing in propose_routings(inst, rng):
        res = solve(inst, routing, time_limit=time_limit)
        rows.append({"routing": routing, "status": res.status, "seconds": round(res.seconds, 4), "kind": kind})
    return {"instance": inst.to_dict(), "rows": rows}


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def read_samples(paths: Sequence[str]) -> List[Sample]:
    return list(iter_samples(paths))


def iter_samples(paths: Sequence[str]) -> Iterator[Sample]:
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                inst = Instance.from_dict(rec["instance"])
                for row in rec["rows"]:
                    yield Sample(inst, row["routing"], row["status"], row["seconds"], row.get("kind", ""))
