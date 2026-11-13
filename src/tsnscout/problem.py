"""Problem definition: a switched Ethernet network carrying periodic time-triggered flows.

Time is measured in integer *ticks* (1 tick = 10 microseconds).  The model follows the
offset-based formulation of IEEE 802.1Qbv schedule synthesis (Craciunas et al., RTNS 2016),
simplified as documented in the README:

* every directed link is a resource that can transmit one frame at a time;
* a flow sends one transmission window of ``length`` ticks every ``period`` ticks;
* the same window is repeated on every link of the flow's route, in order;
* the whole journey must finish within ``deadline`` ticks of the first transmission.

A *routing* picks one candidate route per flow.  Whether a routing admits a valid schedule
is the expensive question the exact solver answers and the surrogate learns to predict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

TICK_MICROSECONDS = 10
#: Periods in ticks: 1, 2, 5, 10 and 20 ms.  The mix of 2 ms and 5 ms periods matters: their
#: greatest common divisor is 1 ms, so such flows compete for the same 1 ms residue window.
PERIODS: Tuple[int, ...] = (100, 200, 500, 1000, 2000)


@dataclass(frozen=True)
class Flow:
    """One periodic time-triggered stream."""

    src: int
    dst: int
    period: int
    length: int
    deadline: int


@dataclass
class Instance:
    """A network, its flows and the candidate routes of every flow.

    ``links[i]`` is the directed link ``(u, v)`` with link id ``i``.
    ``routes[f][p]`` is the list of link ids of candidate route ``p`` of flow ``f``.
    """

    n_nodes: int
    is_switch: List[bool]
    links: List[Tuple[int, int]]
    flows: List[Flow]
    routes: List[List[List[int]]]
    hop_delay: int = 1
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def n_flows(self) -> int:
        return len(self.flows)

    @property
    def n_links(self) -> int:
        return len(self.links)

    def n_candidates(self, f: int) -> int:
        return len(self.routes[f])

    def route(self, f: int, routing: Sequence[int]) -> List[int]:
        return self.routes[f][routing[f]]

    def shortest_routing(self) -> List[int]:
        """Candidate 0 of every flow (candidates are sorted by hop count)."""
        return [0] * self.n_flows

    def min_latency(self, f: int, p: int) -> int:
        """Lower bound on the end-to-end latency of flow ``f`` on candidate ``p``."""
        hops = len(self.routes[f][p])
        return hops * self.flows[f].length + (hops - 1) * self.hop_delay

    def link_users(self, routing: Sequence[int]) -> List[List[Tuple[int, int]]]:
        """For every link, the ``(flow, hop index)`` pairs that traverse it."""
        users: List[List[Tuple[int, int]]] = [[] for _ in self.links]
        for f in range(self.n_flows):
            for hop, link in enumerate(self.route(f, routing)):
                users[link].append((f, hop))
        return users

    def link_utilization(self, routing: Sequence[int]) -> List[float]:
        util = [0.0] * self.n_links
        for f, flow in enumerate(self.flows):
            for link in self.route(f, routing):
                util[link] += flow.length / flow.period
        return util

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "is_switch": [bool(b) for b in self.is_switch],
            "links": [list(l) for l in self.links],
            "flows": [asdict(f) for f in self.flows],
            "routes": self.routes,
            "hop_delay": self.hop_delay,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        return cls(
            n_nodes=d["n_nodes"],
            is_switch=list(d["is_switch"]),
            links=[tuple(l) for l in d["links"]],
            flows=[Flow(**f) for f in d["flows"]],
            routes=[[list(r) for r in cands] for cands in d["routes"]],
            hop_delay=d.get("hop_delay", 1),
            meta=d.get("meta", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Instance":
        return cls.from_dict(json.loads(s))


Schedule = Dict[Tuple[int, int], int]
"""Maps ``(flow, hop index)`` to the transmission start offset within the flow's period."""


def quick_infeasible(inst: Instance, routing: Sequence[int]) -> Optional[str]:
    """Cheap *sound* infeasibility tests.  Returns a reason, or ``None`` if none applies.

    These are necessary conditions only: passing them does not make a routing schedulable.
    Every method in the benchmark gets them for free, so the surrogate is judged on the
    cases they cannot decide.
    """
    for f, flow in enumerate(inst.flows):
        if inst.min_latency(f, routing[f]) > flow.deadline:
            return "deadline"
    for users in inst.link_users(routing):
        if sum(inst.flows[f].length / inst.flows[f].period for f, _ in users) > 1.0 + 1e-12:
            return "utilization"
        for a in range(len(users)):
            fa = inst.flows[users[a][0]]
            for b in range(a + 1, len(users)):
                fb = inst.flows[users[b][0]]
                if fa.length + fb.length > gcd(fa.period, fb.period):
                    return "gcd"
    return None
