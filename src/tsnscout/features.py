"""Turn (instance, routing) pairs into graphs the model can read.

The graph is bipartite: one node per flow, one node per directed link, and an edge wherever
a flow's chosen route crosses a link.  That is the structure the scheduling constraints
live on: exclusivity couples the flows that meet on a link, and order and deadline couple
the links that one flow visits.

``InstanceEncoder`` precomputes everything that does not depend on the routing, then
encodes thousands of routings of the same instance in one vectorised call.  The search
loop depends on that speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .problem import PERIODS, Instance

FLOW_DIM = len(PERIODS) + 7
LINK_DIM = 3
EDGE_DIM = 4
_PERIOD_INDEX = {p: i for i, p in enumerate(PERIODS)}
_WINDOW = float(PERIODS[0])  # 1 ms: the residue window most flows compete for


@dataclass
class Batch:
    """A batch of bipartite graphs stored as one big disconnected graph."""

    x_flow: torch.Tensor  # (n_flow_nodes, FLOW_DIM)
    x_link: torch.Tensor  # (n_link_nodes, LINK_DIM)
    e_flow: torch.Tensor  # (n_edges,) flow node index
    e_link: torch.Tensor  # (n_edges,) link node index
    e_attr: torch.Tensor  # (n_edges, EDGE_DIM)
    flow_graph: torch.Tensor  # (n_flow_nodes,) graph index
    link_graph: torch.Tensor  # (n_link_nodes,) graph index
    n_graphs: int

    def to(self, device) -> "Batch":
        return Batch(
            self.x_flow.to(device), self.x_link.to(device), self.e_flow.to(device), self.e_link.to(device),
            self.e_attr.to(device), self.flow_graph.to(device), self.link_graph.to(device), self.n_graphs,
        )


class InstanceEncoder:
    """Encodes many routings of one instance at once."""

    def __init__(self, inst: Instance):
        self.inst = inst
        F, E = inst.n_flows, inst.n_links
        K = max(len(c) for c in inst.routes)
        H = max(len(r) for c in inst.routes for r in c)
        self.F, self.E = F, E
        self.route_links = np.full((F, K, H), -1, dtype=np.int64)
        self.route_len = np.zeros((F, K), dtype=np.int64)
        for f, cands in enumerate(inst.routes):
            for p, route in enumerate(cands):
                self.route_links[f, p, : len(route)] = route
                self.route_len[f, p] = len(route)
        self.n_cand = np.array([len(c) for c in inst.routes], dtype=np.int64)
        self.T = np.array([fl.period for fl in inst.flows], dtype=np.float32)
        self.L = np.array([fl.length for fl in inst.flows], dtype=np.float32)
        self.D = np.array([fl.deadline for fl in inst.flows], dtype=np.float32)
        self.share = self.L / self.T
        static = np.zeros((F, len(PERIODS) + 4), dtype=np.float32)
        for f, fl in enumerate(inst.flows):
            static[f, _PERIOD_INDEX[fl.period]] = 1.0
        static[:, -4] = self.L / _WINDOW
        static[:, -3] = self.share
        static[:, -2] = self.D / self.T
        static[:, -1] = self.n_cand / 4.0
        self.flow_static = static
        self.is_access = np.array(
            [0.0 if (inst.is_switch[u] and inst.is_switch[v]) else 1.0 for u, v in inst.links], dtype=np.float32
        )

    def encode_numpy(self, routings: np.ndarray) -> dict:
        routings = np.atleast_2d(np.asarray(routings, dtype=np.int64))
        B, F, E = routings.shape[0], self.F, self.E
        fidx = np.arange(F)[None, :]
        links = self.route_links[fidx, routings]  # (B, F, H)
        hops = self.route_len[fidx, routings].astype(np.float32)  # (B, F)
        min_lat = hops * self.L + (hops - 1) * self.inst.hop_delay
        slack = self.D - min_lat

        x_flow = np.empty((B, F, FLOW_DIM), dtype=np.float32)
        x_flow[:, :, : self.flow_static.shape[1]] = self.flow_static
        x_flow[:, :, -3] = hops / 8.0
        x_flow[:, :, -2] = slack / self.D  # share of the deadline left for waiting
        x_flow[:, :, -1] = np.minimum(slack / _WINDOW, 2.0)

        b, f, h = np.nonzero(links >= 0)
        link = links[b, f, h]
        e_flow = b * F + f
        e_link = b * E + link
        hop_count = hops[b, f]
        e_attr = np.empty((len(b), EDGE_DIM), dtype=np.float32)
        e_attr[:, 0] = h / np.maximum(hop_count - 1, 1)
        e_attr[:, 1] = h == 0
        e_attr[:, 2] = h == hop_count - 1
        e_attr[:, 3] = h * (self.L[f] + self.inst.hop_delay) / self.T[f]  # earliest arrival, in periods

        util = np.bincount(e_link, weights=self.share[f], minlength=B * E).astype(np.float32)
        count = np.bincount(e_link, minlength=B * E).astype(np.float32)
        x_link = np.stack([np.tile(self.is_access, B), util, count / 10.0], axis=1)
        return {
            "x_flow": x_flow.reshape(B * F, FLOW_DIM), "x_link": x_link, "e_flow": e_flow, "e_link": e_link,
            "e_attr": e_attr, "n_flow": np.full(B, F), "n_link": np.full(B, E),
        }

    def encode(self, routings: np.ndarray) -> Batch:
        return collate([self.encode_numpy(routings)])


def encode_sample(inst: Instance, routing: Sequence[int]) -> dict:
    return InstanceEncoder(inst).encode_numpy(np.asarray(routing)[None, :])


def collate(parts: List[dict]) -> Batch:
    """Concatenate encoded graphs, shifting node indices so edges stay within their graph."""
    flow_off = np.cumsum([0] + [int(p["n_flow"].sum()) for p in parts])
    link_off = np.cumsum([0] + [int(p["n_link"].sum()) for p in parts])
    n_flow = np.concatenate([p["n_flow"] for p in parts])
    n_link = np.concatenate([p["n_link"] for p in parts])
    graphs = np.arange(len(n_flow))
    return Batch(
        x_flow=torch.from_numpy(np.concatenate([p["x_flow"] for p in parts])),
        x_link=torch.from_numpy(np.concatenate([p["x_link"] for p in parts])),
        e_flow=torch.from_numpy(np.concatenate([p["e_flow"] + o for p, o in zip(parts, flow_off)])),
        e_link=torch.from_numpy(np.concatenate([p["e_link"] + o for p, o in zip(parts, link_off)])),
        e_attr=torch.from_numpy(np.concatenate([p["e_attr"] for p in parts])),
        flow_graph=torch.from_numpy(np.repeat(graphs, n_flow)),
        link_graph=torch.from_numpy(np.repeat(graphs, n_link)),
        n_graphs=len(n_flow),
    )


# --------------------------------------------------------------------------- hand-made features
HANDCRAFTED_NAMES = (
    "max_util", "top3_util", "mean_util", "max_flows_on_link", "max_residue_load", "top3_residue_load",
    "min_slack_frac", "mean_slack_frac", "tight_flows", "max_tight_on_link", "mean_hops", "n_flows", "n_links",
)


def handcrafted(inst: Instance, routing: Sequence[int]) -> np.ndarray:
    """Summary statistics an engineer would compute by hand: the feature set of the
    gradient-boosting baseline.  ``max_residue_load`` is a period-aware load: a clique of
    flows that must all be disjoint modulo 1 ms cannot exceed the window."""
    users = inst.link_users(routing)
    util = np.sort(np.asarray(inst.link_utilization(routing)))[::-1]
    slack = np.array(
        [(fl.deadline - inst.min_latency(f, routing[f])) / fl.deadline for f, fl in enumerate(inst.flows)]
    )
    tight = slack < 0.05
    residue, tight_on_link = [], []
    for us in users:
        by_period = {}
        for f, _ in us:
            by_period.setdefault(inst.flows[f].period, []).append(inst.flows[f].length)
        load = sum(by_period.get(100, []))
        load += max(by_period.get(200, [0])) + max(by_period.get(500, [0]))
        residue.append(load / _WINDOW)
        tight_on_link.append(sum(bool(tight[f]) for f, _ in us))
    residue = np.sort(np.asarray(residue))[::-1]
    hops = [len(inst.route(f, routing)) for f in range(inst.n_flows)]
    return np.array([
        util[0], util[:3].mean(), util.mean(), max(len(us) for us in users), residue[0], residue[:3].mean(),
        slack.min(), slack.mean(), tight.sum(), max(tight_on_link), float(np.mean(hops)), inst.n_flows, inst.n_links,
    ], dtype=np.float32)
