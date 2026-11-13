"""Random problem instances shaped like zonal in-vehicle networks.

The backbone is a ring of switches with a few chords, so every pair of switches has at
least two disjoint paths; end systems hang off one switch each.  That gives routing real
freedom in the backbone while access links stay fixed, which is the situation a network
designer faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from math import ceil
from typing import List, Tuple

import networkx as nx
import numpy as np

from .problem import PERIODS, Flow, Instance


@dataclass(frozen=True)
class GenConfig:
    """Size and load ranges of the instance distribution."""

    n_switches: Tuple[int, int] = (4, 8)
    es_per_switch: Tuple[int, int] = (2, 4)
    n_flows: Tuple[int, int] = (20, 60)
    chord_prob: float = 0.4
    k_routes: int = 4
    max_detour: int = 3
    #: Dirichlet concentration of the per-switch traffic shares; small values make hot spots.
    traffic_skew: float = 1.0
    #: Sampling weights of PERIODS; short periods dominate the load.
    period_weights: Tuple[float, ...] = (0.22, 0.30, 0.28, 0.14, 0.06)
    #: A window may take at most this fraction of its period (keeps multi-hop routes possible).
    max_len_frac: float = 0.12
    max_len: int = 50
    #: Per-instance load scale, drawn uniformly; spreads instances across the feasibility transition.
    load_scale: Tuple[float, float] = (0.5, 1.4)
    #: Deadline = slack factor x minimum latency on the longest candidate route, capped at the period.
    deadline_slack: Tuple[float, float] = (1.0, 1.2)
    hop_delay: int = 1


SMALL = GenConfig()
CONFIGS = {}
#: Larger than anything seen in training; used only to test generalisation to new system sizes.
LARGE = GenConfig(n_switches=(10, 14), n_flows=(50, 110))
CONFIGS.update(small=SMALL, large=LARGE)


def _topology(rng: np.random.Generator, cfg: GenConfig) -> Tuple[nx.Graph, List[bool]]:
    n_sw = int(rng.integers(cfg.n_switches[0], cfg.n_switches[1] + 1))
    g = nx.cycle_graph(n_sw)
    for u in range(n_sw):
        for v in range(u + 2, n_sw):
            if (u, v) != (0, n_sw - 1) and rng.random() < cfg.chord_prob * 4 / n_sw:
                g.add_edge(u, v)
    is_switch = [True] * n_sw
    node = n_sw
    for sw in range(n_sw):
        for _ in range(int(rng.integers(cfg.es_per_switch[0], cfg.es_per_switch[1] + 1))):
            g.add_edge(sw, node)
            is_switch.append(False)
            node += 1
    return g, is_switch


def generate(seed: int, cfg: GenConfig = SMALL) -> Instance:
    """Generate one instance deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    g, is_switch = _topology(rng, cfg)
    n_nodes = g.number_of_nodes()

    links: List[Tuple[int, int]] = []
    for u, v in sorted(g.edges()):
        links.append((u, v))
        links.append((v, u))
    link_id = {l: i for i, l in enumerate(links)}

    end_systems = [n for n in range(n_nodes) if not is_switch[n]]
    attach = {es: next(iter(g.neighbors(es))) for es in end_systems}
    n_flows = int(rng.integers(cfg.n_flows[0], cfg.n_flows[1] + 1))
    load = float(rng.uniform(*cfg.load_scale))
    weights = np.asarray(cfg.period_weights) / sum(cfg.period_weights)

    flows: List[Flow] = []
    routes: List[List[List[int]]] = []
    path_cache = {}
    # Uneven traffic: some switches send or receive much more than others, as a zone
    # controller next to the cameras does.  Shortest-path routing then overloads the
    # backbone links between hot spots while other links idle.
    n_sw = sum(is_switch)
    es_of = {sw: [es for es in end_systems if attach[es] == sw] for sw in range(n_sw)}
    w_src = rng.dirichlet([cfg.traffic_skew] * n_sw)
    w_dst = rng.dirichlet([cfg.traffic_skew] * n_sw)
    while len(flows) < n_flows:
        sw_a, sw_b = int(rng.choice(n_sw, p=w_src)), int(rng.choice(n_sw, p=w_dst))
        if sw_a == sw_b:
            continue  # same switch: nothing to route
        src, dst = int(rng.choice(es_of[sw_a])), int(rng.choice(es_of[sw_b]))
        if (src, dst) not in path_cache:
            shortest = nx.shortest_path_length(g, src, dst)
            paths = []
            for path in islice(nx.shortest_simple_paths(g, src, dst), cfg.k_routes * 3):
                if len(path) - 1 > shortest + cfg.max_detour:
                    break
                paths.append([link_id[(a, b)] for a, b in zip(path, path[1:])])
                if len(paths) == cfg.k_routes:
                    break
            path_cache[(src, dst)] = paths
        cands = path_cache[(src, dst)]

        period = int(rng.choice(PERIODS, p=weights))
        hi = min(cfg.max_len, int(cfg.max_len_frac * period))
        lo = max(1, hi // 5)
        length = int(np.clip(round(rng.integers(lo, hi + 1) * load), 1, hi))

        def min_lat(route: List[int]) -> int:
            return len(route) * length + (len(route) - 1) * cfg.hop_delay

        # A candidate is kept only if the flow could use it when alone in the network, so no
        # routing is infeasible for a reason a one-line check would catch.  The deadline is
        # set against the longest kept candidate: detours are allowed but eat the slack.
        usable = [list(r) for r in cands if min_lat(r) <= period]
        if not usable:
            continue
        deadline = min(period, ceil(min_lat(usable[-1]) * rng.uniform(*cfg.deadline_slack)))
        flows.append(Flow(src, dst, period, length, int(deadline)))
        routes.append(usable)

    return Instance(
        n_nodes=n_nodes,
        is_switch=is_switch,
        links=links,
        flows=flows,
        routes=routes,
        hop_delay=cfg.hop_delay,
        meta={"seed": seed, "load": load},
    )


def sample_routing(inst: Instance, rng: np.random.Generator, mode: str = "random") -> List[int]:
    """Draw a routing: ``shortest``, uniformly ``random``, or ``partial`` (shortest for a
    random subset of flows, random for the rest)."""
    if mode == "shortest":
        return inst.shortest_routing()
    routing = [int(rng.integers(inst.n_candidates(f))) for f in range(inst.n_flows)]
    if mode == "partial":
        keep = rng.random(inst.n_flows) < rng.uniform(0.2, 0.8)
        routing = [0 if k else r for k, r in zip(keep, routing)]
    elif mode != "random":
        raise ValueError(f"unknown mode {mode!r}")
    return routing
