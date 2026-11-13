"""Plot style and the instance/schedule figure."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle

from .problem import PERIODS, TICK_MICROSECONDS, Instance, Schedule

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
#: Fixed categorical order; a method keeps its colour in every figure.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
METHOD_COLOR = {"scout": BLUE, "balanced": ORANGE, "joint": AQUA, "shortest": YELLOW, "random": MAGENTA}
METHOD_LABEL = {
    "scout": "Surrogate-guided (ours)", "balanced": "Load balancing", "joint": "Exact joint solver",
    "shortest": "Shortest paths", "random": "Random routings",
}
PERIOD_COLOR = dict(zip(PERIODS, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)))


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def plot_instance(
    inst: Instance,
    routing: Sequence[int],
    schedule: Optional[Schedule],
    path: str,
    title: str = "",
    n_links: int = 6,
    window_ms: float = 2.0,
) -> None:
    """Left: the network, backbone links shaded by utilisation.  Right: the verified gate
    schedule of the busiest links over the first ``window_ms`` milliseconds."""
    apply_style()
    util = np.asarray(inst.link_utilization(routing))
    fig, (ax_net, ax_gantt) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.55]})

    g = nx.Graph()
    g.add_nodes_from(range(inst.n_nodes))
    edge_util = {}
    for i, (u, v) in enumerate(inst.links):
        key = (min(u, v), max(u, v))
        edge_util[key] = max(edge_util.get(key, 0.0), util[i])
    g.add_edges_from(edge_util)
    switches = [n for n in g if inst.is_switch[n]]
    shell = nx.shell_layout(g, [switches, [n for n in g if not inst.is_switch[n]]])
    pos = nx.spring_layout(g, pos=shell, fixed=switches, seed=1, iterations=60, k=0.25)
    cmap = plt.get_cmap("Blues")
    for (u, v), value in edge_util.items():
        ax_net.plot(*zip(pos[u], pos[v]), color=cmap(0.18 + 0.8 * min(value, 1.0)), lw=1.2 + 4.5 * min(value, 1.0),
                    solid_capstyle="round", zorder=1)
    ax_net.scatter(*zip(*[pos[n] for n in switches]), s=230, marker="s", color=INK, zorder=3)
    ends = [n for n in g if not inst.is_switch[n]]
    ax_net.scatter(*zip(*[pos[n] for n in ends]), s=46, color=SURFACE, edgecolor=INK_2, linewidth=1.2, zorder=3)
    for n in switches:
        ax_net.text(*pos[n], f"S{n}", color=SURFACE, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=4)
    ax_net.set_title(f"Network: {len(switches)} switches, {len(ends)} end systems, {inst.n_flows} flows")
    ax_net.text(0.0, -0.04, "Link width and shade = utilisation under the chosen routing",
                transform=ax_net.transAxes, color=INK_2, fontsize=8.5, va="top")
    ax_net.set_axis_off()

    busiest = [int(i) for i in np.argsort(-util)[:n_links]]
    users = inst.link_users(routing)
    window = int(window_ms * 1000 / TICK_MICROSECONDS)
    to_ms = TICK_MICROSECONDS / 1000.0
    for row, link in enumerate(busiest):
        ax_gantt.add_patch(Rectangle((0, row - 0.36), window * to_ms, 0.72, color=GRID, alpha=0.45, lw=0))
        if schedule is None:
            continue
        for f, hop in users[link]:
            flow = inst.flows[f]
            start = schedule[f, hop]
            while start < window:
                ax_gantt.add_patch(Rectangle(
                    (start * to_ms, row - 0.36), min(flow.length, window - start) * to_ms, 0.72,
                    facecolor=PERIOD_COLOR[flow.period], edgecolor=SURFACE, linewidth=0.9,
                ))
                start += flow.period
    ax_gantt.set_yticks(range(len(busiest)))
    ax_gantt.set_yticklabels([
        f"{'S' if inst.is_switch[u] else 'E'}{u}→{'S' if inst.is_switch[v] else 'E'}{v}   {util[l]:.0%}"
        for l in busiest for u, v in [inst.links[l]]
    ])
    ax_gantt.set_ylim(len(busiest) - 0.4, -0.6)
    ax_gantt.set_xlim(0, window * to_ms)
    ax_gantt.set_xlabel("time within the schedule cycle (ms)")
    ax_gantt.grid(axis="y", visible=False)
    ax_gantt.set_title(title or "Verified gate schedule of the busiest links")
    handles = [Rectangle((0, 0), 1, 1, color=PERIOD_COLOR[p]) for p in PERIODS]
    ax_gantt.legend(handles, [f"{p * TICK_MICROSECONDS / 1000:g} ms" for p in PERIODS], title="flow period",
                    ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), handlelength=1.2, columnspacing=1.4)
    fig.savefig(path)
    plt.close(fig)
