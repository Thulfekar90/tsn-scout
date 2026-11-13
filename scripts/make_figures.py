"""Draw every figure in the README from the JSON files in results/.

    python scripts/make_figures.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from tsnscout.viz import BLUE, GRID, INK, INK_2, METHOD_COLOR, METHOD_LABEL, ORANGE, SURFACE, apply_style

SPLIT_TITLE = {"small": "Training-size networks (4-8 switches)", "large": "Larger, unseen sizes (10-14 switches)"}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _base(method: str) -> str:
    return method.split(":")[0]


def _spread_labels(ys, gap):
    """Push label positions apart so direct labels do not overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def fig_search(results, path, methods_filter=None):
    apply_style()
    fig, axes = plt.subplots(1, len(results), figsize=(6.3 * len(results), 4.3), sharey=True, squeeze=False)
    for ax, (name, data) in zip(axes[0], results.items()):
        records, meta = data["records"], data["meta"]
        methods = [m for m in meta["methods"] if methods_filter is None or m in methods_filter]
        n = len(records)
        t_max = max(meta["joint_limit"], max(r["methods"][m]["trace"][-1][0] for r in records for m in methods if r["methods"][m]["trace"]))
        ends = []
        for m in methods:
            times = sorted(r["methods"][m]["trace"][-1][0] for r in records if r["methods"][m]["solved"])
            xs = [0.01] + [max(t, 0.01) for t in times] + [t_max]
            ys = [0] + list(np.arange(1, len(times) + 1) / n) + [len(times) / n]
            ax.step(xs, ys, where="post", color=METHOD_COLOR[_base(m)], lw=2.4 if _base(m) == "scout" else 1.8)
            ends.append((m, len(times) / n))
        label_y = _spread_labels([e[1] for e in ends], 0.045)
        for (m, share), y in zip(ends, label_y):
            ax.text(t_max * 1.12, y, f"{share:.0%}", color=INK, fontsize=9.5, va="center",
                    fontweight="bold" if _base(m) == "scout" else "normal")
        ax.set_xscale("log")
        ax.set_xlim(0.01, t_max * 1.05)
        ax.set_ylim(0, 1)
        ax.set_xlabel("wall-clock seconds per instance (log scale)")
        ax.set_title(f"{SPLIT_TITLE[name]}  •  {n} instances")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0][0].set_ylabel("instances with a verified schedule")
    handles = [plt.Line2D([], [], color=METHOD_COLOR[_base(m)], lw=2.2) for m in methods]
    fig.legend(handles, [METHOD_LABEL[_base(m)] for m in methods], loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Share of instances solved as time passes", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(path)
    plt.close(fig)


def fig_calls(results, path, methods_filter=None):
    apply_style()
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 3.9), sharey=True, squeeze=False)
    for ax, (name, data) in zip(axes[0], results.items()):
        records, meta = data["records"], data["meta"]
        methods = [m for m in meta["methods"] if _base(m) in ("scout", "balanced", "random")
                   and (methods_filter is None or m in methods_filter)]
        n, k = len(records), meta["k"]
        ends = []
        for m in methods:
            calls = [r["methods"][m]["calls"] for r in records if r["methods"][m]["solved"]]
            ys = [sum(c <= j for c in calls) / n for j in range(1, k + 1)]
            ax.plot(range(1, k + 1), ys, color=METHOD_COLOR[_base(m)], marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ends.append((m, ys[-1]))
        for (m, y0), y in zip(ends, _spread_labels([e[1] for e in ends], 0.05)):
            ax.text(k + 0.25, y, METHOD_LABEL[_base(m)], color=INK, fontsize=9, va="center")
        ax.set_xlim(0.7, k + 3.4)
        ax.set_xticks(range(1, k + 1))
        ax.set_ylim(0, 1)
        ax.set_xlabel("exact-solver calls spent")
        ax.set_title(SPLIT_TITLE[name])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0][0].set_ylabel("instances solved")
    fig.suptitle("How many expensive solver calls each proposer needs", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def fig_prediction(pred, path):
    apply_style()
    splits = [("test", "Training-size networks", BLUE), ("test_large", "Larger, unseen sizes", ORANGE)]
    panels = [("ranking", "Ranking routings of the same network\n(share of good/bad pairs ordered correctly)"),
              ("auc_nontrivial", "Telling schedulable from unschedulable\n(AUC, cases a one-line check cannot decide)")]
    names = list(pred["splits"]["test"]["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        for row, name in enumerate(names):
            vals = [pred["splits"][s]["methods"][name][key] for s, _, _ in splits if s in pred["splits"]]
            ax.plot([min(vals), max(vals)], [row, row], color=GRID, lw=2.5, zorder=1)
            for (s, _, color), v in zip(splits, vals):
                ax.scatter(v, row, s=85, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(max(vals) + 0.012, row, " / ".join(f"{v:.3f}" for v in vals), va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlim(0.5, 1.09)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.axvline(0.5, color=INK_2, lw=1, ls=(0, (2, 3)))
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=10)
    axes[0].text(0.505, len(names) - 0.58, "chance", color=INK_2, fontsize=8, va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8) for _, _, c in splits]
    fig.legend(handles, [t for _, t, _ in splits], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Prediction quality of the surrogate and of two simpler predictors", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.08)
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(pred, path):
    apply_style()
    splits = [("test", "Training-size networks", BLUE), ("test_large", "Larger, unseen sizes", ORANGE)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (2, 3)))
    ax.text(0.63, 0.57, "perfect calibration", color=INK_2, fontsize=8, rotation=38)
    for split, label, color in splits:
        if split not in pred["splits"]:
            continue
        rel = [b for b in pred["splits"][split]["reliability"] if b["n"] >= 10]
        ax.plot([b["predicted"] for b in rel], [b["observed"] for b in rel], color=color, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        sel = pred["splits"][split]["selective"]
        ax2.plot([s["coverage"] for s in sel], [s["accuracy"] for s in sel], color=color, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax2.text(1.012, sel[-1]["accuracy"], f"{sel[-1]['accuracy']:.1%}", color=INK, fontsize=9, va="center")
    ax.set_xlabel("predicted probability of being schedulable")
    ax.set_ylabel("observed share schedulable")
    ax.set_title("When it says 80%, is it right 80% of the time?", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax2.set_xlabel("share of routings the surrogate answers itself\n(the least confident rest go to the solver)")
    ax2.set_ylabel("accuracy on the answered share")
    ax2.set_title("Deferring uncertain cases to the solver", fontsize=10)
    ax2.set_xlim(0.48, 1.07)
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("Does the surrogate know when it does not know?", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    results = {}
    for name in ("small", "large"):
        data = _load(f"results/search_{name}.json")
        if data:
            results[name] = data
    if results:
        main_methods = {"shortest", "random", "balanced", "scout:scout", "scout", "joint"}
        fig_search(results, "figures/search_time.png", main_methods)
        fig_calls(results, "figures/search_calls.png", main_methods)
        print("figures/search_time.png  figures/search_calls.png")
    pred = _load("results/prediction.json")
    if pred:
        fig_prediction(pred, "figures/prediction.png")
        fig_calibration(pred, "figures/calibration.png")
        print("figures/prediction.png  figures/calibration.png")


if __name__ == "__main__":
    main()
