"""How well does the surrogate predict the solver's verdict?

    python scripts/evaluate_prediction.py --model models/scout.pt

Compares the ensemble against two things an engineer would try first: the maximum link
utilisation on its own, and gradient boosting on hand-made summary statistics.  Reports
are given for the test split (same sizes as training) and for the large split (networks
about twice the size of anything seen in training).
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tsnscout.dataset import read_samples
from tsnscout.features import collate, handcrafted
from tsnscout.gnn import Ensemble
from tsnscout.problem import quick_infeasible
from tsnscout.train import encode_samples, expected_calibration_error, predict_logits


def within_instance_ranking(samples, score) -> float:
    """Share of (schedulable, unschedulable) routing pairs *of the same instance* that the
    score orders correctly.  This is the skill search needs: not telling easy networks from
    hard ones, but telling good routings from bad ones on one network."""
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[id(s.inst)].append(i)
    right = total = 0.0
    for idx in groups.values():
        pos = [i for i in idx if samples[i].label == 1]
        neg = [i for i in idx if samples[i].label == 0]
        for p in pos:
            for n in neg:
                total += 1
                right += 1.0 if score[p] > score[n] else 0.5 if score[p] == score[n] else 0.0
    return right / total if total else float("nan")


def metrics(samples, y, prob, name):
    hard = np.array([quick_infeasible(s.inst, s.routing) is None for s in samples])
    decided = np.array([s.status != "unknown" for s in samples])
    out = {
        "auc": roc_auc_score(y, prob),
        "auc_nontrivial": roc_auc_score(y[hard], prob[hard]),
        "auc_decided": roc_auc_score(y[decided], prob[decided]),
        "accuracy": float(((prob > 0.5) == y).mean()),
        "ranking": within_instance_ranking(samples, prob),
        "ece": expected_calibration_error(prob, y),
    }
    print(f"  {name:22s} AUC {out['auc']:.4f}  non-trivial {out['auc_nontrivial']:.4f}  acc {out['accuracy']:.3f}"
          f"  within-instance ranking {out['ranking']:.3f}  ECE {out['ece']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/scout.pt")
    ap.add_argument("--train", nargs="+", default=["data/train.jsonl"])
    ap.add_argument("--splits", nargs="+", default=["test", "test_large"])
    ap.add_argument("--out", default="results/prediction.json")
    args = ap.parse_args()
    torch.set_num_threads(1)

    ens = Ensemble.load(args.model)
    train = read_samples(args.train)
    x_train = np.array([handcrafted(s.inst, s.routing) for s in train])
    y_train = np.array([s.label for s in train])
    util_only = LogisticRegression().fit(x_train[:, :1], y_train)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(x_train, y_train)

    report = {"model": args.model, "n_train": len(train), "splits": {}}
    for split in args.splits:
        samples = read_samples([f"data/{split}.jsonl"])
        graphs, y, log_t = encode_samples(samples)
        z, t_pred = predict_logits(ens.models, graphs)
        z = z / ens.temperature
        prob = 1 / (1 + np.exp(-z.mean(0)))
        x = np.array([handcrafted(s.inst, s.routing) for s in samples])
        statuses = [s.status for s in samples]
        print(f"{split}: {len(samples)} samples, {y.mean():.1%} schedulable, "
              f"{statuses.count('unknown') / len(samples):.1%} solver timeouts")
        res = {
            "n": len(samples), "schedulable": float(y.mean()),
            "timeouts": statuses.count("unknown") / len(samples),
            "mean_flows": float(np.mean([s.inst.n_flows for s in samples])),
            "mean_links": float(np.mean([s.inst.n_links for s in samples])),
            "methods": {
                "max utilisation": metrics(samples, y, util_only.predict_proba(x[:, :1])[:, 1], "max utilisation"),
                "gradient boosting": metrics(samples, y, gbm.predict_proba(x)[:, 1], "gradient boosting"),
                "single GNN": metrics(samples, y, 1 / (1 + np.exp(-z[0])), "single GNN"),
                "GNN ensemble": metrics(samples, y, prob, "GNN ensemble"),
            },
        }
        # Calibration curve and selective prediction for the ensemble.
        edges = np.linspace(0, 1, 11)
        res["reliability"] = [
            {"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
             "predicted": float(prob[m].mean()) if m.any() else None,
             "observed": float(y[m].mean()) if m.any() else None}
            for lo, hi in zip(edges[:-1], edges[1:])
            for m in [(prob >= lo) & (prob < hi if hi < 1 else prob <= hi)]
        ]
        confidence = np.abs(prob - 0.5)
        order = np.argsort(-confidence)
        correct = ((prob > 0.5) == y)[order]
        res["selective"] = [
            {"coverage": c, "accuracy": float(correct[: max(1, int(c * len(y)))].mean())}
            for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        ]
        res["runtime_spearman"] = float(spearmanr(t_pred.mean(0), log_t).correlation)
        # Speed: surrogate versus exact solver, both on one CPU thread.
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, min(len(graphs), 512), 64):
                ens.logits(collate(graphs[i : i + 64]))
        per_graph = (time.perf_counter() - t0) / min(len(graphs), 512)
        solver = np.array([s.seconds for s in samples])
        res["speed"] = {
            "surrogate_ms": per_graph * 1e3, "solver_mean_ms": float(solver.mean() * 1e3),
            "solver_median_ms": float(np.median(solver) * 1e3), "speedup_mean": float(solver.mean() / per_graph),
        }
        print(f"  ensemble: {per_graph * 1e3:.2f} ms per routing; solver: mean {solver.mean() * 1e3:.0f} ms, "
              f"median {np.median(solver) * 1e3:.0f} ms; run-time rank correlation {res['runtime_spearman']:.2f}")
        res["scores"] = {"prob": prob.round(4).tolist(), "label": y.astype(int).tolist(),
                         "max_util": x[:, 0].round(4).tolist(), "std": z.std(0).round(4).tolist()}
        report["splits"][split] = res

    with open(args.out, "w") as fh:
        json.dump(report, fh)
    print("saved", args.out)


if __name__ == "__main__":
    main()
