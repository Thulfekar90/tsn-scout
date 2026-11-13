"""Training and evaluation of the surrogate ensemble."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .dataset import Sample
from .features import InstanceEncoder, collate
from .gnn import Ensemble, ScoutNet

LOG_T_FLOOR = -3.0  # solver times are clipped to 1 ms before taking log10


def encode_samples(samples: Sequence[Sample]) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Encode samples once; returns (graphs, labels, log10 solver seconds)."""
    cache: Dict[int, InstanceEncoder] = {}
    graphs = []
    for s in samples:
        enc = cache.get(id(s.inst))
        if enc is None:
            enc = cache[id(s.inst)] = InstanceEncoder(s.inst)
        graphs.append(enc.encode_numpy(np.asarray(s.routing)[None, :]))
    y = np.array([s.label for s in samples], dtype=np.float32)
    t = np.array([max(LOG_T_FLOOR, math.log10(max(s.seconds, 1e-9))) for s in samples], dtype=np.float32)
    return graphs, y, t


@torch.no_grad()
def predict_logits(models: Sequence[ScoutNet], graphs: Sequence[dict], batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Returns logits and log-time predictions, each of shape (n_models, n_samples)."""
    for m in models:
        m.eval()
    z, t = [], []
    for i in range(0, len(graphs), batch_size):
        batch = collate(graphs[i : i + batch_size])
        outs = [m(batch) for m in models]
        z.append(torch.stack([o[0] for o in outs]))
        t.append(torch.stack([o[1] for o in outs]))
    return torch.cat(z, 1).numpy(), torch.cat(t, 1).numpy()


def train_one(
    train: Tuple[List[dict], np.ndarray, np.ndarray],
    val: Tuple[List[dict], np.ndarray, np.ndarray],
    seed: int,
    epochs: int = 40,
    hidden: int = 64,
    rounds: int = 4,
    lr: float = 2e-3,
    batch_size: int = 128,
    time_weight: float = 0.1,
    log=print,
) -> Tuple[ScoutNet, dict]:
    """Train one network; keep the weights of the epoch with the best validation AUC."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    graphs, y, t = train
    model = ScoutNet(hidden, rounds)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(graphs) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    bce, huber = nn.BCEWithLogitsLoss(), nn.HuberLoss()
    best_auc, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(graphs))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = collate([graphs[j] for j in idx])
            logit, log_t = model(batch)
            loss = bce(logit, torch.from_numpy(y[idx])) + time_weight * huber(log_t, torch.from_numpy(t[idx]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        z, _ = predict_logits([model], val[0])
        auc = roc_auc_score(val[1], z[0])
        history.append({"epoch": epoch, "loss": total / len(order), "val_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs:
            log(f"  seed {seed} epoch {epoch:3d}  loss {total / len(order):.4f}  val AUC {auc:.4f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_auc": best_auc, "history": history}


def fit_temperature(z_mean: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: one scalar that makes validation probabilities honest."""
    z = torch.from_numpy(z_mean.astype(np.float32))
    target = torch.from_numpy(y.astype(np.float32))
    log_temp = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(z / log_temp.exp(), target)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_temp.exp())


def _train_member(args):
    """Worker process: train one ensemble member on a single thread."""
    train, val, seed, kwargs = args
    torch.set_num_threads(1)
    model, info = train_one(train, val, seed=seed, log=lambda *a: None, **kwargs)
    return model.state_dict(), info


def train_ensemble(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    n_models: int = 5,
    log=print,
    parallel: bool = True,
    **kwargs,
) -> Tuple[Ensemble, dict]:
    """Train ``n_models`` networks that differ only in their random seed.

    Members train in separate processes: they are independent, so this is the cheapest
    speed-up available on a laptop CPU.
    """
    train = encode_samples(train_samples)
    val = encode_samples(val_samples)
    hidden, rounds = kwargs.get("hidden", 64), kwargs.get("rounds", 4)
    t0 = time.time()
    if parallel and n_models > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(n_models) as pool:
            results = pool.map(_train_member, [(train, val, seed, kwargs) for seed in range(n_models)])
        models, infos = [], []
        for state, info in results:
            model = ScoutNet(hidden, rounds)
            model.load_state_dict(state)
            models.append(model)
            infos.append(info)
    else:
        models, infos = [], []
        for seed in range(n_models):
            model, info = train_one(train, val, seed=seed, log=log, **kwargs)
            models.append(model)
            infos.append(info)
    z, _ = predict_logits(models, val[0])
    temperature = fit_temperature(z.mean(0), val[1])
    ens = Ensemble(models, temperature)
    report = {
        "n_train": len(train_samples), "n_val": len(val_samples), "temperature": temperature,
        "train_seconds": time.time() - t0,
        "member_val_auc": [i["best_val_auc"] for i in infos],
        "ensemble_val_auc": float(roc_auc_score(val[1], z.mean(0))),
        "history": [i["history"] for i in infos],
    }
    log(f"members val AUC {[round(a, 4) for a in report['member_val_auc']]}  ensemble {report['ensemble_val_auc']:.4f}"
        f"  temperature {temperature:.2f}  ({report['train_seconds']:.0f}s)")
    return ens, report


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)
