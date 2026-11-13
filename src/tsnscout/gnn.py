"""The surrogate: a message-passing network on the flow-link graph, in plain PyTorch.

Each round, links collect messages from the flows crossing them (who competes here?) and
flows collect messages from the links on their route (how crowded is my path?).  Links
aggregate with sum, mean *and* max: load adds up, but one oversized flow can break a link
on its own.  The graph-level readout uses max and mean pooling, never sum, so the output
does not drift when the network grows beyond the sizes seen in training.

No graph library is used; scatter operations are a few lines of ``index_add_``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .features import EDGE_DIM, FLOW_DIM, LINK_DIM, Batch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, src.shape[1], dtype=src.dtype, device=src.device).index_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    count = torch.zeros(n, dtype=src.dtype, device=src.device).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return scatter_sum(src, index, n) / count.clamp(min=1).unsqueeze(1)


def scatter_max(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.full((n, src.shape[1]), float("-inf"), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(0, index.unsqueeze(1).expand_as(src), src, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)  # nodes without neighbours


def mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out))


class Round(nn.Module):
    """One exchange of messages: flows -> links, then links -> flows."""

    def __init__(self, d: int):
        super().__init__()
        self.to_link = mlp(2 * d, d, d)
        self.link_update = mlp(4 * d, d, d)
        self.link_norm = nn.LayerNorm(d)
        self.to_flow = mlp(2 * d, d, d)
        self.flow_update = mlp(3 * d, d, d)
        self.flow_norm = nn.LayerNorm(d)

    def forward(self, h_flow, h_link, e_flow, e_link, e):
        m = self.to_link(torch.cat([h_flow[e_flow], e], dim=1))
        n = h_link.shape[0]
        agg = torch.cat([scatter_sum(m, e_link, n) / 8.0, scatter_mean(m, e_link, n), scatter_max(m, e_link, n)], dim=1)
        h_link = self.link_norm(h_link + self.link_update(torch.cat([h_link, agg], dim=1)))

        m = self.to_flow(torch.cat([h_link[e_link], e], dim=1))
        n = h_flow.shape[0]
        agg = torch.cat([scatter_mean(m, e_flow, n), scatter_max(m, e_flow, n)], dim=1)
        h_flow = self.flow_norm(h_flow + self.flow_update(torch.cat([h_flow, agg], dim=1)))
        return h_flow, h_link


class ScoutNet(nn.Module):
    """Predicts, for each graph, a schedulability logit and the solver's log10 run time."""

    def __init__(self, hidden: int = 64, rounds: int = 4):
        super().__init__()
        self.hidden, self.n_rounds = hidden, rounds
        self.enc_flow = mlp(FLOW_DIM, hidden, hidden)
        self.enc_link = mlp(LINK_DIM, hidden, hidden)
        self.enc_edge = mlp(EDGE_DIM, hidden, hidden)
        self.rounds = nn.ModuleList(Round(hidden) for _ in range(rounds))
        self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        h_flow, h_link, e = self.enc_flow(batch.x_flow), self.enc_link(batch.x_link), self.enc_edge(batch.e_attr)
        for layer in self.rounds:
            h_flow, h_link = layer(h_flow, h_link, batch.e_flow, batch.e_link, e)
        g = batch.n_graphs
        pooled = torch.cat([
            scatter_max(h_link, batch.link_graph, g), scatter_mean(h_link, batch.link_graph, g),
            scatter_max(h_flow, batch.flow_graph, g), scatter_mean(h_flow, batch.flow_graph, g),
        ], dim=1)
        out = self.head(pooled)
        return out[:, 0], out[:, 1]


class Ensemble:
    """A few independently initialised networks.  Their disagreement is the uncertainty.

    ``score`` is the mean logit minus ``beta`` standard deviations: a routing only ranks
    high if every member likes it.  Search otherwise drifts towards inputs where one
    network is wrong and confident.
    """

    def __init__(self, models: List[ScoutNet], temperature: float = 1.0):
        self.models = models
        self.temperature = temperature
        for m in models:
            m.eval()

    @torch.no_grad()
    def logits(self, batch: Batch) -> torch.Tensor:
        """(n_models, n_graphs) raw logits."""
        return torch.stack([m(batch)[0] for m in self.models])

    @torch.no_grad()
    def predict(self, batch: Batch, beta: float = 0.0):
        """Returns (probability, uncertainty, score) per graph."""
        z = self.logits(batch) / self.temperature
        mean = z.mean(0)
        std = z.std(0) if z.shape[0] > 1 else torch.zeros_like(mean)
        return torch.sigmoid(mean), std, mean - beta * std

    @torch.no_grad()
    def log_seconds(self, batch: Batch) -> torch.Tensor:
        return torch.stack([m(batch)[1] for m in self.models]).mean(0)

    def save(self, path: str) -> None:
        torch.save({
            "states": [m.state_dict() for m in self.models], "hidden": self.models[0].hidden,
            "rounds": self.models[0].n_rounds, "temperature": self.temperature,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Ensemble":
        ckpt = torch.load(path, map_location="cpu")
        models = []
        for state in ckpt["states"]:
            m = ScoutNet(ckpt["hidden"], ckpt["rounds"])
            m.load_state_dict(state)
            models.append(m)
        return cls(models, ckpt.get("temperature", 1.0))
