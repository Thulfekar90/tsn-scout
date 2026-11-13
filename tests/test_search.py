import numpy as np
import torch

from tsnscout.checker import is_valid
from tsnscout.generator import generate
from tsnscout.gnn import Ensemble, ScoutNet
from tsnscout.search import ScoutConfig, balanced_proposals, run_method, scout_proposals, verify


def untrained_ensemble():
    torch.manual_seed(0)
    return Ensemble([ScoutNet(16, 2), ScoutNet(16, 2)])


def test_scout_returns_distinct_legal_routings(tiny_cfg):
    inst = generate(3, tiny_cfg)
    proposals, info = scout_proposals(
        inst, untrained_ensemble(), 5, np.random.default_rng(0), ScoutConfig(chains=8, steps=10, rescore=20)
    )
    assert 1 <= len(proposals) <= 5
    assert len({tuple(p) for p in proposals}) == len(proposals)
    for routing in proposals:
        assert all(0 <= routing[f] < inst.n_candidates(f) for f in range(inst.n_flows))
    assert info["scores"] == sorted(info["scores"], reverse=True)


def test_whatever_is_reported_solved_is_really_solved(tiny_cfg):
    """Even a useless, untrained surrogate cannot produce a wrong answer: the solver and
    the independent checker stand between its proposals and the result."""
    solved = 0
    for seed in range(8):
        inst = generate(seed, tiny_cfg)
        for method in ("shortest", "random", "balanced", "scout"):
            out = run_method(
                method, inst, 3, np.random.default_rng(seed), untrained_ensemble(), time_limit=5,
                scout_cfg=ScoutConfig(chains=8, steps=5, rescore=16),
            )
            assert out.solver_calls <= 3
            if out.solved:
                solved += 1
                assert is_valid(inst, out.routing, out.schedule)
    assert solved > 0


def test_verify_stops_at_the_first_success(tiny_cfg):
    for seed in range(20):
        inst = generate(seed, tiny_cfg)
        proposals = balanced_proposals(inst, 3, np.random.default_rng(seed))
        out = verify(inst, proposals + proposals, time_limit=5)
        if out.solved:
            assert out.trace[-1][1] == "feasible"
            assert all(status != "feasible" for _, status in out.trace[:-1])
            return
    raise AssertionError("no solvable instance found")
