"""The solver is the ground truth of the whole project, so it is tested three ways:
against an independent checker, against a second formulation, and on cases solved by hand."""

import numpy as np
import pytest

from tsnscout.checker import is_valid, violations
from tsnscout.generator import generate, sample_routing
from tsnscout.problem import Flow, Instance, quick_infeasible
from tsnscout.solver import FEASIBLE, INFEASIBLE, INTERVAL, MODULAR, UNKNOWN, solve, solve_joint


def two_flows_one_link(len_a: int, len_b: int) -> Instance:
    """Flows with periods 2 ms and 5 ms share one link; gcd = 1 ms = 100 ticks."""
    return Instance(
        n_nodes=2, is_switch=[False, False], links=[(0, 1)],
        flows=[Flow(0, 1, 200, len_a, 200), Flow(0, 1, 500, len_b, 500)],
        routes=[[[0]], [[0]]],
    )


@pytest.mark.parametrize("formulation", [MODULAR, INTERVAL])
def test_windows_must_fit_in_the_gcd(formulation):
    # 60 + 50 > 100: some instance of one flow always hits the other.
    assert solve(two_flows_one_link(60, 50), [0, 0], formulation=formulation).status == INFEASIBLE
    # 50 + 50 = 100: fits exactly, with no tick to spare.
    res = solve(two_flows_one_link(50, 50), [0, 0], formulation=formulation)
    assert res.status == FEASIBLE and is_valid(two_flows_one_link(50, 50), [0, 0], res.schedule)


def test_three_way_conflict_that_no_pair_reveals():
    # Periods 1, 2 and 5 ms: every pair must be disjoint modulo 1 ms, so the three windows
    # need 40 + 40 + 30 = 110 ticks of a 100-tick window.  Each pair alone fits.
    inst = Instance(
        n_nodes=2, is_switch=[False, False], links=[(0, 1)],
        flows=[Flow(0, 1, 100, 30, 100), Flow(0, 1, 200, 40, 200), Flow(0, 1, 500, 40, 500)],
        routes=[[[0]], [[0]], [[0]]],
    )
    assert quick_infeasible(inst, [0, 0, 0]) is None
    assert solve(inst, [0, 0, 0]).status == INFEASIBLE


def test_schedules_pass_the_independent_checker(tiny_cfg):
    found = 0
    for seed in range(40):
        inst = generate(seed, tiny_cfg)
        routing = sample_routing(inst, np.random.default_rng(seed), "partial")
        res = solve(inst, routing, time_limit=5)
        if res.status == FEASIBLE:
            found += 1
            assert violations(inst, routing, res.schedule) == []
    assert found >= 10  # the test would be vacuous otherwise


def test_formulations_agree(tiny_cfg):
    verdicts = set()
    for seed in range(40):
        inst = generate(seed, tiny_cfg)
        routing = sample_routing(inst, np.random.default_rng(seed), "random")
        a = solve(inst, routing, time_limit=5, formulation=MODULAR).status
        b = solve(inst, routing, time_limit=5, formulation=INTERVAL).status
        if UNKNOWN not in (a, b):
            assert a == b, f"seed {seed}: modular says {a}, interval says {b}"
            verdicts.add(a)
    assert verdicts == {FEASIBLE, INFEASIBLE}  # both outcomes were actually exercised


def test_checker_catches_a_collision(tiny_cfg):
    for seed in range(40):
        inst = generate(seed, tiny_cfg)
        routing = inst.shortest_routing()
        res = solve(inst, routing, time_limit=5)
        shared = [u for u in inst.link_users(routing) if len(u) >= 2]
        if res.status == FEASIBLE and shared:
            (fa, ha), (fb, hb) = shared[0][:2]
            broken = dict(res.schedule)
            broken[fb, hb] = min(broken[fa, ha], inst.flows[fb].period - inst.flows[fb].length)
            assert not is_valid(inst, routing, broken)
            return
    pytest.fail("no instance with a shared link was found")


def test_quick_checks_are_sound(tiny_cfg):
    import dataclasses

    heavy = dataclasses.replace(tiny_cfg, n_flows=(30, 45), load_scale=(1.2, 1.4))
    flagged = 0
    for seed in range(30):
        inst = generate(seed, heavy)
        routing = inst.shortest_routing()
        if quick_infeasible(inst, routing) is not None:
            flagged += 1
            assert solve(inst, routing, time_limit=10).status != FEASIBLE
    assert flagged >= 3


def test_joint_solver_is_consistent_with_fixed_routing(tiny_cfg):
    solved = 0
    for seed in range(20):
        inst = generate(seed, tiny_cfg)
        joint = solve_joint(inst, time_limit=10)
        if joint.status == FEASIBLE:
            solved += 1
            assert is_valid(inst, joint.routing, joint.schedule)
        fixed = solve(inst, inst.shortest_routing(), time_limit=5)
        if fixed.status == FEASIBLE:
            assert joint.status != INFEASIBLE  # a routing exists, so the joint problem has a solution
    assert solved >= 5
