import numpy as np

from tsnscout.baselines import balance_objective, balanced_routing
from tsnscout.generator import LARGE, SMALL, generate
from tsnscout.problem import Instance


def test_same_seed_same_instance():
    assert generate(7, SMALL).to_json() == generate(7, SMALL).to_json()
    assert generate(7, SMALL).to_json() != generate(8, SMALL).to_json()


def test_json_round_trip():
    inst = generate(3, SMALL)
    assert Instance.from_json(inst.to_json()).to_json() == inst.to_json()


def test_routes_are_real_paths_and_usable_alone():
    for cfg in (SMALL, LARGE):
        inst = generate(11, cfg)
        for f, flow in enumerate(inst.flows):
            assert inst.routes[f], "every flow needs at least one candidate"
            lengths = [len(r) for r in inst.routes[f]]
            assert lengths == sorted(lengths), "candidates are sorted by hop count"
            for p, route in enumerate(inst.routes[f]):
                nodes = [inst.links[route[0]][0]] + [inst.links[l][1] for l in route]
                assert nodes[0] == flow.src and nodes[-1] == flow.dst
                assert all(inst.links[a][1] == inst.links[b][0] for a, b in zip(route, route[1:]))
                assert len(set(nodes)) == len(nodes), "no loops"
                assert inst.min_latency(f, p) <= flow.deadline <= flow.period


def test_load_balancing_never_makes_the_objective_worse():
    for seed in range(5):
        inst = generate(seed, SMALL)
        before = balance_objective(np.asarray(inst.link_utilization(inst.shortest_routing())))
        after = balance_objective(np.asarray(inst.link_utilization(balanced_routing(inst))))
        assert after <= before + 1e-12
