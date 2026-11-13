"""Properties the surrogate must have by construction, checked on an untrained network."""

import numpy as np
import torch

from tsnscout.features import InstanceEncoder, collate, encode_sample
from tsnscout.generator import SMALL, generate, sample_routing
from tsnscout.gnn import Ensemble, ScoutNet, scatter_max, scatter_mean, scatter_sum
from tsnscout.problem import Instance


def test_scatter_ops_match_a_loop():
    src = torch.randn(50, 3)
    index = torch.randint(0, 7, (50,))
    index[index == 4] = 5  # leave node 4 without neighbours
    for fn, ref in ((scatter_sum, torch.sum), (scatter_mean, torch.mean), (scatter_max, torch.amax)):
        out = fn(src, index, 7)
        for node in range(7):
            rows = src[index == node]
            expect = ref(rows, dim=0) if len(rows) else torch.zeros(3)
            assert torch.allclose(out[node], expect, atol=1e-6)


def test_prediction_ignores_the_order_of_flows():
    torch.manual_seed(0)
    model = ScoutNet().eval()
    inst = generate(5, SMALL)
    routing = sample_routing(inst, np.random.default_rng(0), "random")
    perm = np.random.default_rng(1).permutation(inst.n_flows)
    shuffled = Instance(
        inst.n_nodes, inst.is_switch, inst.links,
        [inst.flows[i] for i in perm], [inst.routes[i] for i in perm], inst.hop_delay,
    )
    with torch.no_grad():
        a = model(collate([encode_sample(inst, routing)]))[0]
        b = model(collate([encode_sample(shuffled, [routing[i] for i in perm])]))[0]
    assert torch.allclose(a, b, atol=1e-4)


def test_batching_does_not_change_predictions():
    torch.manual_seed(0)
    model = ScoutNet().eval()
    graphs = []
    for seed in range(4):
        inst = generate(seed, SMALL)
        graphs.append(encode_sample(inst, sample_routing(inst, np.random.default_rng(seed), "random")))
    with torch.no_grad():
        together = model(collate(graphs))[0]
        alone = torch.cat([model(collate([g]))[0] for g in graphs])
    assert torch.allclose(together, alone, atol=1e-4)


def test_vectorised_encoder_matches_one_at_a_time():
    inst = generate(2, SMALL)
    rng = np.random.default_rng(0)
    routings = np.array([sample_routing(inst, rng, "random") for _ in range(6)])
    batch = InstanceEncoder(inst).encode(routings)
    single = collate([encode_sample(inst, r) for r in routings])
    for name in ("x_flow", "x_link", "e_attr"):
        assert torch.allclose(getattr(batch, name), getattr(single, name), atol=1e-6)
    for name in ("e_flow", "e_link", "flow_graph", "link_graph"):
        assert torch.equal(getattr(batch, name), getattr(single, name))


def test_link_utilisation_feature_is_exact():
    inst = generate(4, SMALL)
    routing = inst.shortest_routing()
    batch = InstanceEncoder(inst).encode(np.array([routing]))
    assert np.allclose(batch.x_link[:, 1].numpy(), inst.link_utilization(routing), atol=1e-5)


def test_ensemble_save_and_load(tmp_path):
    torch.manual_seed(0)
    ens = Ensemble([ScoutNet(16, 2), ScoutNet(16, 2)], temperature=1.3)
    path = str(tmp_path / "ens.pt")
    ens.save(path)
    again = Ensemble.load(path)
    inst = generate(1, SMALL)
    batch = InstanceEncoder(inst).encode(np.array([inst.shortest_routing()]))
    assert torch.allclose(ens.predict(batch)[0], again.predict(batch)[0])
    assert again.temperature == 1.3
