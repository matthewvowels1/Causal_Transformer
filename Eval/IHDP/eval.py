import networkx as nx
import numpy
import numpy as np

from Eval.eval_utils import train_model
from CaT.datasets import reorder_dag, get_full_ordering
from utils.inference import CausalInference
from CaT.test_model import CaT
from loader import load_IHDP
import torch


def instantiate_model(device):
    dropout_rate = 0.0
    ff_n_embed = 4
    num_heads = 3
    n_layers = 1
    embed_dim = 5
    head_size = 4
    input_dim = 1

    DAGnx = nx.DiGraph()

    # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
    var_types = {
        'x': 'bin',
        'y': 'cont',
        **{str(i): 'bin' for i in range(7, 26)},
        **{str(i): 'cont' for i in range(1, 7)}
    }

    DAGnx.add_edges_from(
        [('x', 'y')] + [(str(i), 'x') for i in range(1, 26)] + [(str(i), 'y') for i in range(1, 26)])
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    print(causal_ordering)
    model = CaT(
        input_dim=input_dim,
        embed_dim=embed_dim,
        dropout_rate=dropout_rate,
        head_size=head_size,
        num_heads=num_heads,
        ff_n_embed=ff_n_embed,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        n_layers=n_layers,
        device=device,
        var_types=var_types,
        activation_function='Swish'
    ).to(device)
    return model


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    device = 'cuda'
    for i in range(1, 101):
        model = instantiate_model(device=device)
        dataset = {}
        true_ite = {}
        for split in ('train', 'test'):
            z, x, y, y1, y0 = load_IHDP(path="data/", replication=i, split=split)
            data = numpy.concatenate([z, x, y], axis=1)
            data = data.reshape([-1, 27, 1])
            dataset[split] = data
            true_ite[split] = y1 - y0

        train_model(model=model, train=dataset['train'], test=dataset['test'], device=device)

        for split in ('train', 'test'):
            ci = CausalInference(model=model, device=device)

            D0 = ci.forward(data=dataset[split], intervention_nodes_vals={'x': 0})
            D1 = ci.forward(data=dataset[split], intervention_nodes_vals={'x': 1})

            output0 = D0[:, -1, :].reshape([-1])
            output1 = D1[:, -1, :].reshape([-1])

            est_ite = output1 - output0
            est_ate = est_ite.mean()

            pehe = np.sqrt(
                np.mean((true_ite[split].squeeze() - est_ite) * (true_ite[split].squeeze() - est_ite)))
            eate = np.abs(true_ite[split].mean() - est_ate)

            with open("output.txt", "w") as file:
                file.write(f"i: {i}\nsplit: {split}\npehe: {pehe}\neate: {eate}\n")
