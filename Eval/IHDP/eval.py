import networkx as nx
import numpy
import numpy as np

from Eval.eval_utils import train_model, instantiate_CaT
from CaT.datasets import reorder_dag, get_full_ordering
from utils.inference import CausalInference
from loader import load_IHDP
import torch


if __name__ == "__main__":
    seed = 4
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    device = 'cuda'

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


    for i in range(1, 101):
        model = instantiate_CaT(device=device, var_types=var_types, DAGnx= DAGnx, embed_dim=20, n_layers=2)
        dataset = {}
        true_ite = {}
        for split in ('train', 'test'):
            z, x, y, y1, y0 = load_IHDP(path="data/", replication=i, split=split)
            data = numpy.concatenate([z, x, y], axis=1)
            data = data.reshape([-1, 27, 1])
            dataset[split] = data
            true_ite[split] = y1 - y0

        train_model(model=model, train=dataset['train'], test=dataset['test'], device=device, max_iters=5000)

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

            with open("output.txt", "a") as file:
                file.write(f"i: {i}\nsplit: {split}\npehe: {pehe}\neate: {eate}\n")
