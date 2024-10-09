import networkx as nx
import numpy
import numpy as np

from Eval.eval_utils import train_model, instantiate_CaT
from CaT.datasets import reorder_dag, get_full_ordering
from utils.inference import CausalInference
from loader import load_IHDP
import torch

def evaluate(model_constructor, output_path='output.txt', device='cuda', seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    DAGnx = nx.DiGraph()

    # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
    var_types = {
        **{str(i): 'cont' for i in range(1, 7)},
        **{str(i): 'bin' for i in range(7, 26)},
        'x': 'bin',
        'y': 'cont',
    }
    print(list(var_types.keys()))
    DAGnx.add_edges_from(
        [('x', 'y')] + [(str(i), 'x') for i in range(1, 26)] + [(str(i), 'y') for i in range(1, 26)])
    DAGnx = reorder_dag(dag=DAGnx)
    perm_map = [list(var_types.keys()).index(i) for i in DAGnx.nodes()]

    with open(output_path, "w") as file:
        for i in range(1, 101):
            model = model_constructor(device=device, var_types=var_types, DAGnx= DAGnx)
            dataset = {}
            true_ite = {}


            for split in ('train', 'test'):
                z, x, y, y1, y0 = load_IHDP(path="data/", replication=i, split=split)
                data = numpy.concatenate([z, x, y], axis=1)
                data = data.reshape([-1, 27])
                data = data[:, perm_map]
                dataset[split] = data
                true_ite[split] = y1 - y0

            train_model(model=model, train_data=dataset['train'], val_data=dataset['test'], device=device)

            for split in ('train', 'test'):
                ci = CausalInference(model=model, device=device)

                D0 = ci.forward(data=dataset[split], intervention_nodes_vals={'x': 0})
                D1 = ci.forward(data=dataset[split], intervention_nodes_vals={'x': 1})

                output0 = ci.get(D0, 'y')
                output1 = ci.get(D1, 'y')

                est_ite = output1 - output0
                est_ate = est_ite.mean()

                pehe = np.sqrt(
                    np.mean((true_ite[split].squeeze() - est_ite) * (true_ite[split].squeeze() - est_ite)))
                eate = np.abs(true_ite[split].mean() - est_ate)

                file.write(f"i: {i}\nsplit: {split}\npehe: {pehe}\neate: {eate}\n")

if __name__ == "__main__":
    evaluate(instantiate_CaT)