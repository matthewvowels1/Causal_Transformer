import networkx as nx
import torch
import numpy as np

from Eval.eval_utils import train_model, compute_eatt, instantiate_CaT, instantiate_old_CFCN
from Eval.twins.loader import load_twins
from utils.inference import CausalInference
from utils.utils import reorder_dag


def evaluate(model_constructor, output_path='output.txt', device='cuda', seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    df, var_types = load_twins(data_format='pandas')
    var_names = df.columns.to_list()

    data = df.to_numpy()

    DAGnx = nx.DiGraph()
    # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)

    DAGnx.add_edges_from(
        [(x, 't') for x in var_names if x not in ('t', 'y')] +
        [(x, 'y') for x in var_names if x != 'y']
    )
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag

    # Create a mapping from original to new order
    perm_map = [list(var_types.keys()).index(i) for i in list(DAGnx.nodes())]
    data = data[:, perm_map]

    shuffle_indices = np.random.permutation(len(data))
    data = data[shuffle_indices]
    split_ratio = 0.8
    train_size = int(len(data) * split_ratio)
    train = data[:train_size]
    test = data[train_size:]

    model = model_constructor(device=device, var_types=var_types, DAGnx=DAGnx)

    train_model(model=model, train_data=train, val_data=test, device=device)

    with open(output_path, "w") as file:
        for name_split, split in {'train': train, 'test': test}.items():
            ci = CausalInference(model=model, device=device)

            D0 = ci.forward(data=split, intervention_nodes_vals={'t': 0})
            D1 = ci.forward(data=split, intervention_nodes_vals={'t': 1})

            output0 = ci.get(D0, 'y')
            output1 = ci.get(D1, 'y')

            eatt = compute_eatt(ypred1=output1, ypred0=output0, y=ci.get(split, 'y'), t=ci.get(split, 't'))
            file.write(f"split: {name_split}\neatt: {eatt}\n\n")
if __name__ == "__main__":
    evaluate(instantiate_old_CFCN)