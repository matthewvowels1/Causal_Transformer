from collections import defaultdict

import networkx as nx
import torch
import numpy as np

from Eval.eval_utils import train_model, instantiate_old_CFCN, predict
from utils.metrics import eate, pehe
from Eval.twins.loader import load_twins
from utils.inference import CausalInference
from utils.utils import reorder_dag


def evaluate(model_constructor, seeds=range(10), test_ratio=0.2):
    results = defaultdict(list)
    for seed in seeds:
        np.random.seed(seed=seed)
        torch.manual_seed(seed)

        data, y_potential, var_types = load_twins()
        var_names = var_types.keys()

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

        test_size = int(len(data) * test_ratio)
        shuffled_idx = np.random.permutation(len(data))
        test_idx = shuffled_idx[:test_size]
        train_idx = shuffled_idx[test_size:]


        dataset = {}
        data_y_potential = {}
        for name_split, split_idx in {'test': test_idx, 'train': train_idx}.items():
            dataset[name_split] = data[split_idx]
            data_y_potential[name_split] = y_potential[split_idx]

        ci = CausalInference(dag=DAGnx)

        model = model_constructor(DAGnx=DAGnx, var_types=var_types)

        train_model(model=model, train=dataset['train'], test=dataset['test'], ci=ci)

        for name_split in ('train', 'test'):
            output0 = predict(model=model, data=dataset[name_split], ci=ci, interventions_nodes={'t': 0})
            output1 = predict(model=model, data=dataset[name_split], ci=ci, interventions_nodes={'t': 1})

            results[f'eate {name_split}'].append(
                eate(ypred1=output1, ypred0=output0, ypotential=data_y_potential[name_split]))
            results[f'pehe {name_split}'].append(
                pehe(ypred1=output1, ypred0=output0, ypotential=data_y_potential[name_split]))
    return results


if __name__ == "__main__":
    evaluate(instantiate_old_CFCN)
