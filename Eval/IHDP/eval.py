import os
from collections import defaultdict

import networkx as nx
import numpy
import numpy as np

from Eval.eval_utils import train_model, instantiate_CaT, predict
from CaT.datasets import reorder_dag
from utils.inference import CausalInference
from Eval.IHDP.loader import load_IHDP
import torch

from utils.metrics import pehe, eate


def evaluate(model_constructor, seed=0):
    results = defaultdict(list)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    DAGnx = nx.DiGraph()

    # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
    var_types = {
        **{str(i): 'cont' for i in range(1, 7)},
        **{str(i): 'bin' for i in range(7, 26)},
        't': 'bin',
        'y': 'cont',
    }
    DAGnx.add_edges_from(
        [('t', 'y')] + [(str(i), 't') for i in range(1, 26)] + [(str(i), 'y') for i in range(1, 26)])
    DAGnx = reorder_dag(dag=DAGnx)
    perm_map = [list(var_types.keys()).index(i) for i in DAGnx.nodes()]

    for i in range(1, 101):
        dataset = {}
        y_potential = {}

        for split_name in ('train', 'test'):
            z, t, y, y1, y0 = load_IHDP(replication=i, split=split_name)
            data = numpy.concatenate([z, t, y], axis=1)
            data = data.reshape([-1, 27])
            data = data[:, perm_map]
            dataset[split_name] = data
            y_potential[split_name] = np.concatenate((y0[-1:None], y1[-1:None]),axis=1)

        ci = CausalInference(dag=DAGnx)

        model = model_constructor(DAGnx=DAGnx, var_types=var_types)

        train_model(model=model, train=dataset['train'], test=dataset['test'], ci=ci)

        for split_name in ('train', 'test'):
            output0 = predict(model=model, data=dataset[split_name], ci=ci, interventions_nodes={'t': 0})
            output1 = predict(model=model, data=dataset[split_name], ci=ci, interventions_nodes={'t': 1})

            results[f'pehe {split_name}'].append(
                pehe(ypred0=output0, ypred1=output1, ypotential=y_potential[split_name]))
            results[f'eate {split_name}'].append(
                eate(ypred0=output0, ypred1=output1, ypotential=y_potential[split_name]))
    return results

if __name__ == "__main__":
    evaluate(instantiate_CaT)
