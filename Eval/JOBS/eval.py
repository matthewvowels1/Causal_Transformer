from collections import defaultdict

import networkx as nx
import numpy as np

from CaT.datasets import reorder_dag
from utils.inference import CausalInference
from Eval.JOBS.loader import load_JOBS
import torch
from Eval.eval_utils import train_model, instantiate_CaT, predict
from utils.metrics import risk, eatt


def evaluate(model_constructor, seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    results = defaultdict(list)
    for random_state in range(100):

        train, test, contfeats, bin_feats = load_JOBS(random_state=random_state)

        DAGnx = nx.DiGraph()
        # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
        var_types = {
            **{str(i): 'cont' for i in contfeats},
            **{str(i): 'bin' for i in bin_feats},
            't': 'bin',
            'y': 'cont',
        }

        DAGnx.add_edges_from(
            [('t', 'y')] + [(str(i), 't') for i in bin_feats + contfeats] + [(str(i), 'y') for i in
                                                                             bin_feats + contfeats])
        DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag

        # Create a mapping from original to new order
        perm_map = [(list(var_types.keys()) + ['e']).index(i) for i in list(DAGnx.nodes()) + ['e']]

        def preprocess(data):
            # Expand dims for data[1] and data[2]
            data1_exp = np.expand_dims(data[1], axis=1)
            data2_exp = np.expand_dims(data[2], axis=1)
            data3_exp = np.expand_dims(data[3], axis=1)

            # Concatenate along axis 1
            concatenated = np.concatenate([data[0], data1_exp, data2_exp, data3_exp], axis=1)

            # Apply the permutation
            permuted = concatenated[:, perm_map]

            return permuted

        train = preprocess(train)
        test = preprocess(test)

        ci = CausalInference(dag=DAGnx)

        model = model_constructor(DAGnx=DAGnx, var_types=var_types)

        train_model(model=model, train=train[:, :-1], test=test[:, :-1], ci=ci)

        for name_split, split in {'train': train, 'test': test}.items():
            # take only RCT
            split = split[split[:, -1] == 1][:, :-1]

            output0 = predict(model=model, data=split, ci=ci, interventions_nodes={'t': 0})
            output1 = predict(model=model, data=split, ci=ci, interventions_nodes={'t': 1})

            results[f'risk {name_split}'].append(
                risk(ypred1=output1, ypred0=output0, y=ci.get(split, 'y'), t=ci.get(split, 't')))
            results[f'eatt {name_split}'].append(
                eatt(ypred1=output1, ypred0=output0, y=ci.get(split, 'y'), t=ci.get(split, 't')))
    return results


if __name__ == "__main__":
    evaluate(instantiate_CaT)
