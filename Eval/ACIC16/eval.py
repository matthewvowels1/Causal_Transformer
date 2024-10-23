from collections import defaultdict

import networkx as nx
import torch
import numpy as np

from Eval.eval_utils import train_model, instantiate_old_CFCN, predict
from utils.metrics import eate, pehe
from Eval.ACIC16.loader import load_ACIC
from utils.inference import CausalInference
from utils.utils import reorder_dag
from tqdm.contrib import itertools


def evaluate(model_constructor, experiments=range(77), replications=range(10), seed=0, eval_ratio=0.3, test_ratio=0.1,
             verbose=True):
    results = defaultdict(list)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    intermediate_results = defaultdict(list)
    for replication, experiment in itertools.product(replications, experiments, desc="ACIC", disable=not verbose):

        data, y_potential, var_types = load_ACIC(experiment=experiment, replication=replication)

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

        sep_test = int(len(data) * test_ratio)
        sep_eval = int(len(data) * (eval_ratio + test_ratio))
        shuffled_idx = np.random.permutation(len(data))
        test_idx = shuffled_idx[:sep_test]
        eval_idx = shuffled_idx[sep_test: sep_eval]
        train_idx = shuffled_idx[sep_eval:]

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

            r_eate = eate(ypred1=output1, ypred0=output0, ypotential=data_y_potential[name_split])

            results[f'eate {name_split}'].append(r_eate)
            intermediate_results[f'eate {name_split}|{replication}'].append(r_eate)
            r_pehe = pehe(ypred1=output1, ypred0=output0, ypotential=data_y_potential[name_split])
            results[f'pehe {name_split}'].append(r_pehe)
            intermediate_results[f'pehe {name_split}|{replication}'].append(r_pehe)

    for key, values in intermediate_results.items():
        mean = np.mean(values)
        prefix = key.split('|')[0]
        results[prefix + " experiment average"].append(mean)
    return results


if __name__ == "__main__":
    evaluate(instantiate_old_CFCN)
