import networkx as nx
import numpy as np

from CaT.datasets import reorder_dag
from utils.inference import CausalInference
from loader import load_JOBS
import torch
from Eval.eval_utils import train_model, instantiate_CaT, policy_val, compute_eatt


def evaluate(model_constructor, output_path='output.txt', device='cuda', seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    with open(output_path, "w") as file:
        for random_state in range(100):

            train, test, contfeats, bin_feats = load_JOBS("data", random_state=random_state)

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

            model = model_constructor(device=device, var_types=var_types, DAGnx=DAGnx)

            train_model(model=model, train_data=train[:, :-1], val_data=test[:, :-1], device=device)

            for name_split, split in {'train': train, 'test': test}.items():
                # take only RCT
                split = split[split[:, -1] == 1][:, :-1]
                ci = CausalInference(dag=DAGnx)

                D0 = ci.forward(data=split, model=model, intervention_nodes_vals={'t': 0})
                D1 = ci.forward(data=split, model=model, intervention_nodes_vals={'t': 1})

                output0 = ci.get(D0, 'y')
                output1 = ci.get(D1, 'y')

                R = policy_val(ypred1=output1, ypred0=output0, y=ci.get(split, 'y'), t=ci.get(split, 't'))
                eatt = compute_eatt(ypred1=output1, ypred0=output0, y=ci.get(split, 'y'), t=ci.get(split, 't'))

                file.write(f"random_state: {random_state}\nsplit: {name_split}\nrisk: {R}\neatt: {eatt}\n")


if __name__ == "__main__":
    evaluate(instantiate_CaT)
