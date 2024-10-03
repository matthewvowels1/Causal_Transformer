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

                # Expand along a new dimension
                final_output = np.expand_dims(permuted, axis=2)

                return final_output

            train = preprocess(train)
            test = preprocess(test)

            model = model_constructor(device=device, var_types=var_types, DAGnx=DAGnx)

            train_model(model=model, train=train[:, :-1, :], test=test[:, :-1, :], device=device)

            for name_split, split in {'train': train, 'test': test}.items():
                # take only RCT
                split = split[split[:, -1, 0] == 1][:, :-1, :]
                ci = CausalInference(model=model, device=device)

                D0 = ci.forward(data=split, intervention_nodes_vals={'t': 0})
                D1 = ci.forward(data=split, intervention_nodes_vals={'t': 1})

                output0 = D0[:, -1, :].reshape([-1])
                output1 = D1[:, -1, :].reshape([-1])

                R = policy_val(ypred1=output1, ypred0=output0, y=split[:, -1, 0], t=split[:, -2, 0])
                eatt = compute_eatt(ypred1=output1, ypred0=output0, y=split[:, -1, 0], t=split[:, -2, 0])


                file.write(f"random_state: {random_state}\nsplit: {name_split}\nrisk: {R}\neatt: {eatt}\n")


if __name__ == "__main__":
    evaluate(instantiate_CaT)
