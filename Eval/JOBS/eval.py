import networkx as nx
import numpy as np

from CaT.datasets import reorder_dag
from utils.inference import CausalInference
from loader import load_JOBS
import torch
from Eval.eval_utils import train_model, instantiate_CaT, instantiate_new_CFCN, instantiate_old_CFCN

from numpy.typing import NDArray


def safe_mean(arr):
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return 0
    return np.nanmean(arr)


def policy_val(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
               y: NDArray[np.float_], t: NDArray[np.int_]) -> float:
    # ypred, y and t should be RCT
    # Adapted from https://github.com/clinicalml/cfrnet/
    # Determine where ypred1 is better than ypred0
    better_pred = ypred1 > ypred0

    # Mean outcome for treated group (t == 1) where ypred1 > ypred0
    y1_mean = safe_mean(y[(t == 1) & better_pred])

    # Mean outcome for control group (t == 0) where ypred1 <= ypred0
    y0_mean = safe_mean(y[(t == 0) & ~better_pred])

    # Proportion of times ypred1 is better than ypred0
    p_fx1 = safe_mean(better_pred)

    # Calculate policy risk (1 - policy value)
    policy_risk = 1 - (y1_mean * p_fx1 + y0_mean * (1 - p_fx1))

    return policy_risk


def compute_eatt(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
                 y: NDArray[np.float_], t: NDArray[np.int_]) -> float:
    # ypred, y and t should be RCT

    true_att = safe_mean(y[t == 1]) - safe_mean(y[t == 0])

    estimated_att = safe_mean(ypred1[t == 1] - ypred0[t == 1])

    return abs(true_att - estimated_att)


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
