import networkx as nx
import numpy as np

from utils.inference import CausalInference
from loader import load_JOBS
import torch
from Eval.eval_utils import train_model, instantiate_CaT

from numpy.typing import NDArray


def policy_val(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
               y: NDArray[np.float_], t: NDArray[np.int_]) -> float:
    # Adapted from https://github.com/clinicalml/cfrnet/
    # Mean outcome for treated (t == 1) and control (t == 0) groups
    y1_mean = np.mean(y[t == 1])
    y0_mean = np.mean(y[t == 0])

    # Proportion of times ypred1 is greater than ypred0
    p_fx1 = np.mean(ypred1 > ypred0)

    # Calculate policy risk (1 - policy value)
    policy_risk = 1 - (y1_mean * p_fx1 + y0_mean * (1 - p_fx1))

    return policy_risk


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    device = 'cuda'

    train, test, contfeats, bin_feats = load_JOBS("data")


    def preprocess(data):
        # Expand dims for data[1] and data[2]
        data1_exp = np.expand_dims(data[1], axis=1)
        data2_exp = np.expand_dims(data[2], axis=1)

        # Concatenate along axis 1
        concatenated = np.concatenate([data[0], data1_exp, data2_exp], axis=1)

        # Expand along a new dimension
        final_output = np.expand_dims(concatenated, axis=2)

        return final_output


    train = preprocess(train)
    test = preprocess(test)

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

    model = instantiate_CaT(device=device, var_types=var_types, DAGnx=DAGnx)

    train_model(model=model, train=train, test=test, device=device)

    for name_split, split in {'train':train,'test':test}.items():
        ci = CausalInference(model=model, device=device)

        D0 = ci.forward(data=split, intervention_nodes_vals={'t': 0})
        D1 = ci.forward(data=split, intervention_nodes_vals={'t': 1})

        output0 = D0[:, -2, :].reshape([-1])
        output1 = D1[:, -2, :].reshape([-1])

        R = policy_val(ypred1=output1, ypred0=output0, y=split[:, -2, 0], t=split[:, -1, 0])

        with open("output.txt", "a") as file:
            file.write(f"split: {name_split}\nrisk: {R}\n")
