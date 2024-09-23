import networkx as nx
import numpy as np
import torch
from test_model import CaT
import pandas as pd
import matplotlib.pyplot as plt
from datasets import get_full_ordering, reorder_dag
import utils
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from utils import inference

shuffling = 0
seed = 1
standardize = 0
sample_size = 100000
batch_size = 300
max_iters = 10000
eval_interval = 1000
eval_iters = 100
validation_fraction = 0.1
np.random.seed(seed=seed)
torch.manual_seed(seed)
device = 'cuda'
dropout_rate = 0.0
learning_rate = 2e-4
ff_n_embed = 4
num_heads = 3
n_layers = 1
embed_dim = 1
head_size = 4
d = 1

x_shift = 0
b_shift = 0
y_shift = 0  # makes no difference


# def generate_data(N, d=3):
#     DAGnx = nx.DiGraph()
#
#     Ux = np.random.randn(N,d)
#     X = (Ux > 0).astype(float)
#
#
#     Ub = np.random.randn(N,d)
#     B =  Ub
#
#     Uc = np.random.randn(N,d)
#     C =  Uc
#
#     Uy = np.random.randn(N,d)
#     Y = 0.3 * X + 0.6 * B + 1.2 * C +  0.01 * Uy
#
#     Y0 = 0.3 * 0 + 0.6 * B + 1.2 * C +  0.01 * Uy
#     Y1 =  0.3 * 1 + 0.6 * B + 1.2 * C + 0.01 * Uy
#
#     all_data_dict = {'X': X, 'B': B, 'C': C, 'Y': Y}
#
#     # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
#     var_types = {'X': 'cont', 'B': 'cont', 'C': 'cont', 'Y': 'cont'}
#
#     DAGnx.add_edges_from([('X', 'Y'), ('B', 'Y'), ('C', 'Y')])
#     DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
#     var_names = list(DAGnx.nodes())  # topologically ordered list of variables
#     all_data = np.stack([all_data_dict[key] for key in var_names], axis=1)
#     causal_ordering = get_full_ordering(DAGnx)
#     ordered_var_types = dict(sorted(var_types.items(), key=lambda item: causal_ordering[item[0]]))
#
#     return all_data, DAGnx, var_names, causal_ordering, ordered_var_types, Y0, Y1


def generate_data(N, x_shift=0, y_shift=0, b_shift=0, d=1):
    DAGnx = nx.DiGraph()

    Ux = np.random.randn(N, d)
    X = x_shift + Ux

    Ub = np.random.randn(N, d)
    B = b_shift + (Ub)

    Uy = np.random.randn(N, d)
    Y = y_shift + 0.8 * X + 0.3 * B + 0.01 * Uy

    Y0 = y_shift + 0.8 * 0 + 0.3 * B + 0.01 * Uy
    Y1 = y_shift + 0.8 * 1 + 0.3 * B + 0.01 * Uy

    all_data_dict = {'X': X, 'B': B, 'Y': Y}

    # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
    var_types = {'X': 'cont', 'B': 'cont', 'Y': 'cont'}

    DAGnx.add_edges_from([('X', 'Y'), ('B', 'Y',)])
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    var_names = list(DAGnx.nodes())  # topologically ordered list of variables
    all_data = np.stack([all_data_dict[key] for key in var_names], axis=1)
    causal_ordering = get_full_ordering(DAGnx)
    ordered_var_types = dict(sorted(var_types.items(), key=lambda item: causal_ordering[item[0]]))

    return all_data, DAGnx, var_names, causal_ordering, ordered_var_types, Y0, Y1


def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)


def input_output_sensitivity_matrix(input, model, epsilon=100.0, full=True):
    model.eval()
    input = input.detach().clone()

    if full:
        original_output, _, _ = model(X=input, targets=input, shuffling=False)
    else:
        original_output = model(X=input)

    sensitivity_matrix = torch.zeros((input.size(1), original_output.size(1)))

    for i in range(input.size(1)):
        perturbed_input = input.clone()
        perturbed_input[0, i] += epsilon

        if full:
            perturbed_output, _, _ = model(X=perturbed_input, targets=perturbed_input, shuffling=False)
        else:
            perturbed_output = model(X=perturbed_input)

        output_difference = perturbed_output - original_output

        sensitivity_matrix[i, :] = torch.abs(output_difference.view(-1))

    return sensitivity_matrix


_, _, _, _, _, Y0, Y1 = generate_data(N=1000000, d=d)
ATE = (Y1 - Y0).mean(0)  # multi-dim ATE based off a large sample

print('ATE:', ATE)

estimates = []
x_shifts = []
x_inc = 0.0
for i in range(1):
    x_shifts.append(x_shift)

    all_data, DAGnx, var_names, causal_ordering, var_types, Y0, Y1 = generate_data(N=sample_size, x_shift=x_shift, d=d)
    #print(all_data[:5])
    indices = np.arange(0, len(all_data))
    np.random.shuffle(indices)

    val_inds = indices[:int(validation_fraction * len(indices))]
    train_inds = indices[int(validation_fraction * len(indices)):]

    train_data = all_data[train_inds]
    val_data = all_data[val_inds]

    train_data, val_data = torch.from_numpy(train_data).float(), torch.from_numpy(val_data).float()
    input_dim = all_data.shape[2]
    input_n_var = all_data.shape[1]

    model = CaT(
        input_dim=input_dim,
        embed_dim= embed_dim,
        dropout_rate=dropout_rate,
        head_size=head_size,
        num_heads=num_heads,
        ff_n_embed=ff_n_embed,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        n_layers=n_layers,
        device=device,
        var_types=var_types, activation_function='Swish'
    ).to(device)
    #print(f"Data vs model :{all_data[:2]} \nModel:{model(torch.from_numpy(all_data[:2]).float().to(device))}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #from torchinfo import summary

    #summary(model, input_size=all_data.shape)
    #print(model)

    #for name, layer in model.named_children():
    #    print(f"Layer: {name} \nType: {layer}\n")


    def lr_lambda(epoch):
        if epoch < warmup_iters:
            return float(epoch) / float(max(1, warmup_iters))
        return 1.0


    warmup_iters = max_iters // 5  # Number of iterations for warmup
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_cyclic = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters)

    all_var_losses = {}
    inter_model_ests = []
    sub_epoch = []
    mses = []
    for iter_ in range(0, max_iters):
        # train and update the model
        model.train()

        xb = get_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
        xb_mod = torch.clone(xb.detach())
        X, loss, loss_dict = model(X=xb, targets=xb_mod, shuffling=shuffling)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_ < warmup_iters:
            scheduler_warmup.step()
        else:
            scheduler_cyclic.step()

        if iter_ % eval_interval == 0:  # evaluate the loss (no gradients)
            for key in loss_dict.keys():
                if key not in all_var_losses.keys():
                    all_var_losses[key] = []
                all_var_losses[key].append(loss_dict[key])

            model.eval()
            eval_loss = {}
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    xb = get_batch(train_data=train_data, val_data=val_data, split=split, device=device,
                                   batch_size=batch_size)
                    xb_mod = torch.clone(xb.detach())
                    X, loss, loss_dict = model(X=xb, targets=xb_mod, shuffling=False)
                    losses[k] = loss.item()
                eval_loss[split] = losses.mean()
            mses.append(eval_loss['train'])
            sub_epoch.append(iter_)
            model.train()
            print(f"step {iter_} of {max_iters}: train_loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")

    model.eval()
    intervention_nodes_vals_0 = {'X': x_shift + 0}
    intervention_nodes_vals_1 = {'X': x_shift + 1}
    ci = inference.CausalInference(model=model, device=device)
    D0 = ci.forward(data=all_data[:5], intervention_nodes_vals=intervention_nodes_vals_0)
    D1 = ci.forward(data=all_data[:5], intervention_nodes_vals=intervention_nodes_vals_1)
    #print(all_data[:5])
    #print(ci.forward(data=all_data[:5]))

    effect_var = 'Y'
    effect_index = utils.find_element_in_list(var_names, target_string=effect_var)

    est_ATE = (D1[:, effect_index] - D0[:, effect_index]).mean()
    print('ATE:', ATE, 'est ATE:', est_ATE)
    estimates.append(est_ATE)

    x_shift += x_inc
    print(x_shift)

print(estimates)

'''Notes:
activation seems to play some role, but not be the main issue. Mish, Swish, and LeakyReLU have been tested and work, but conditional on the shift issue (see below).
shifting y seems not to have an impact, but shifting x does. It impacts the loss, and the estimation of the causal effect. 
If X and B are both shifted by the same amount, the network fails. However, if X is shifted away from zero and B is not, it works.
Importantly, bypassing the attention mechanism (and feeding Vprime only as the output of the MHA head doesn't work either!).
'''
