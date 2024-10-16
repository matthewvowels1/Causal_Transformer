import numpy as np
import torch
import torch.nn as nn
import sklearn as sk
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from CaT.datasets import reorder_dag, get_full_ordering
from CaT.model import CaT

from CFCN.test_model import CFCN
from CFCN.model import CFCN as OldCFCN
from baseline.transformer import TransformerRegressor
from utils.inference import CausalInference


def instantiate_new_CFCN(var_types,
                         DAGnx,
                         neurons_per_layer=None,
                         dropout_rate=0.0, device='cuda'):
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    if neurons_per_layer is None:
        x = len(var_types)
        neurons_per_layer = [x, 2 * x, 2 * x, 2 * x, 2 * x, x]
    model = CFCN(
        neurons_per_layer=neurons_per_layer,
        dropout_rate=dropout_rate,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        device=device,
        var_types=var_types,
    ).to(device)
    return model


def instantiate_old_CFCN(
        var_types,
        DAGnx,
        neurons_per_layer=None,
        dropout_rate=0.0, device='cuda'):
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    if neurons_per_layer is None:
        x = len(var_types)
        neurons_per_layer = [x, 2 * x, 2 * x, 2 * x, 2 * x, x]
    model = OldCFCN(
        neurons_per_layer=neurons_per_layer,
        dropout_rate=dropout_rate,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        device=device,
        var_types=var_types,
    ).to(device)
    return model


def instantiate_CaT(
        var_types,
        DAGnx,
        dropout_rate=0.0,
        ff_n_embed=6,
        num_heads=2,
        n_layers=2,
        embed_dim=5,
        head_size=6,
        input_dim=1, device='cuda'):
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    model = CaT(
        input_dim=input_dim,
        embed_dim=embed_dim,
        dropout_rate=dropout_rate,
        head_size=head_size,
        num_heads=num_heads,
        ff_n_embed=ff_n_embed,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        n_layers=n_layers,
        device=device,
        var_types=var_types,
        activation_function='Swish'
    ).to(device)
    return model


def train_model(model, train, test, ci):
    if isinstance(model, nn.Module):
        hyperparam = {
            'learning_rate': 1e-3,
            'max_iters': 1#20000
        } if isinstance(model, (CFCN, OldCFCN)) else {
            'learning_rate': 5e-3,
            'max_iters': 1#6000
        }
        train_module(model=model, train_data=train, val_data=test, **hyperparam)
    elif isinstance(model, sk.base.BaseEstimator):
        model.fit(X=ci.remove(train, 'y'), y=ci.get(train, 'y'))
    else:
        raise ValueError(f"unexpected type {type(model)}")


def predict(model, data, ci: CausalInference, interventions_nodes, output_var='y'):
    if isinstance(model, nn.Module):
        if isinstance(model, TransformerRegressor):
            data = ci.apply_intervention(data=data, intervention_nodes_vals=interventions_nodes)
            data = ci.remove(data=data, var_name=output_var)
            return model.forward(data)
        else:
            prediction = ci.forward(data=data, model=model, intervention_nodes_vals=interventions_nodes)
            return ci.get(data=prediction, var_name=output_var)
    elif isinstance(model, sk.base.BaseEstimator):
        return model.predict(ci.remove(data=data, var_name=output_var))
    else:
        raise ValueError(f"unexpected type {type(model)}")


def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)


def train_module(model, train_data, val_data, shuffling=0, max_iters=6000, eval_interval=0, eval_iters=100,
                 learning_rate=5e-3, batch_size=100, use_scheduler=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if use_scheduler:
        def lr_lambda(epoch):
            if epoch < warmup_iters:
                return float(epoch) / float(max(1, warmup_iters))
            return 1.0

        warmup_iters = max_iters // 5  # Number of iterations for warmup
        scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_cyclic = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters)

    def to_torch_device(data):
        if not isinstance(data, torch.Tensor):
            # If not a tensor, convert from numpy to tensor
            data = torch.from_numpy(data)
        return data.to(model.device).float()

    train_data = to_torch_device(train_data)
    val_data = to_torch_device(val_data)

    all_var_losses = {}
    sub_epoch = []
    mses = []
    for iter_ in range(0, max_iters):
        # train and update the model
        model.train()

        # Get a batch of data
        xb = get_batch(train_data, val_data, 'train', model.device, batch_size)
        xb_mod = torch.clone(xb.detach())  # Create target data

        X, loss, loss_dict = model(X=xb, targets=xb_mod, shuffling=shuffling)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if use_scheduler:
            if iter_ < warmup_iters:
                scheduler_warmup.step()
            else:
                scheduler_cyclic.step()

        if eval_interval and iter_ % eval_interval == 0:  # evaluate the loss (no gradients)
            for key in loss_dict.keys():
                if key not in all_var_losses.keys():
                    all_var_losses[key] = []
                all_var_losses[key].append(loss_dict[key])

            model.eval()
            eval_loss = {}
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    # Get a batch of data for evaluation
                    xb = get_batch(train_data, val_data, split, model.device, batch_size)
                    xb_mod = torch.clone(xb.detach())  # Create target data

                    X, loss, loss_dict = model(X=xb, targets=xb_mod, shuffling=False)
                    losses[k] = loss.item()
                eval_loss[split] = losses.mean()
            mses.append(eval_loss['train'])
            sub_epoch.append(iter_)
            model.train()
            print(
                f"step {iter_} of {max_iters}: train_loss {eval_loss['train']:.4f}, val_loss {eval_loss['val']:.4f}")
    model.eval()


def compute_result(input_path='output.txt', output_path='result.txt'):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    datas = {}

    def parse(split, sort, value):
        if sort == "pehe":
            sort = "square root pehe"
            value = np.sqrt(value)
        index = f"{split}: {sort}"
        if index not in datas:
            datas[index] = []
        datas[index].append(value)

    for i in range(0, len(lines), 4):
        split_type = lines[i + 1].split(": ")[1].strip()
        for j in range(2, 4):
            splits = lines[i + j].split(": ")
            parse(split=split_type,
                  sort=splits[0].strip(),
                  value=float(splits[1]))

    with open(output_path, "w") as file:
        for index, values in datas.items():
            file.write(f"{index}: {np.mean(values):.3f} +- {np.std(values):.3f}  n={len(values)}\n")
