import numpy as np
from numpy.typing import NDArray
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from CaT.datasets import reorder_dag, get_full_ordering
from CaT.model import CaT

from CFCN.test_model import CFCN
from CFCN.model import CFCN as OldCFCN


def instantiate_new_CFCN(device,
                         var_types,
                         DAGnx,
                         neurons_per_layer=None,
                         dropout_rate=0.5, ):
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    print(causal_ordering)
    if neurons_per_layer is None:
        x = len(var_types)
        neurons_per_layer = [x, 2 * x, x]
    model = CFCN(
        neurons_per_layer=neurons_per_layer,
        dropout_rate=dropout_rate,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        device=device,
        var_types=var_types,
    ).to(device)
    return model


def instantiate_old_CFCN(device,
                         var_types,
                         DAGnx,
                         neurons_per_layer=None,
                         dropout_rate=0.5, ):
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    print(causal_ordering)
    if neurons_per_layer is None:
        x = len(var_types)
        neurons_per_layer = [x, 2 * x, x]
    model = OldCFCN(
        neurons_per_layer=neurons_per_layer,
        dropout_rate=dropout_rate,
        dag=DAGnx,
        causal_ordering=causal_ordering,
        device=device,
        var_types=var_types,
    ).to(device)
    return model


def instantiate_CaT(device,
                    var_types,
                    DAGnx,
                    dropout_rate=0.0,
                    ff_n_embed=4,
                    num_heads=3,
                    n_layers=1,
                    embed_dim=5,
                    head_size=4,
                    input_dim=1):
    DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
    causal_ordering = get_full_ordering(DAGnx)
    print(causal_ordering)
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


def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)


def train_model(model, train_data, val_data, device, shuffling=0, max_iters=10000, eval_interval=500, eval_iters=20,
                learning_rate=2e-4, batch_size=32, use_scheduler=True):
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
        return data.to(device).float()

    train_data = to_torch_device(train_data)
    val_data = to_torch_device(val_data)

    all_var_losses = {}
    sub_epoch = []
    mses = []
    for iter_ in range(0, max_iters):
        # train and update the model
        model.train()

        # Get a batch of data
        xb = get_batch(train_data, val_data, 'train', device, batch_size)
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
                    # Get a batch of data for evaluation
                    xb = get_batch(train_data, val_data, split, device, batch_size)
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


def compute_eate(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
                 y1: NDArray[np.float_], y0: NDArray[np.float_]) -> dict:
    # ypred, y and t should be RCT

    true_ate = safe_mean(y1 - y0)

    estimated_ate = safe_mean(ypred1 - ypred0)

    return {'eate': abs(true_ate - estimated_ate), 'estimated ate': estimated_ate}
