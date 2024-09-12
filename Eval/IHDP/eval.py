import networkx as nx
import numpy
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from CaT.datasets import reorder_dag, get_full_ordering
from CaT.test_model import CaT
from loader import load_IHDP
import torch

seed = 1
np.random.seed(seed=seed)
torch.manual_seed(seed)
device = 'cuda'


def instantiate_model():
    dropout_rate = 0.0
    ff_n_embed = 4
    num_heads = 3
    n_layers = 1
    embed_dim = 5
    head_size = 4
    input_dim = 1

    DAGnx = nx.DiGraph()


    # types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
    var_types = {
        'treatment': 'bin',
        'y': 'cont',
        **{str(i): 'bin' for i in range(7, 26)},
        **{str(i): 'cont' for i in range(1, 7)}
    }

    DAGnx.add_edges_from(
        [('treatment', 'y')] + [(str(i), 'treatment') for i in range(1, 26)] + [(str(i), 'y') for i in range(1, 26)])
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


def train(model, train, test):
    shuffling = 0
    max_iters = 5000
    eval_interval = 100
    eval_iters = 10
    learning_rate = 2e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def lr_lambda(epoch):
        if epoch < warmup_iters:
            return float(epoch) / float(max(1, warmup_iters))
        return 1.0

    warmup_iters = max_iters // 5  # Number of iterations for warmup
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_cyclic = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters)

    all_var_losses = {}
    sub_epoch = []
    mses = []
    for iter_ in range(0, max_iters):
        # train and update the model
        model.train()
        xb = torch.clone(train)
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
                    xb = torch.clone(train if split == 'train' else test)
                    xb_mod = torch.clone(xb.detach())
                    X, loss, loss_dict = model(X=xb, targets=xb_mod, shuffling=False)
                    losses[k] = loss.item()
                eval_loss[split] = losses.mean()
            mses.append(eval_loss['train'])
            sub_epoch.append(iter_)
            model.train()
            print(
                f"step {iter_} of {max_iters}: train_loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")
    model.eval()


for i in range(1, 3):
    model = instantiate_model()
    dataset = {}
    true_ite = {}
    for split in ('train', 'test'):
        z, x, y, y1, y0 = load_IHDP(path="data/", replication=i, split=split)
        data = numpy.concatenate([z, x, y], axis=1)
        data = data.reshape([-1, 27, 1])
        data = torch.from_numpy(data).to(device=device).float()
        dataset[split] = data
        true_ite[split] = y1 - y0

    train(model=model, train=dataset['train'], test=dataset['test'])

    for split in ('train', 'test'):


        data0 = dataset[split].clone()
        data0[:, -2, :] = 0
        data1 = dataset[split].clone()
        data1[:, -2, :] = 1

        output0 = model(data0)[:, -1, :].detach().cpu().numpy().reshape([-1])
        output1 = model(data1)[:, -1, :].detach().cpu().numpy().reshape([-1])

        est_ite = output1 - output0
        est_ate = est_ite.mean()

        pehe = np.sqrt(
            np.mean((true_ite[split].squeeze() - est_ite) * (true_ite[split].squeeze() - est_ite)))
        eate = np.abs(true_ite[split].mean() - est_ate)

        with open("output.txt", "a") as file:
            file.write(f"i: {i}\nsplit: {split}\npehe: {pehe}\neate: {eate}\n")
