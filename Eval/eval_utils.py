import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from CaT.datasets import reorder_dag, get_full_ordering
from CaT.test_model import CaT

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

def train_model(model, train, test, device, shuffling=0, max_iters=5000, eval_interval=100, eval_iters=10,
                learning_rate=2e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

    train = to_torch_device(train)
    test = to_torch_device(test)

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
