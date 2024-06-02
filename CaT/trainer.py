import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import generate_data
import warnings
from model import CaT
import inference
warnings.filterwarnings("ignore")



def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)

def plot_losses(loss_dict):

    plt.figure(figsize=(10,8))

    for key, values_list in loss_dict.items():
        plt.plot(values_list, label=key)

    # Add the legend to the plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('losses.png')
    plt.close()


def train(train_data, val_data, max_iters, eval_interval, eval_iters, device, model, batch_size, save_iter,
          model_save_path, optimizer, start_iter=None, checkpointing_on=0, shuffling=False):

    train_data, val_data = torch.from_numpy(train_data).float(),  torch.from_numpy(val_data).float()

    if start_iter == None:
        start_iter = 0
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,  max_iters//10, eta_min=1e-7, verbose=False)
    all_var_losses = {}
    for iter_ in range(start_iter, max_iters):
        # train and update the model
        model.train()

        xb = get_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
        xb_mod = torch.clone(xb.detach())
        X, loss, loss_dict = model(X=xb, targets=xb_mod, shuffling=shuffling)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


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
            model.train()
            print(f"step {iter_} of {max_iters}: train_loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")


        if (iter_ > 1 ) and (iter_ != start_iter) and ((iter_ + 1) % save_iter == 0) and (checkpointing_on==1):
            print('Saving model checkpoint.')
            torch.save({
                'iteration': iter_,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_path, 'model_{}_{}.ckpt'.format(iter_+1, np.round(loss.item(), 2))))

    print('Finished training!')

    plot_losses(all_var_losses)




# for now set the intervention stuff globally here:
intervention_nodes_vals_0 = {'X': 0}
intervention_nodes_vals_1 = {'X': 1}


def objective(trial, args):
    shuffling = args.shuffling
    seed = args.seed
    standardize = args.standardize
    sample_size = args.sample_size
    batch_size = args.batch_size
    head_size = args.head_size
    ff_n_embed = args.ff_n_embed
    save_iter = args.save_iter
    eval_interval = args.eval_interval
    eval_iters = args.eval_iters
    validation_fraction = args.validation_fraction
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    device = args.device
    dataset = args.dataset
    fn = args.data_path
    existing_model_path = args.existing_model_path
    model_save_path = args.model_save_path
    checkpointing_on = args.checkpointing_on

    if args.run_optuna:
        # optuna parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        max_iters = trial.suggest_int('max_iters', 500, 400000)
        num_heads = trial.suggest_int('num_heads', 2, 2)
        n_layers = trial.suggest_int('n_layers', 2, 2)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

        print('learning rate', learning_rate,
                'max iters', max_iters,
                'num_heads', num_heads,
                'n_layers', n_layers,
                'dropout_rate', dropout_rate,
                'optimizer', optimizer_name)

    else:
        learning_rate = args.learning_rate
        max_iters = args.max_iters
        num_heads = args.num_heads
        n_layers = args.n_layers
        dropout_rate = args.dropout_rate
        optimizer_name = "Adam"

    _, _, _, _, _, Y0, Y1 = generate_data(N=1000000, seed=seed, dataset=dataset, standardize=0)
    ATE = (Y1 - Y0).mean(0)  # ATE based off a large sample

    all_data, DAGnx, var_names, causal_ordering, var_types, Y0, Y1 = generate_data(N=sample_size, seed=seed,
                                                                                   dataset=dataset, standardize=0)


    input_dim = all_data.shape[2]
    indices = np.arange(0, len(all_data))
    np.random.shuffle(indices)

    val_inds = indices[:int(validation_fraction*len(indices))]
    train_inds = indices[int(validation_fraction*len(indices)):]
    train_data = all_data[train_inds]
    val_data = all_data[val_inds]

    print('Training data size:', train_data.shape, ' Validation data size:', val_data.shape)


    model = CaT(input_dim=input_dim,
                dropout_rate=dropout_rate,
                head_size=head_size,
                num_heads=num_heads,
                ff_n_embed=ff_n_embed,
                dag=DAGnx,
                causal_ordering=causal_ordering,
                n_layers=n_layers,
                device=device,
                var_types_sorted=var_types,
                ).to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    if existing_model_path == 'None':  # if no specified checkpoint file is given, train the model
        print('No existing model path specified, training from scratch for {} iterations.'.format(max_iters))
        train(train_data=train_data,
                      eval_iters=eval_iters,
                      val_data=val_data,
                      max_iters=max_iters,
                      eval_interval=eval_interval,
                      device=device,
                      model=model,
                      batch_size=batch_size,
                      save_iter=save_iter,
                      model_save_path=model_save_path,
                      optimizer=optimizer,
                      checkpointing_on=checkpointing_on,
                      shuffling=shuffling
                      )

    else:  # if existing checkpoint file is given, compare iteration against max_iters and finish training if necessary
        print('Loading checkpoint file at: ', existing_model_path)
        checkpoint = torch.load(existing_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_iter = checkpoint['iteration']

        if checkpoint_iter != max_iters:
            train(train_data=train_data,
                      eval_iters=eval_iters,
                      val_data=val_data,
                      max_iters=max_iters,
                      eval_interval=eval_interval,
                      device=device,
                      model=model,
                      batch_size=batch_size,
                      save_iter=save_iter,
                      model_save_path=model_save_path,
                      optimizer=optimizer,
                      checkpointing_on=checkpointing_on,
                      shuffling=shuffling
                          )


    ######### check ATE (temporary code, to be generalised and integrated into testbed.py)########################

    model.eval()
    inf = inference.CausalInference(model=model, device=device)

    int_nodes_vals0 = {'X': np.array([0.0, ])}
    int_nodes_vals1 = {'X': np.array([1.0, ])}
    effect_var = 'Y'
    effect_index = var_names.index(effect_var)

    preds0 = inf.forward(all_data, int_nodes_vals0)
    preds1 = inf.forward(all_data, int_nodes_vals1)
    ATE_pred = (preds1[:, effect_index, :] - preds0[:, effect_index, :]).mean(0)
    eATE = np.abs(ATE_pred - ATE)
    print('ATE:', ATE, 'est ATE:', ATE_pred, 'error:', eATE)


    # view attention maps
    maps = []
    for j in range(n_layers):
        heads = model.blocks[j].mha.heads
        for i in range(args.num_heads):
            maps.append(heads[i].att_wei.mean(0).cpu().detach().numpy())

    maps = np.stack(maps).mean(0)
    fig, ax = plt.subplots()
    im = ax.imshow(maps, cmap='hot', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=1)
    # Setting the axis tick labels
    ax.set_xticks(np.arange(len(list(DAGnx.nodes))))
    ax.set_yticks(np.arange(len(list(DAGnx.nodes))))

    ax.set_xticklabels(list(DAGnx.nodes))
    ax.set_yticklabels(list(DAGnx.nodes))

    # Rotating the tick labels inorder to set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig('attention_maps.png')
    plt.close()