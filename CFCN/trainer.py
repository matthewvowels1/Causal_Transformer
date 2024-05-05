import networkx as nx
import pandas as pd
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import generate_data
import warnings
from model import DAGAutoencoder
import utils
warnings.filterwarnings("ignore")
from inference import CausalInference, CausalMetrics


def assert_neuron_layers(layers, input_size):
    # Assert that the smallest number of neurons is never lower than input_size
    assert min(layers) >= input_size, "The smallest layer size must be at least the input size."

    # Assert that subsequent layers change either not at all, or by a factor of 2
    for i in range(1, len(layers)):
        previous_layer, current_layer = layers[i-1], layers[i]
        is_same = current_layer == previous_layer
        is_double = current_layer == 2 * previous_layer
        is_half = current_layer == previous_layer / 2
        assert is_same or is_double or is_half, "Layer sizes must stay the same or change by a factor of 2."

    # Assert that the first layer is a multiple of 2 of the input_size
    assert layers[0] % input_size == 0 and ((layers[0] // input_size) & ((layers[0] // input_size) - 1)) == 0, \
        "The first layer must be a multiple of 2 of the input size."




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


def objective(trial, args):
    shuffling = args.shuffling
    seed = args.seed
    standardize = args.standardize
    sample_size = args.sample_size
    batch_size = args.batch_size
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
        max_iters = trial.suggest_int('max_iters', 500, 40000)
        num_heads = trial.suggest_int('num_heads', 2, 2)
        n_layers = trial.suggest_int('n_layers', 2, 2)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        interaction_type = trial.suggest_int('interaction_type', 0, 2)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

        print('learning rate', learning_rate,
                'max iters', max_iters,
                'num_heads', num_heads,
                'n_layers', n_layers,
                'dropout_rate', dropout_rate,
                'interaction_type', interaction_type,
                'optimizer', optimizer_name)

    else:
        learning_rate = args.learning_rate
        max_iters = args.max_iters
        neurons_per_layer = args.neurons_per_layer
        dropout_rate = args.dropout_rate
        optimizer_name = "Adam"

    _, _, _, Y0, Y1 = generate_data(N=1000000, seed=seed, dataset=dataset, standardize=standardize)
    ATE = (Y1 - Y0).mean()  # ATE based off a large sample

    all_data, DAG, var_types, Y0, Y1 = generate_data(N=sample_size, seed=seed, dataset=dataset)

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(fn, 'all_{}.csv'.format(dataset)), index=False)

    input_dim = all_data.shape[1]
    print(input_dim)
    # prepend the input size to neurons_per_layer
    # append the intput size to neurons_per_layer (output)
    neurons_per_layer.insert(0, input_dim)
    neurons_per_layer.append(input_dim)
    assert_neuron_layers(layers=neurons_per_layer, input_size=input_dim)

    indices = np.arange(0, len(all_data))
    np.random.shuffle(indices)

    val_inds = indices[:int(validation_fraction*len(indices))]
    train_inds = indices[int(validation_fraction*len(indices)):]
    train_data = all_data[train_inds]
    val_data = all_data[val_inds]

    print('Training data size:', train_data.shape, ' Validation data size:', val_data.shape)

    initial_adj_matrix = nx.to_numpy_array(DAG)

    initial_masks = [torch.from_numpy(mask).float().to(torch.float64) for mask in
                     utils.expand_adjacency_matrix(neurons_per_layer[1:], initial_adj_matrix)]


    model = DAGAutoencoder(neurons_per_layer=neurons_per_layer, dag=DAG, var_types=var_types, dropout_rate=dropout_rate).to(device)
    model.initialize_masks(initial_masks)

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
            train(rain_data=train_data,
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

    causal_mod =
    predictions_x0, predictions_x1 = CausalInference.