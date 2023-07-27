import torch
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import generate_data
import warnings
from inference import CausalInference, find_element_in_list, predict
from model import CaT
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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


def train(train_data, val_data, max_iters, eval_interval, eval_iters, device, model, batch_size, save_iter, model_save_path, optimizer, start_iter=None):
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
        X, loss, loss_dict = model(X=xb, targets=xb_mod)

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
                    X, loss, loss_dict = model(X=xb, targets=xb_mod)
                    losses[k] = loss.item()
                eval_loss[split] = losses.mean()
            model.train()
            print(f"step {iter_}: train_loss {eval_loss['train']:.4f}, val loss {eval_loss['val']:.4f}")


        if (iter_ > 1 ) and (iter_ != start_iter) and ((iter_ + 1) % save_iter == 0):
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
    seed = args.seed
    standardize = args.standardize
    sample_size = args.sample_size
    batch_size = args.batch_size
    head_size = args.head_size
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

    # learning_rate = args.learning_rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    # max_iters = args.max_iters
    max_iters = trial.suggest_int('max_iters', 1000, 20000)
    # num_heads = args.num_heads
    num_heads  = trial.suggest_int('num_heads', 2, 10)
    # n_layers = args.n_layers
    n_layers = trial.suggest_int('n_layers', 2, 6)
    # dropout_rate = args.dropout_rate
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])


    _, _, _, _, Y0, Y1 = generate_data(N=1000000, seed=seed, dataset=dataset, standardize=standardize)
    ATE = (Y1 - Y0).mean()  # ATE based off a large sample

    all_data, DAG, causal_ordering, var_types, Y0, Y1 = generate_data(N=sample_size, seed=seed, dataset=dataset)

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(fn, 'all_{}.csv'.format(dataset)), index=False)

    indices = np.arange(0, len(all_data))
    np.random.shuffle(indices)

    val_inds = indices[:int(validation_fraction*len(indices))]
    train_inds = indices[int(validation_fraction*len(indices)):]
    train_data = all_data[train_inds]
    val_data = all_data[val_inds]

    print('Training data size:', train_data.shape, ' Validation data size:', val_data.shape)

    model = CaT(dropout_rate=dropout_rate,
                head_size=head_size,
                num_heads=num_heads,
                dag=DAG,
                ordering=causal_ordering,
                n_layers=n_layers,
                device=device,
                var_types=var_types
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
                      optimizer=optimizer
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
                          optimizer=optimizer, start_iter=checkpoint_iter
                          )


    ######### check ATE (temporary code, to be generalised and integrated into testbed.py)########################

    ci_module = CausalInference(model, device=device)
    D = ci_module.forward(data=all_data, intervention_nodes_vals=None)
    D0 = ci_module.forward(data=all_data, intervention_nodes_vals=intervention_nodes_vals_0)
    D1 = ci_module.forward(data=all_data, intervention_nodes_vals=intervention_nodes_vals_1)
    outcome_of_interest = 'Y'
    outcome_index = find_element_in_list(list(DAG.nodes()), outcome_of_interest)
    est_ATE = (D1[:, outcome_index] - D0[:, outcome_index]).mean()

    print('ATE results', est_ATE, ATE, abs(ATE-est_ATE))
    r2x = r2_score(all_data[:, 0], D[:, 0].detach().cpu().numpy())
    r2x1 = r2_score(all_data[:, 1], D[:, 1].detach().cpu().numpy())
    r2y = r2_score(all_data[:, 2], D[:, 2].detach().cpu().numpy())
    print('R2X:', r2x)
    print('R2X1:', r2x1)
    print('R2Y:', r2y)


    rdf = RandomForestRegressor()
    rdf.fit(train_data[:, :2], train_data[:, 2])
    preds = rdf.predict(val_data)

    print('sanity check R2 with RDF:')
    r2y_rdf = r2_score(val_data[:, 2], preds)
    print('R2Y:', r2y_rdf)

    return r2y

    ############################################################################################################

    # # view attention maps
    # maps = []
    # for j in range(n_layers):
    # 	heads = model.blocks[j].mha.heads
    # 	for i in range(args.num_heads):
    # 		maps.append(heads[i].att_wei.mean(0).cpu().detach().numpy())
    #
    # maps = np.stack(maps).mean(0)
    # fig, ax = plt.subplots()
    # im = ax.imshow(maps, cmap='hot', interpolation='nearest')
    # cbar = ax.figure.colorbar(im, ax=ax, shrink=1)
    # # Setting the axis tick labels
    # ax.set_xticks(np.arange(len(list(DAG.nodes))))
    # ax.set_yticks(np.arange(len(list(DAG.nodes))))
    #
    # ax.set_xticklabels(list(DAG.nodes))
    # ax.set_yticklabels(list(DAG.nodes))
    #
    # # Rotating the tick labels inorder to set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # fig.tight_layout()
    # plt.savefig('attention_maps.png')
    # plt.close()