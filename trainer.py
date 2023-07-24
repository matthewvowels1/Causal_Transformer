import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

def predict(model, data, device):
    data = torch.from_numpy(data).float().to(device)
    return model.forward(data)


def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)


def train(train_data, val_data, max_iters, eval_interval, eval_iters, device, model, batch_size, save_iter, model_save_path, optimizer, start_iter=None):
    train_data, val_data = torch.from_numpy(train_data).float(),  torch.from_numpy(val_data).float()

    if start_iter == None:
        start_iter = 0

    for iter_ in range(start_iter, max_iters):
        # train and update the model
        model.train()
        xb = get_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
        xb_mod = torch.clone(xb.detach())

        X, loss = model(X=xb, targets=xb_mod)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_ % eval_interval == 0:  # evaluate the loss (no gradients)
            model.eval()
            eval_loss = {}
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):

                    xb = get_batch(train_data=train_data, val_data=val_data, split=split, device=device,
                                   batch_size=batch_size)
                    X, loss = model(X=xb, targets=xb_mod)
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