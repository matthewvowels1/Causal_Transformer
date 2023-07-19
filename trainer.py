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

def risk_eval(model, data,  device):
    bas = None
    data_mod = data.copy()
    # data_mod = data_mod[:, :-1]

    preds = predict(model, data_mod, device).cpu().detach().numpy()
    risk = ((preds - data[:, -1])**2).mean()
    bas = balanced_accuracy_score(data[:, -1], np.round(preds))
    return risk, bas


def intervention_eval(model, data, int_column, device):

    d0 = data.copy()
    d1 = data.copy()
    d0[:, int_column] = 0.0
    d1[:, int_column] = 1.0

    preds_d0 = (predict(model, d0, device)).cpu().detach().numpy()
    preds_d1 = (predict(model, d1, device)).cpu().detach().numpy()

    est_ATE = (preds_d1 - preds_d0).mean()

    return est_ATE


def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, eval_iters, device):
    out = {}
    bas_ = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        basses = []
        for k in range(eval_iters):
            xb = get_batch(train_data, val_data, split, device, batch_size)
            xb_mod = torch.clone(xb.detach())
            # xb_mod = xb_mod[:, : -1]
            preds, loss = model(X=xb_mod, targets=xb)
            losses[k] = loss.item()

            preds = preds.cpu().detach().numpy()
            bas = balanced_accuracy_score(xb[:, -1].cpu().detach().numpy(), np.round(preds))
            basses.append(bas)

        out[split] = losses.mean()

        bas_[split] = np.stack(basses).mean()

    model.train()
    return out, bas_


def train(train_data, val_data, max_iters, eval_interval, eval_iters, device, model, batch_size, save_iter, model_save_path, optimizer, start_iter=None):
    train_data, val_data = torch.from_numpy(train_data).float(),  torch.from_numpy(val_data).float()

    if start_iter == None:
        start_iter = 0

    for iter_ in range(start_iter, max_iters):
        xb = get_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
        xb_mod = torch.clone(xb.detach())
        # xb_mod = xb_mod[:, :-1]   # exclude last variable (which is the one we wish to predict)

        if iter_ % eval_interval == 0:
            losses, bas = estimate_loss(model=model,
                          train_data=train_data,
                          val_data=val_data,
                          eval_iters=eval_iters,
                          batch_size=batch_size,
                          device=device)

            print(f"step {iter_}: train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            print(f"BAS train {bas['train']:.4f}, BAS val {bas['val']:.4f}")


        X, loss = model(X=xb_mod, targets=xb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (iter_ > 1 ) and (iter_ != start_iter) and ((iter_ + 1) % save_iter == 0):
            print('Saving model checkpoint.')
            torch.save({
                'iteration': iter_,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_path, 'model_{}_{}.ckpt'.format(iter_+1, np.round(loss.item(), 2))))

    xb = get_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
    xb_mod = torch.clone(xb.detach())
    # xb_mod = xb_mod[:, :-1]  # exclude last variable (which is the one we wish to predict)
    X, loss = model(X=xb_mod, targets=xb)
    return loss