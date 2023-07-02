import torch
import os
import numpy as np

def get_batch(train_data, val_data, split, device, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb = get_batch(train_data, val_data, split, device, batch_size)
            logits, loss = model(X=xb, targets=xb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(train_data, val_data, max_iters, eval_interval, eval_iters, device, model, batch_size, save_iter, model_save_path, optimizer, start_iter=None):
    train_data, val_data = torch.from_numpy(train_data).float(),  torch.from_numpy(val_data).float()

    if start_iter == None:
        start_iter = 0

    for iter_ in range(start_iter, max_iters):
        xb = get_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
        if iter_ % eval_interval == 0:
            losses = estimate_loss(model=model,
                          train_data=train_data,
                          val_data=val_data,
                          eval_iters=eval_iters,
                          batch_size=batch_size,
                          device=device)
            print(f"step {iter_}: train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


        X, loss = model(X=xb, targets=xb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (iter_ > 1 ) and (iter_ != start_iter) and ((iter_ + 1) % save_iter == 0):
            torch.save({
                'iteration': iter_,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_path, 'model_{}_{}.ckpt'.format(iter_+1, np.round(loss.item(), 2))))

    return loss