

import torch
def predict(model, data, device):
	data = torch.from_numpy(data).float().to(device)
	return model.forward(data)

def eval(model, data, int_column, device):

	d0 = data.copy()
	d1 = data.copy()
	d0[:, int_column] = 0.0
	d1[:, int_column] = 1.0

	preds_d0 = predict(model, d0, device)
	preds_d1 = predict(model, d1, device)

	est_ATE = (preds_d1[:-2] - preds_d0[:-2]).mean()

	return est_ATE