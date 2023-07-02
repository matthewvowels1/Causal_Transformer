import tiktoken
import torch
from torch.nn import functional as F
import torch.nn as nn

# adapted from example GPT code  https://github.com/karpathy/ng-video-lecture
class Head(nn.Module):
	def __init__(self, head_size, dropout_rate, dag):
		super().__init__()
		self.key = nn.Linear(1, head_size, bias=False)
		self.query = nn.Linear(1, head_size, bias=False)
		self.value = nn.Linear(1, head_size, bias=False)
		# user a register buffer (not a module parameter) for the creation of self.dag
		# dag will determine what variables can communicate with each other
		self.register_buffer('dag', dag)
		self.dropout = nn.Dropout(dropout_rate)
	def forward(self, X):
		B, T, C = X.shape  # batch size, time steps, channels
		K = self.key(X)  # B, T, hs
		Q = self.query(X)  # B, T, hs

		wei = Q @ K.transpose(-2, -1) * K.shape[-1] ** -0.5  # (B,T,hs) @ (B,hs,T) -> (B,T,T)
		wei = wei.masked_fill(self.dag == 0, float('-inf'))  # (B, T, T)

		inf_rows = torch.all(wei == float('-inf'), dim=-1)  # check if any rows are <all> -inf, these need to be masked to 0
		all_inf_mask = inf_rows.unsqueeze(-1).expand_as(wei)
		wei[all_inf_mask] = 0.0  # set any rows which are all -inf (because they have no causal parents) to 0 to avoid nans

		wei = F.softmax(wei, dim=-1)  # B,T,T
		wei = self.dropout(wei)
		V = self.value(X)  # B, T, hs
		out = wei @ V  # (B, T, T) @ (B, T, hs) ->  (B, T, hs)
		return out

class MultiHeadAttention(nn.Module):

	def __init__(self, num_heads, head_size, dropout_rate, dag):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size=head_size, dropout_rate=dropout_rate, dag=dag) for _ in range(num_heads)])
		self.projection = nn.Linear(int(head_size*num_heads), 1)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, X):
		out = torch.cat([h(X) for h in self.heads], dim=-1)
		out = self.dropout(self.projection(out))
		return out


class FF(nn.Module):
	def __init__(self, n_embed, dropout_rate):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed, 4 * n_embed),
			nn.ReLU(),
			nn.Linear(4 * n_embed, n_embed),
			nn.Dropout(dropout_rate),
		)

	def forward(self, X):
		return self.net(X)


class Block(nn.Module):

	def __init__(self, n_embed, num_heads, head_size,  dropout_rate, dag):
		super().__init__()
		self.sa = MultiHeadAttention( num_heads, head_size, dropout_rate, dag)

		self.ff = FF(n_embed, dropout_rate)

	def forward(self, X):
		X = X + self.sa(X)  # with residual skip connection and learnable normalization of features
		X = X + self.ff(X)  # with residual skip connection and learnable normalization of features
		return X


class CaT(nn.Module):

	def __init__(self, num_vars, dropout_rate, num_heads, head_size, n_layers, dag, device):
		super().__init__()
		self.dag = torch.tensor(dag).to(device)
		self.device = device
		self.blocks = nn.Sequential(
			*[Block(n_embed=1, num_heads=num_heads, head_size=head_size, dropout_rate=dropout_rate, dag=self.dag) for _ in range(n_layers)])
		self.lm_head = nn.Linear(1, num_vars)  # goes from embedding to vocab size
		self.mse_loss = torch.nn.MSELoss()

	def forward(self, X, targets=None):
		X = X[:, :, None]  # (B, num_vars, C=1)
		X = self.blocks(X)  # B, num_vars, head_size
		X = self.lm_head(X)  # (B, T, vocab_size
		X = X[:, :, 0]
		if targets == None:
			return X
		else:
			loss = self.mse_loss(X, targets)
			return X, loss


