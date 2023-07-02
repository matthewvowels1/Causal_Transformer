import tiktoken
import torch
from torch.nn import functional as F
import torch.nn as nn

# adapted from example GPT code  https://github.com/karpathy/ng-video-lecture
class Head(nn.Module):
	def __init__(self, head_size, n_embed, block_size, dropout_rate, dag):
		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias=False)
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		# user a register buffer (not a module parameter) for the creation of self.dag
		# dag will determine what variables can communicate with each other
		self.register_buffer('dag', dag)
		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, X):
		B, T, C = X.shape  # batch size, time steps, channels
		K = self.key(X)  # B, T, hs
		Q = self.query(X)  # B, T, hs

		wei = Q @ K.transpose(-2, -1) * K.shape[-1] ** -0.5  # (B,T,hs) @ (B,hs,T) -> (B,T,T)
		wei = wei.masked_fill(self.dag[:T, :T] == 0, float('-inf'))  # (B, T, T)
		wei = F.softmax(wei, dim=-1)  # B,T,T
		wei = self.dropout(wei)
		V = self.value(X)  # B, T, hs
		out = wei @ V  # (B, T, T) @ (B, T, hs) ->  (B, T, hs)
		return out

class MultiHeadAttention(nn.Module):

	def __init__(self, num_heads, head_size, n_embed, block_size, dropout_rate, dag):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout_rate, dag=dag) for _ in range(num_heads)])
		self.projection = nn.Linear(n_embed, n_embed)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, X):
		out = torch.cat([h(X) for h in self.heads], dim=-1)
		return self.dropout(self.projection(out))


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

	def __init__(self, n_embed, num_heads, head_size, block_size, dropout_rate, dag):
		super().__init__()
		self.sa = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout_rate, dag)
		self.ff = FF(n_embed, dropout_rate)
		self.layernorm1 = nn.LayerNorm(n_embed)
		self.layernorm2 = nn.LayerNorm(n_embed)

	def forward(self, X):
		X = X + self.sa(self.layernorm1(X))  # with residual skip connection and learnable normalization of features
		X = X + self.ff(self.layernorm1(X))  # with residual skip connection and learnable normalization of features
		return X


class CaT(nn.Module):

	def __init__(self, num_vars, n_embed, dropout_rate, num_heads, n_layers, dag, device):
		super().__init__()
		head_size = n_embed // num_heads
		self.dag = torch.tensor(dag)
		self.device = device
		self.token_embedding_table = nn.Embedding(num_vars, n_embed)
		self.position_embedding_table = nn.Embedding(num_vars, n_embed)
		self.blocks = nn.Sequential(
			*[Block(n_embed, num_heads, head_size, num_vars, dropout_rate, self.dag) for _ in range(n_layers)])
		self.layernorm_f = nn.LayerNorm(n_embed)
		self.lm_head = nn.Linear(n_embed, num_vars)  # goes from embedding to vocab size

	def forward(self, idx, targets=None):
		B, T = idx.shape
		# idx are the indices of the inputs and targets are the indices of the things to predict
		tok_emb = self.token_embedding_table(idx)  # (B, T, n_embed)
		pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, n_embed )
		X = tok_emb + pos_emb  # (B, T, n_embed)
		X = self.blocks(X)  # B, T, head_size
		X = self.layernorm_f(X)
		logits = self.lm_head(X)  # (B, T, vocab_size)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B * T, C)
			targets = targets.view(B * T)  # targets treated as the correct class indices
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop idx to the last block_size tokens
			idx_cond = idx[:, -block_size:]
			# get the predictions
			logits, loss = self(idx_cond)
			# focus only on the last time step
			logits = logits[:, -1, :]  # becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1)  # (B, C)
			# sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
			# append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
		return idx
