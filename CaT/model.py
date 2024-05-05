import torch
from torch.nn import functional as F
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Optional, Dict, List, Tuple, Union


# 1. introduce dag into attention
# 2. introduce diagonal after first layer
# 3. handle skip connection masking issue

# note that the network has MHA with blocks in parallel, but this is also done sequentially, combining
# both network width and network depth.

# for the causal transformer, we have to be careful that we include a 'diagonal' pass-thru after the first layer
# otherwise, and e.g. in a three variable chain A->B->C, the dependency structure will prevent B from being predicted
# from A <after the first layer>, because B is caused by A, not by itself. So the diagonal of ones should be
# introduced after the first layer.

class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)


# adapted from example GPT code  https://github.com/karpathy/ng-video-lecture
class Head(nn.Module):
	"""
	Implements a single attention head.
	"""

	def __init__(self, head_size: int, input_dim: int, dropout_rate: float, dag: torch.Tensor):
		super().__init__()
		self.key = nn.Linear(input_dim, head_size, bias=False)
		self.query = nn.Linear(input_dim, head_size, bias=False)
		self.value = nn.Linear(input_dim, head_size, bias=False)

		self.head_size = head_size
		# user a register buffer (not a module parameter) for the creation of self.dag
		# dag will determine what variables can communicate with each other
		self.dag_orig = dag
		self.register_buffer('dag_mod', self.dag_orig)  # include transpose
		self.dropout = nn.Dropout(dropout_rate)
		self.act = Swish()
		self.att_wei = None

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for the Head module.
		"""
		K = self.key(X)  # B, T, hs
		Q = self.query(X)  # B, T, hs
		V = self.value(X)  # B, T, hs
		B, T, HS = Q.shape
		QK = torch.matmul(Q, K.transpose(1, 2)) / (self.head_size ** 0.5)
		self.att_wei = QK.masked_fill(self.dag_mod == 0, float('-inf'))
		self.att_wei = F.softmax(self.att_wei, dim=-1)
		nan_rows = torch.any(torch.isnan(self.att_wei),
		                     dim=-1)  # check if any rows are <all> -inf, these need to be masked to 0
		nan_mask = nan_rows.unsqueeze(-1).expand_as(self.att_wei)
		self.att_wei = torch.where(nan_mask, torch.zeros_like(self.att_wei),
		                           self.att_wei)  # set any rows have nan values (because they have no causal parents) to 0 to avoid nans
		out = self.att_wei.transpose(1, 2) @ V  # B, T, hs  Transpose DAG to deal extract correct embeddings from V
		out = self.act(out)
		return out


class MultiHeadAttention(nn.Module):
	"""
	Implements multi-head attention combining several heads.
	"""

	def __init__(self, num_heads: int, input_dim: int, head_size: int, dropout_rate: float, dag: torch.Tensor):
		super().__init__()
		self.heads = nn.ModuleList(
			[Head(head_size=head_size, input_dim=input_dim, dropout_rate=dropout_rate, dag=dag) for _ in
			 range(num_heads)])
		self.projection = nn.Linear(int(head_size * num_heads), input_dim)
		self.dropout = nn.Dropout(dropout_rate)
		self.act = nn.LeakyReLU()

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for the MultiHeadAttention module.
		"""
		out = torch.cat([h(X) for h in self.heads], dim=-1)  # B, T, num_heads * head_size
		out = self.dropout(self.projection(out))  # B, T, input_dim
		return self.act(out)


class FF(nn.Module):
	"""
	Implements a feedforward neural network.
	"""

	def __init__(self, input_dim: int, ff_n_embed: int, dropout_rate: float):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, ff_n_embed),
			nn.ReLU(),
			nn.Linear(ff_n_embed, input_dim),
			nn.Dropout(dropout_rate),
		)

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for the FF module.
		"""
		out = self.net(X)
		return out


class Block(nn.Module):
	"""
	Implements a block containing MultiHeadAttention followed by a feedforward network.
	"""

	def __init__(self, ff_n_embed: int, num_heads: int, input_dim: int, head_size: int, dropout_rate: float,
	             dag: torch.Tensor):
		super().__init__()

		self.ff_n_embed = ff_n_embed
		self.input_dim = input_dim
		self.head_size = head_size
		self.num_heads = num_heads
		self.mha = MultiHeadAttention(num_heads=self.num_heads, input_dim=self.input_dim, head_size=self.head_size,
		                              dropout_rate=dropout_rate, dag=dag)
		self.ff = FF(input_dim=input_dim, ff_n_embed=self.ff_n_embed, dropout_rate=dropout_rate)
		if isinstance(dag, torch.Tensor):
			dag = dag.clone().detach()
		else:
			dag = torch.tensor(dag, dtype=torch.float)  # Only convert to tensor if not already one
		self.register_buffer('dag_mask', dag.unsqueeze(0))  # Adding batch dimension

		self.ln1 = nn.LayerNorm(self.input_dim)
		self.ln2 = nn.LayerNorm(self.input_dim)
		self.ln3 = nn.LayerNorm(self.input_dim)

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for the Block module.
		"""

		mha_out = self.ln1(self.mha(X))

		ff_out = self.ln2(self.ff(mha_out))
		mha_out = self.ln3(mha_out + ff_out)
		return mha_out


class CaT(nn.Module):
	def __init__(
			self,
			input_dim: int,
			num_heads: int,
			ff_n_embed: int,
			head_size: int,
			n_layers: int,
			dag: nx.DiGraph,
			dropout_rate: float,
			var_types_sorted: Dict[str, str],
			causal_ordering: Dict[str, int],
			device: torch.device
	):
		'''
		Initialize components of the Causal Transformer.

		Args:
			input_dim (int): Dimensionality of the input embeddings.
			num_heads (int): Number of attention heads.
			ff_n_embed (int): Dimensionality of the feedforward network inside the multi-head attention.
			head_size (int): Dimension of each attention head.
			n_layers (int): Number of layers in the network.
			dag (networkx.DiGraph): Topologically sorted directed acyclic graph.
			dropout_rate (float): Dropout rate to use within attention and feedforward layers.
			var_types_sorted (dict): Dictionary specifying the variable types ('bin', 'cont', 'cat').
			causal_ordering (dict): Ordering of variables for causal relationships.
			device (torch.device): The device the model should use.
		'''
		super().__init__()
		self.input_dim = input_dim
		self.num_heads = num_heads
		self.ff_n_embed = ff_n_embed
		self.head_size = head_size
		self.nxdag = dag
		self.orig_var_name_ordering = list(self.nxdag.nodes())
		dag = torch.tensor(nx.to_numpy_array(self.nxdag)).to(device)
		self.device = device
		self.n_layers = n_layers
		self.dropout_rate = dropout_rate
		self.causal_ordering = causal_ordering

		self.blocks = nn.ModuleList()
		self.loss_func = MixedLoss(var_types_sorted, orig_var_name_ordering=self.orig_var_name_ordering)
		self.lm_head = nn.Linear(self.input_dim, self.input_dim)

		self.ln = nn.LayerNorm(self.input_dim)

		# Store original and setup DAG
		self.original_dag = torch.tensor(dag, dtype=torch.float, device=device)
		self.eye = torch.eye(self.original_dag.size(0), device=device)
		self.was_shuffled = False

		self.initialize_blocks()

	def modify_dag(self, layer_index: int, dag: torch.Tensor) -> torch.Tensor:
		"""
		Adjusts the DAG for a given layer by optionally adding an identity matrix.

		Args:
			layer_index (int): Index of the current layer, determining if identity should be added.
			dag (torch.Tensor): The current DAG tensor.

		Returns:
			torch.Tensor: The modified DAG tensor.
		"""

		modified_dag = dag.clone()
		if layer_index > 0:  # Add identity diagonal to ensure self-connections in subsequent layers
			modified_dag += self.eye
		return torch.clamp(modified_dag, 0, 1)

	def initialize_blocks(self) -> None:
		"""
		Initializes each layer/block with the appropriate DAG setup for the model.
		"""
		for i in range(self.n_layers):
			current_dag = self.modify_dag(i, self.original_dag)
			self.blocks.append(Block(ff_n_embed=self.ff_n_embed, num_heads=self.num_heads,
			                         input_dim=self.input_dim, head_size=self.head_size,
			                         dropout_rate=self.dropout_rate, dag=current_dag))

	def reset_blocks(self) -> None:
		"""
		Resets the DAGs in all heads of all blocks to their original configurations.
		"""
		for i, block in enumerate(self.blocks):
			original_dag = self.modify_dag(i, self.original_dag)
			for head in block.mha.heads:
				head.dag_mod = original_dag

	def forward(self, X: torch.Tensor, targets: Optional[torch.Tensor] = None, shuffling: bool = False) -> Union[
		torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict]]:
		"""
		Processes input through the model, applying shuffling if specified.

		Args:
			X (torch.Tensor): Input tensor.
			targets (torch.Tensor): Target tensor, optional.
			shuffling (bool): Whether to shuffle the input and corresponding DAG.

		Returns:
			torch.Tensor or tuple: Output tensor or tuple of output, loss, and loss tracker if targets provided.
		"""

		shuffle_ordering = None
		if shuffling:
			# Shuffle X, targets, and the DAG using the shuffler function
			X, shuffled_dag, targets, shuffle_ordering = shuffler(X, targets, self.original_dag.clone().cpu().numpy())
			shuffled_dag = torch.tensor(shuffled_dag, dtype=torch.float, device=self.device)

			# Apply the shuffled DAG to each block
			for i, block in enumerate(self.blocks):
				updated_dag = self.modify_dag(i, shuffled_dag)
				for head in block.mha.heads:
					head.dag_mod = updated_dag
			self.was_shuffled = True  # Mark that we have shuffled in this pass
		else:
			if self.was_shuffled:
				# If the previous call used shuffling but this one does not, reset the DAGs
				self.reset_blocks()
				self.was_shuffled = False  # Reset the shuffling flag as we have reverted to the original DAG
			shuffle_ordering = np.arange(X.shape[1])

		for block in self.blocks:
			X = block(X)
		X = self.lm_head(self.ln(X))

		if targets is None:
			return X
		else:
			loss, loss_tracker = self.loss_func(X, targets, shuffle_ordering)
			return X, loss, loss_tracker


def init_weights(m):
	if isinstance(m, nn.Linear):
		#         nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
		m.weight.data.fill_(1)
		# Check if the linear module has a bias term and set it to 1 if it exists
		if m.bias is not None:
			m.bias.data.fill_(0)


def initialize_model_weights(model):
	# Apply init_weights to all sub-modules of the model
	model.apply(init_weights)


class MixedLoss(nn.Module):
	def __init__(self, var_types_sorted, orig_var_name_ordering):
		super(MixedLoss, self).__init__()
		self.orig_var_name_ordering = orig_var_name_ordering
		self.var_types_sorted = var_types_sorted  # sorted types for determining which loss to use
		self.cont_loss = nn.MSELoss()  # Loss for continuous variables
		self.bin_loss = nn.BCEWithLogitsLoss()  # Loss for binary variables
		self.cat_loss = nn.CrossEntropyLoss()  # takes logits for each class as input

	def forward(self, pred, target, shuffle_ordering):
		total_loss = 0
		loss_tracking = {}

		pred = pred[:, shuffle_ordering]
		target = target[:, shuffle_ordering]

		for i, var_name in enumerate(self.orig_var_name_ordering):

			var_type = self.var_types_sorted[var_name]
			idx = shuffle_ordering[i]

			if var_type == 'cont':
				loss = self.cont_loss(pred[:, i], target[:, i])
			elif var_type == 'bin':
				loss = self.bin_loss(pred[:, i], target[:, i])
			elif var_type == 'cat':
				loss = self.cat_loss(pred[:, i].unsqueeze(0), target[:, i].long())

			loss_tracking[var_name] = loss.item()
			total_loss += loss

		return total_loss, loss_tracking


def shuffler(X, targets, dag):
	shuffle_ordering = np.random.permutation(X.shape[1])
	shuffled_X = X[:, shuffle_ordering]
	shuffled_dag = dag[shuffle_ordering, :][:, shuffle_ordering]
	shuffled_targets = targets[:, shuffle_ordering]
	return shuffled_X, shuffled_dag, shuffled_targets, shuffle_ordering