import torch
# from scipy.special import gradient
from torch.nn import functional as F
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Optional, Dict, List, Tuple, Union


# 1. introduce dag into attention [DONE]
# 2. introduce diagonal after first layer [DONE]
# 3. handle skip connection masking issue [DONE]
# 4. layer norm not good [DONE]
# 5. introduce a linear combination layer which still respects the mask [DONE]

# note that the network has MHA with blocks in parallel, but this is also done sequentially, combining
# both network width and network depth.

# for the causal transformer, we have to be careful that we include a 'diagonal' pass-thru after the first layer
# otherwise, and e.g. in a three variable chain A->B->C, the dependency structure will prevent B from being predicted
# from A <after the first layer>, because B is caused by A, not by itself. So the diagonal of ones should be
# introduced after the first layer.

class SwishAct(nn.SiLU):
    def __init__(self):
        super(SwishAct, self).__init__()


class MishAct(nn.Mish):
    def __init__(self):
        super(MishAct, self).__init__()


class LeakyReLUAct(nn.LeakyReLU):
    def __init__(self):
        super(LeakyReLUAct, self).__init__()


class TanhAct(nn.Module):
    def forward(self, x):
        return x * torch.tanh(x)


class ReLUAct(nn.Module):
    def forward(self, x):
        return x * torch.relu(x)


class IdentAct(nn.Module):
    def forward(self, x):
        return x


# adapted from example GPT code  https://github.com/karpathy/ng-video-lecture
class Head(nn.Module):
    """
    Implements a single attention head.
    """

    def __init__(self, head_size: int, input_dim: int, dropout_rate: float, dag: torch.Tensor, use_bias: bool,
                 activation_function: str = 'Swish', device=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.key = nn.Linear(input_dim, head_size, bias=use_bias)
        self.query = nn.Linear(input_dim, head_size, bias=use_bias)
        self.value = nn.Linear(input_dim, head_size, bias=use_bias)

        self.activation_function = activation_function

        self.head_size = head_size
        # user a register buffer (not a module parameter) for the creation of self.dag
        # dag will determine what variables can communicate with each other
        self.dag_orig = dag.to(self.device).float()
        self.register_buffer('dag_mod', self.dag_orig)  # include transpose
        self.dropout = nn.Dropout(dropout_rate)

        activation_map = {
            'Swish': SwishAct(),
            'ReLU': ReLUAct(),
            'tanh': TanhAct(),
            'Ident': IdentAct(),
            'Mish': MishAct(),
            'LeakyReLU': LeakyReLUAct(),
        }

        self.act = activation_map.get(self.activation_function, None)

        self.to(self.device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Head module.
        """
        # print(X.shape)
        # print(f"Y.shape {Y.shape}")
        K = self.key(X)  # B, T, hs
        Q = self.query(Y)  # B, T, hs
        V = self.value(X)  # B, T, hs
        # print(f"Q.shape {Q.shape}")
        # print(f"K.shape {K.shape}")
        # B, T, HS = Q.shape
        S_qk = torch.matmul(Q, K.transpose(1, 2)) / (self.head_size ** 0.5)
        # print(f"S_qk.shape {S_qk.shape}")
        # Could add a matrix with -inf values instead
        self.Sprime = self.dag_mod.T * S_qk
        self.Sprime = self.Sprime.masked_fill(self.Sprime == 0, float('-inf'))
        self.Sprime = F.softmax(self.Sprime, dim=-1)
        nan_rows = torch.any(torch.isnan(self.Sprime),
                             dim=-1)  # check if any rows are <all> -inf, these need to be masked to 0
        nan_mask = nan_rows.unsqueeze(-1).expand_as(self.Sprime).to(self.device)
        self.Sprime = torch.where(nan_mask, torch.zeros_like(self.Sprime),
                                  self.Sprime)  # set any rows have nan values (because they have no causal parents) to 0 to avoid nans
        self.Sprime = self.Sprime.float()
        O = self.Sprime @ V  # B, T, hs  Transpose DAG to deal extract correct embeddings from V
        # print(f"O.shape {O.shape}")
        return O


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention combining several heads.
    """

    def __init__(self, num_heads: int, input_dim: int, head_size: int, dropout_rate: float, dag: torch.Tensor,
                 use_bias: bool = False, activation_function: str = 'Swish', device: Optional[torch.device] = None):
        super().__init__()

        self.activation_function = activation_function

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.heads = nn.ModuleList(
            [Head(head_size=head_size, input_dim=input_dim, dropout_rate=dropout_rate,
                  dag=dag, use_bias=use_bias, device=self.device, activation_function=self.activation_function)
             for i in range(num_heads)]
        )

        self.projection = nn.Linear(int(head_size * num_heads), input_dim, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.to(self.device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiHeadAttention module.
        """
        out = torch.cat([h(X, Y) for h in self.heads], dim=-1)  # B, T, num_heads * head_size
        out = self.projection(out)
        out = self.dropout(out)  # B, T, input_dim
        return out


class FF(nn.Module):
    """
    Implements a feedforward neural network.
    """

    def __init__(self, input_dim: int, ff_n_embed: int, dropout_rate: float, activation_function: str = 'Swish'):
        super().__init__()

        self.activation_function = activation_function

        self.dropout = nn.Dropout(dropout_rate)
        activation_map = {
            'Swish': SwishAct(),
            'ReLU': ReLUAct(),
            'tanh': TanhAct(),
            'Ident': IdentAct(),
            'Mish': MishAct(),
            'LeakyReLU': LeakyReLUAct(),
        }

        self.act = activation_map.get(self.activation_function, None)

        self.net = nn.Sequential(
            nn.Linear(input_dim, ff_n_embed),
            self.act,
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
                 dag: torch.Tensor, use_batch_norm: bool, input_n_var: int, use_bias: bool = False,
                 device: Optional[torch.device] = None,
                 activation_function: str = 'Swish'):
        super().__init__()
        self.activation_function = activation_function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.ff_n_embed = ff_n_embed
        self.input_dim = input_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(num_heads=self.num_heads, input_dim=self.input_dim, head_size=self.head_size,
                                      dropout_rate=dropout_rate, dag=dag, use_bias=use_bias, device=self.device,
                                      activation_function=self.activation_function)
        self.ff = FF(input_dim=input_dim, ff_n_embed=self.ff_n_embed, dropout_rate=dropout_rate,
                     activation_function=self.activation_function)
        if use_batch_norm:
            self.batch_norms = nn.ModuleList(nn.BatchNorm1d(num_features=input_n_var) for _ in range(2))
        else:
            self.batch_norms = None
        if isinstance(dag, torch.Tensor):
            dag = dag.clone().detach()
        else:
            dag = torch.tensor(dag, dtype=torch.float32)  # Only convert to tensor if not already one
        self.register_buffer('dag_mask', dag.unsqueeze(0))  # Adding batch dimension
        self.to(self.device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Block module.
        """
        mha_out = self.mha(X, Y) + Y
        if self.batch_norms is not None:
            mha_out = self.batch_norms[0](mha_out)
        ff_out = self.ff(mha_out) + mha_out
        if self.batch_norm is not None:
            ff_out = self.batch_norms[1](ff_out)
        return ff_out


class DynamicLinearEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_vectors):
        super(DynamicLinearEmbedding, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_vectors)])

    def forward(self, X):
        # print(f"Dynamic lin Emb X:{X}")
        return torch.stack([self.linear_layers[i](X[:, i, :]) for i in range(X.shape[-2])], dim=-2)


class CaT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            num_heads: int,
            ff_n_embed: int,
            head_size: int,
            n_layers: int,
            dag: nx.DiGraph,
            dropout_rate: float,
            var_types: Dict[str, str],
            causal_ordering: Dict[str, int],
            device=None,
            use_batch_norm=True,
            activation_function='Swish'
    ):
        '''
        Initialize components of the Causal Transformer.

        Args:
            input_dim (int): Dimensionality of the input embeddings before modifying it.
            embed_dim (int): Dimensionality of the input embeddings after modification.
            num_heads (int): Number of attention heads.
            ff_n_embed (int): Dimensionality of the feedforward network inside the multi-head attention.
            head_size (int): Dimension of each attention head.
            n_layers (int): Number of layers in the network.
            dag (networkx.DiGraph): Topologically sorted directed acyclic graph.
            dropout_rate (float): Dropout rate to use within attention and feedforward layers.
            var_types(dict): Dictionary specifying the variable types ('bin', 'cont', 'cat').
            causal_ordering (dict): Ordering of variables for causal relationships.
            device (torch.device): The device the model should use.
            activation_function (str): 'Swish', 'ReLU', or 'tanh'
        '''
        super().__init__()
        self.input_n_var = len(var_types)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_n_embed = ff_n_embed
        self.head_size = head_size
        self.nxdag = dag
        self.orig_var_name_ordering = list(self.nxdag.nodes())
        dag = torch.tensor(nx.to_numpy_array(self.nxdag), dtype=torch.float32).to(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.causal_ordering = causal_ordering
        self.var_types = var_types
        self.activation_function = activation_function
        self.use_batch_norm = use_batch_norm

        self.y_0 = nn.Parameter(torch.randn([self.input_n_var, self.embed_dim]), requires_grad=True)
        self.embedding = DynamicLinearEmbedding(input_dim=self.input_dim, output_dim=self.embed_dim,
                                                num_vectors=self.input_n_var)

        self.blocks = nn.ModuleList()  # main bulk of network
        self.lm_head = nn.Linear(self.embed_dim, self.input_dim, bias=True)  # very last dense layer

        self.loss_func = MixedLoss(self.var_types, orig_var_name_ordering=self.orig_var_name_ordering,
                                   causal_ordering=self.causal_ordering)

        self.original_dag = dag.clone().detach()
        self.was_shuffled = False

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        # Initialize each block and the rest of the model
        self.initialize_blocks()
        self.apply(init_weights)
        self.to(self.device)

    def initialize_blocks(self) -> None:
        """
        Initializes each layer/block with the appropriate DAG setup for the model.
        """
        for i in range(self.n_layers):
            self.blocks.append(Block(ff_n_embed=self.ff_n_embed, num_heads=self.num_heads,
                                     input_dim=self.embed_dim, head_size=self.head_size,
                                     dropout_rate=self.dropout_rate,
                                     use_batch_norm=self.use_batch_norm,
                                     input_n_var=self.input_n_var,
                                     use_bias=True, dag=self.original_dag,
                                     device=self.device,
                                     activation_function=self.activation_function))  # use_bias=(i >= 1), dag=current_dag, device=self.device))

    def reset_dags(self) -> None:
        """
        Resets the DAGs in all heads of all blocks to their original configurations.
        """
        for i, block in enumerate(self.blocks):
            for head in block.mha.heads:
                head.dag_mod = self.original_dag

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                shuffling: bool = False) -> Union[
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
            shuffled_dag = torch.tensor(shuffled_dag, dtype=torch.float32, device=self.device)

            # Apply the shuffled DAG to each block
            for i, block in enumerate(self.blocks):
                for head in block.mha.heads:
                    head.dag_mod = shuffled_dag
            self.was_shuffled = True  # Mark that we have shuffled in this pass
        else:
            if self.was_shuffled:
                # If the previous call used shuffling but this one does not, reset the DAGs
                self.reset_dags()
                self.was_shuffled = False  # Reset the shuffling flag as we have reverted to the original DAG
            shuffle_ordering = np.arange(X.shape[1])
        X_emb = self.embedding(X)
        Y = self.y_0
        for block in self.blocks:
            Y = block(X_emb, Y)

        Y = self.lm_head(Y)
        if targets is None:
            for i, var_name in enumerate(self.orig_var_name_ordering):

                var_type = self.var_types[var_name]
                idx = shuffle_ordering[i]
                if var_type == 'cont':
                    Y[:, idx] = Y[:, idx]
                elif var_type == 'bin':
                    Y[:, idx] = torch.sigmoid(Y[:, idx])
            if mask:
                Y = Y * mask
            return Y
        else:
            loss, loss_tracker = self.loss_func(Y, targets, shuffle_ordering)
            return Y, loss, loss_tracker


class MixedLoss(nn.Module):
    def __init__(self, var_types, orig_var_name_ordering, causal_ordering):
        super(MixedLoss, self).__init__()
        self.causal_ordering = causal_ordering
        self.orig_var_name_ordering = orig_var_name_ordering
        self.var_types = var_types  # sorted types for determining which loss to use
        self.cont_loss = nn.MSELoss()  # Loss for continuous variables
        self.bin_loss = nn.BCEWithLogitsLoss()  # Loss for binary variables

    def forward(self, pred, target, shuffle_ordering):
        total_loss = 0
        loss_tracking = {}

        pred = pred[:, shuffle_ordering]
        target = target[:, shuffle_ordering]

        for i, var_name in enumerate(self.orig_var_name_ordering):
            if self.causal_ordering[var_name] != 0:  # don't compute loss for exogenous vars
                var_type = self.var_types[var_name]
                idx = shuffle_ordering[i]

                if var_type == 'cont':
                    loss = self.cont_loss(pred[:, idx], target[:, idx])
                elif var_type == 'bin':
                    loss = self.bin_loss(pred[:, idx], target[:, idx])

                loss_tracking[var_name] = loss.item()
                total_loss += loss

        return total_loss, loss_tracking


def shuffler(X, targets, dag):
    shuffle_ordering = np.random.permutation(X.shape[1])
    shuffled_X = X[:, shuffle_ordering]
    shuffled_dag = dag[shuffle_ordering, :][:, shuffle_ordering]
    shuffled_targets = targets[:, shuffle_ordering]
    return shuffled_X, shuffled_dag, shuffled_targets, shuffle_ordering
