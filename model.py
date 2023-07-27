import torch
from torch.nn import functional as F
import torch.nn as nn
import networkx as nx
import numpy as np

def adjacency_mod(adjacency_matrix, causal_ordering):  # TODO: remove if function not needed
    # this is just a temporary function to see if adding a 1 to the diagonal for all zeroth-order variables helps
    for i, var in enumerate(causal_ordering.keys()):
        order = causal_ordering[var]
        if order == 0:
            adjacency_matrix[i, i] = 1  # add 1 to the i-th diagonal element
    return adjacency_matrix



# adapted from example GPT code  https://github.com/karpathy/ng-video-lecture
class Head(nn.Module):
    def __init__(self, head_size, dropout_rate, dag):
        super().__init__()
        self.key = nn.Linear(1, head_size, bias=False)
        self.query = nn.Linear(1, head_size, bias=False)
        self.value = nn.Linear(1, head_size, bias=False)
        # user a register buffer (not a module parameter) for the creation of self.dag
        # dag will determine what variables can communicate with each other

        self.dag_orig = dag

        matrix = torch.ones_like(self.dag_orig)
        self.dag_orig = torch.tril(matrix, diagonal=-1)   #if triu.diagonal=1 or tril.diagonal=-1 then <remove> diagonal  # TODO: remove these last two lines once testing is complete

        self.register_buffer('dag_mod', self.dag_orig)  # include transpose
        self.dropout = nn.Dropout(dropout_rate)

        self.att_wei = None
    def forward(self, X):

        K = self.key(X)  # B, T, hs
        Q = self.query(X)  # B, T, hs
        V = self.value(X)  # B, T, hs
        B, T, HS = Q .shape
        QK = (Q * K)  # elementwise multiplication  (bs, dims, 1), this is not the regular attention computation
        self.att_wei = QK.repeat(1, 1, T).transpose(-2, -1)  # stack horizontally (bs, dims, dims)
        self.att_wei = self.att_wei.masked_fill(self.dag_mod == 0, float('-inf'))
        self.att_wei = F.softmax(self.att_wei, dim=-1)
        nan_rows = torch.any(torch.isnan(self.att_wei), dim=-1)  # check if any rows are <all> -inf, these need to be masked to 0
        nan_mask = nan_rows.unsqueeze(-1).expand_as(self.att_wei)
        self.att_wei = torch.where(nan_mask, torch.zeros_like(self.att_wei), self.att_wei) # set any rows have nan values (because they have no causal parents) to 0 to avoid nans
        self.att_wei = self.dropout(self.att_wei)
        out = self.att_wei @ V
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
        out = self.net(X)
        return out


class Block(nn.Module):

    def __init__(self, n_embed, num_heads, head_size,  dropout_rate, dag):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, head_size, dropout_rate, dag)
        self.ff = FF(n_embed, dropout_rate)

    def forward(self, X):
        X = self.mha(X)  # + X  with skip connection (careful with adding back in after having masked it)
        X = X + self.ff(X)  # with skip connection
        return X


class MixedLoss(nn.Module):
    def __init__(self, var_types_sorted, causal_ordering):
        super(MixedLoss, self).__init__()
        self.causal_ordering = causal_ordering
        self.var_types_sorted = var_types_sorted  # sorted types for determining which loss to use
        self.cont_loss = nn.MSELoss()  # Loss for continuous variables
        self.bin_loss = nn.BCEWithLogitsLoss()  # Loss for binary variables
        self.cat_loss = nn.CrossEntropyLoss()   # takes logits for each class as input

    def forward(self, pred, target):

        total_loss = 0
        loss_tracking = {}
        for i, var_name in enumerate(self.var_types_sorted.keys()):
            var_type = self.var_types_sorted[var_name]
            order = self.causal_ordering[var_name]
            # if order != 0:  # TODO: include condition if needed (up to function return)
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
    # shuffles the order of X, targets, and adjacency matrix for a batch
    shuffle_ordering = np.random.permutation(X.shape[1])
    return X[:, shuffle_ordering], dag[shuffle_ordering, :], targets[:, shuffle_ordering]

class CaT(nn.Module):

    def __init__(self, dropout_rate, num_heads, head_size, n_layers, dag, device, ordering, var_types):
        '''
        :param dropout_rate:
        :param num_heads:
        :param head_size:
        :param n_layers:
        :param dag: topologically sorted networkx digraph object
        :param device: 'cuda' or 'cpu'
        :param ordering: a dictionary of variable names with corresponding index in causal ordering
        :param var_types: a dictionary of variable types 'bin' (binary) 'cont' (continuous) or 'cat' (categorical)
        '''

        super().__init__()
        var_types_sorted = {k: var_types[k] for k in list(dag.nodes)}  # get the variable types bin/cont/cat in the 'right order'
        self.n_layers = n_layers
        self.num_heads =num_heads
        self.ordering = ordering
        self.nxdag = dag
        dag = torch.tensor(nx.to_numpy_array(dag)).to(device).T  # get adjacency matrix and add diagonals (TODO: remove adj_mod if not needed)
        self.device = device
        self.blocks = nn.Sequential(
            *[Block(n_embed=1, num_heads=num_heads, head_size=head_size, dropout_rate=dropout_rate, dag=dag) for _ in range(n_layers)])
        self.lm_head = nn.Linear(1, 1)
        self.loss_func = MixedLoss(var_types_sorted, ordering)

    def forward(self, X, targets=None):
        X = X[:, :, None]  # (B, num_vars, C=1)

        if targets is not None:
            orig_dag = self.blocks[0].mha.heads[0].dag_orig
            X, shuffled_dag, targets = shuffler(X, targets, orig_dag)
            for i in range(self.n_layers):
                for j in range(self.num_heads):
                    self.blocks[i].mha.heads[j].dag_mod = shuffled_dag  # changes the ordering of X, targets, and dag on a per batch basis

        elif targets == None:
            for i in range(self.n_layers):
                for j in range(self.num_heads):
                    self.blocks[i].mha.heads[j].dag_mod = self.blocks[0].mha.heads[0].dag_orig

        X = self.blocks(X)  # B, num_vars, head_size
        X = self.lm_head(X)
        X = X[:, :, 0]

        if targets == None:
            return X
        elif targets is not None:
            loss, loss_tracker = self.loss_func(X, targets)  # pull out the Y variable as the target
            return X, loss, loss_tracker


