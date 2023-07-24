import tiktoken
import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
import networkx as nx


# adapted from example GPT code  https://github.com/karpathy/ng-video-lecture
class Head(nn.Module):
    def __init__(self, head_size, dropout_rate, dag):
        super().__init__()
        self.key = nn.Linear(1, head_size, bias=False)
        self.query = nn.Linear(1, head_size, bias=False)
        self.value = nn.Linear(1, head_size, bias=False)
        # user a register buffer (not a module parameter) for the creation of self.dag
        # dag will determine what variables can communicate with each other
        self.register_buffer('dag', dag.T)  # include transpose
        self.dropout = nn.Dropout(dropout_rate)

        self.wei = None
    def forward(self, X):
        # B, T, C = X.shape  batch size, time steps, channels
        K = self.key(X)  # B, T, hs
        Q = self.query(X)  # B, T, hs

        self.wei = Q @ K.transpose(-2, -1) * K.shape[-1] ** -0.5
        self.wei = self.wei.masked_fill(self.dag == 0, float('-inf'))

        self.wei = F.softmax(self.wei, dim=-1)
        nan_rows = torch.any(torch.isnan(self.wei), dim=-1)  # check if any rows are <all> -inf, these need to be masked to 0
        nan_mask = nan_rows.unsqueeze(-1).expand_as(self.wei)
        self.wei = torch.where(nan_mask, torch.zeros_like(self.wei), self.wei) # set any rows have nan values (because they have no causal parents) to 0 to avoid nans
        self.wei = self.dropout(self.wei)
        V = self.value(X)
        out = self.wei @ V
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
        self.sa = MultiHeadAttention(num_heads, head_size, dropout_rate, dag)
        self.ff = FF(n_embed, dropout_rate)

    def forward(self, X):
        X = X + self.sa(X)  # with residual skip connection and learnable normalization of features
        X = X + self.ff(X)  # with residual skip connection and learnable normalization of features
        return X


class MixedLoss(nn.Module):
    def __init__(self, var_types_sorted):
        super(MixedLoss, self).__init__()
        self.var_types_sorted = var_types_sorted  # sorted types for determining which loss to use
        self.cont_loss = nn.MSELoss()  # Loss for continuous variables
        self.bin_loss = nn.BCEWithLogitsLoss()  # Loss for binary variables
        self.cat_loss = nn.CrossEntropyLoss()   # takes logits for each class as input

    def forward(self, pred, target):
        total_loss = 0

        for i, var_type in enumerate(self.var_types_sorted.values()):
            if var_type == 'cont':
                total_loss += self.cont_loss(pred[:, i], target[:, i])
            elif var_type == 'bin':
                total_loss += self.bin_loss(pred[:, i], target[:, i])
            elif var_type == 'cat':
                total_loss += self.cat_loss(pred[:, i].unsqueeze(0), target[:, i].long())

        return total_loss

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
        self.ordering = ordering
        self.nxdag = dag
        self.dag = torch.tensor(nx.to_numpy_array(dag)).to(device)  # get adjacency matrix in numpy form and torch.tensor it
        self.device = device
        self.blocks = nn.Sequential(
            *[Block(n_embed=1, num_heads=num_heads, head_size=head_size, dropout_rate=dropout_rate, dag=self.dag) for _ in range(n_layers)])
        self.lm_head = nn.Linear(1, 1)
        self.loss_func = MixedLoss(var_types_sorted)

    def forward(self, X, targets=None):
        X = X[:, :, None]  # (B, num_vars, C=1)
        X = self.blocks(X)  # B, num_vars, head_size
        X = self.lm_head(X)
        X = X[:, :, 0]

        if targets == None:
            return X
        else:
            loss = self.loss_func(X, targets)  # pull out the Y variable as the target

            return X, loss


