import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MaskedLinear(nn.Module):
    """A linear layer with an optional mask applied to its weights."""

    def __init__(self, in_features: int, out_features: int, mask, use_bias: bool, device=None):
        """
        Initializes the MaskedLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(MaskedLinear, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features), requires_grad=True).float()
        self.set_mask(mask)

        # set bias conditional on layer number
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True).float()
        else:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False).float()

        # after resetting parameters we have to make sure that the bias is set to zero
        self.reset_parameters()
        if not self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False).float()

        self.to(self.device)

    def set_mask(self, mask: Union[np.ndarray, torch.Tensor]):
        """
        Sets the mask for the linear layer weights.

        Args:
            mask (Union[np.ndarray, torch.Tensor]): The mask to apply to the weights.
                Must be either a NumPy array or a PyTorch tensor.
        """
        if isinstance(mask, np.ndarray):
            original_rows, original_columns = mask.shape
            target_rows, target_columns = self.in_features, self.out_features

            # Calculate the number of times to repeat each row and column
            row_repeats = target_rows // original_rows
            column_repeats = target_columns // original_columns

            # Repeat rows and columns to expand the matrix
            mask = np.repeat(np.repeat(mask, row_repeats, axis=0), column_repeats, axis=1)

            # Convert from NumPy array to Tensor and set the correct dtype
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
        elif isinstance(mask, torch.Tensor):
            # Ensure the tensor is the correct dtype
            mask_tensor = mask.float().to(self.device)
        else:
            raise TypeError("Mask must be a NumPy array or a PyTorch tensor.")

        self.mask = nn.Parameter(mask_tensor, requires_grad=False).to(self.device)

    def reset_parameters(self):
        """Initializes or resets the weights and biases of the layer."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # self.weight.data.fill_(1.0)
        try:  # this only applies if a bias is actually used, but the use of bias is a hyperparameter
            self.bias.data.fill_(0.01)
        except:
            pass

    def forward(self, input):
        masked_weight = self.weight * self.mask
        masked_linear = torch.matmul(input, masked_weight) + self.bias
        return masked_linear


class DAGAutoencoder(nn.Module):
    """A directed acyclic graph (DAG) autoencoder with optional input shuffling."""

    def __init__(self, neurons_per_layer: List[int], dag: nx.DiGraph,
                 var_types: Dict[str, str], causal_ordering: Dict[str, int], dropout_rate: float = 0.5,
                 device=None):
        """
        Initializes the DAGAutoencoder with specified neuron layers, a graph representing the DAG,
        variable types, and a dropout rate.

        Args:
            neurons_per_layer (List[int]): List containing the number of neurons in each layer.
            dag (nx.DiGraph): A directed acyclic graph representing the relationships between neurons in the layers.
            var_types (Dict[str, str]): Dictionary mapping variable names to their types (e.g., continuous, categorical).
            dropout_rate (float, optional): Probability of an element to be zeroed. Defaults to 0.5.
        """
        super(DAGAutoencoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.nxdag = dag
        self.orig_var_name_ordering = list(self.nxdag.nodes())
        self.neurons_per_layer = neurons_per_layer
        self.causal_ordering = causal_ordering
        self.dropout_rate = dropout_rate
        self.var_types = var_types
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(p=self.dropout_rate) for _ in range(len(neurons_per_layer) - 1)])
        self.was_shuffled = False
        self.loss_func = MixedLoss(self.var_types, orig_var_name_ordering=self.orig_var_name_ordering,
                                   causal_ordering=self.causal_ordering)

        initial_adj_matrix = nx.to_numpy_array(dag)
        diag_adj_matrix = initial_adj_matrix + np.eye(initial_adj_matrix.shape[0], dtype=np.float32)

        self.input_layers = nn.ModuleList(
            [MaskedLinear(self.neurons_per_layer[0], self.neurons_per_layer[i], mask=initial_adj_matrix, use_bias=True,
                          device=self.device)
             for i in range(1, len(self.neurons_per_layer))])

        self.feature_layers = nn.ModuleList(
            [MaskedLinear(self.neurons_per_layer[i - 1], self.neurons_per_layer[i], mask=diag_adj_matrix,
                          use_bias=False, device=self.device)
             for i in range(2, len(self.neurons_per_layer))])

        self.activations = nn.ModuleList(
            [Swish() for _ in range(1, len(self.neurons_per_layer) - 1)])

        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(p=self.dropout_rate) for _ in range(1, len(self.neurons_per_layer) - 1)])

        self.to(self.device)

    def set_masks(self, mask: np.ndarray):
        """
        Applies mask to each layer in the autoencoder.

        Args:
            mask (np.ndarray): Mask to apply to the linear layers.
        """

        for layer in self.input_layers:
            layer.set_mask(mask)

        for layer in self.feature_layers:
            layer.set_mask(mask + np.eye(mask.shape[0], dtype=np.float32))

    def forward(self, X: torch.Tensor, targets: Optional[torch.Tensor] = None,
                shuffling: bool = False, verbose: bool = False) -> torch.Tensor:
        """
        Processes the input through the autoencoder, optionally shuffling input connections.

        Args:
            X (torch.Tensor): The input tensor.
            shuffle (bool, optional): Whether to shuffle the input connections. Defaults to False.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        orginal_shape = X.shape
        X = X.view(X.size(0), -1)
        shuffle_ordering = np.arange(X.shape[1])
        if shuffling:
            shuffle_ordering = torch.randperm(X.size(1))
            X = X[:, shuffle_ordering]
            shuffled_mask = nx.to_numpy_array(self.nxdag)[shuffle_ordering][:, shuffle_ordering]
            self.set_masks(shuffled_mask)
            self.was_shuffled = True

        elif shuffling == False and self.was_shuffled:
            self.set_masks(nx.to_numpy_array(self.nxdag))
            self.was_shuffled = False

        for i, (input_linear, feature_linear, activation, dropout) in enumerate(
                zip(self.input_layers,
                    [None] + list(self.feature_layers),
                    list(self.activations) + [None],
                    list(self.dropout_layers) + [None])):
            if feature_linear is None:
                Y = input_linear(X)
            else:
                Y = input_linear(X) + feature_linear(Y)
            if activation is not None:
                Y = activation(Y)
            if dropout is not None:
                Y = dropout(Y)
            if verbose:
                print('layer:', i)
                print(Y)

        Y = Y.view(orginal_shape)

        if targets is None:
            for i, var_name in enumerate(self.orig_var_name_ordering):

                var_type = self.var_types[var_name]
                idx = shuffle_ordering[i]
                if var_type == 'cont':
                    Y[:, idx] = Y[:, idx]
                elif var_type == 'bin':
                    Y[:, idx] = torch.sigmoid(Y[:, idx])

            return Y
        else:
            loss, loss_tracker = self.loss_func(pred=Y, target=targets, shuffle_ordering=shuffle_ordering)
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
                    loss = self.cont_loss(pred[:, idx], target[:, i])
                elif var_type == 'bin':
                    loss = self.bin_loss(pred[:, idx], target[:, i])

                loss_tracking[var_name] = loss.item()
                total_loss += loss

        return total_loss, loss_tracking
