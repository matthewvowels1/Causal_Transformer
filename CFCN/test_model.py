import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Union, Optional, Dict
import utils


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MaskedLinear(nn.Module):
    """A linear layer with an optional mask applied to its weights."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool, device=None):
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
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True).float()
        self.mask = None

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
        masked_weight = self.weight.transpose(0, 1) * self.mask
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
        utils.assert_neuron_layers(layers=neurons_per_layer, input_size=len(var_types.keys()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.dag = dag
        self.orig_var_name_ordering = list(self.dag.nodes())
        self.neurons_per_layer = neurons_per_layer
        self.causal_ordering = causal_ordering
        self.input_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.var_types = var_types
        self.activations = nn.ModuleList()
        self.original_masks = []  # This will be set initially and not changed
        self.current_masks = []  # Masks currently being used by the layers
        self.original_adj_matrix = torch.tensor(nx.to_numpy_array(self.dag),
                                                dtype=torch.float64)  # Store the original adjacency matrix
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(p=self.dropout_rate) for _ in range(len(neurons_per_layer) - 1)])
        self.was_shuffled = False
        self.loss_func = MixedLoss(self.var_types, orig_var_name_ordering=self.orig_var_name_ordering,
                                   causal_ordering=self.causal_ordering)

        for i in range(len(self.neurons_per_layer) - 1):
            linear_layer = MaskedLinear(neurons_per_layer[i], self.neurons_per_layer[i + 1], use_bias=False,
                                        device=self.device)
            input_layer = MaskedLinear(neurons_per_layer[0], self.neurons_per_layer[i + 1], use_bias=True,
                                       device=self.device)
            self.layers.append(linear_layer)
            self.input_layers.append(input_layer)
            if i < len(self.neurons_per_layer) - 2:
                self.activations.append(Swish())

        self.to(self.device)

    def initialize_masks(self, masks: List[torch.Tensor]):
        """
        Initializes the original masks for the autoencoder. This is intended to be called only once.

        Args:
            masks (List[torch.Tensor]): List of tensors representing the masks for each layer.
        """
        # Initialize original masks, only called once
        self.original_masks = [mask.clone() for mask in masks]
        self.set_masks(masks)

    def set_masks(self, masks: List[torch.Tensor]):
        """
        Applies masks to each layer in the autoencoder.

        Args:
            masks (List[torch.Tensor]): Masks to apply to the linear layers.
        """
        self.current_masks = []
        # Apply masks only to linear layers
        assert len(masks) == len(self.layers), "The number of masks must match the number of linear layers."
        for layer, mask in zip(self.layers, masks):
            layer.set_mask(mask)
            self.current_masks.append(mask)

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
        shuffle_ordering = np.arange(X.shape[1])
        if shuffling:
            shuffle_ordering = torch.randperm(X.size(1))
            X = X[:, shuffle_ordering]
            shuffled_matrix = self.original_adj_matrix[shuffle_ordering][:, shuffle_ordering].numpy()
            shuffled_masks = [torch.from_numpy(mask).float().to(torch.float64) for mask in
                              utils.expand_adjacency_matrix(self.neurons_per_layer[1:], shuffled_matrix)]
            self.set_masks(shuffled_masks)
            self.was_shuffled = True

        elif shuffling == False and self.was_shuffled:
            self.set_masks(self.original_masks)
            self.was_shuffled = False

        for i, (input_linear, linear, activation) in enumerate(zip(self.input_layers, self.layers, self.activations)):
            # Put stuff here X -> X + softmax(linear layer) -> X
            if i == 0:
                Y = activation(input_linear(X))
            else:
                Y = activation(Y + input_linear(X) + linear(Y))
            if verbose:
                print('layer:', i)
                print(Y)
        # X = self.dropout_layers[i](X)

        X = Y + self.input_layers[-1](Y) + self.layers[-1](Y)

        if targets is None:
            for i, var_name in enumerate(self.orig_var_name_ordering):

                var_type = self.var_types[var_name]
                idx = shuffle_ordering[i]
                if var_type == 'cont':
                    X[:, idx] = X[:, idx]
                elif var_type == 'bin':
                    X[:, idx] = torch.sigmoid(X[:, idx])

            return X
        else:
            loss, loss_tracker = self.loss_func(pred=X, target=targets, shuffle_ordering=shuffle_ordering)
            return X, loss, loss_tracker


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
