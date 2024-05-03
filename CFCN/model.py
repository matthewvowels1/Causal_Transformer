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

    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the MaskedLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.mask = None
        self.reset_parameters()

    def set_mask(self, mask: Union[np.ndarray, torch.Tensor]):
        """
        Sets the mask for the linear layer weights.

        Args:
            mask (Union[np.ndarray, torch.Tensor]): The mask to apply to the weights.
                Must be either a NumPy array or a PyTorch tensor.
        """
        if isinstance(mask, np.ndarray):
            # Convert from NumPy array to Tensor and set the correct dtype
            mask_tensor = torch.from_numpy(mask).to(torch.float64)
        elif isinstance(mask, torch.Tensor):
            # Ensure the tensor is the correct dtype
            mask_tensor = mask.to(torch.float64)
        else:
            raise TypeError("Mask must be a NumPy array or a PyTorch tensor.")

        self.mask = nn.Parameter(mask_tensor, requires_grad=False)


    def reset_parameters(self):
        """Initializes or resets the weights and biases of the layer."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        #         self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.01)

    def forward(self, input):
        if self.mask is not None:
            # Check if mask is not None

            masked_weight = self.weight.transpose(0, 1) * self.mask + self.bias
            # Perform matrix multiplication using torch.matmul()
            return torch.matmul(input, masked_weight)
        else:
            # Perform matrix multiplication using torch.matmul()
            return torch.matmul(input, self.weight.transpose(0, 1)) + self.bias


class DAGAutoencoder(nn.Module):
    """A directed acyclic graph (DAG) autoencoder with optional input shuffling."""

    def __init__(self, neurons_per_layer: List[int], dag: nx.DiGraph,
                 var_types: Dict[str, str], dropout_rate: float = 0.5):
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
        self.dag = dag
        self.orig_var_name_ordering = list(self.dag.nodes())
        self.initial_adj_matrix = nx.to_numpy_array(self.dag)
        self.neurons_per_layer = neurons_per_layer
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.var_types = var_types
        self.activations = nn.ModuleList()
        self.original_masks = []  # This will be set initially and not changed
        self.current_masks = []  # Masks currently being used by the layers
        self.original_adj_matrix = torch.tensor(self.initial_adj_matrix,
                                                dtype=torch.float64)  # Store the original adjacency matrix
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=self.dropout_rate) for _ in range(len(neurons_per_layer) - 1)])
        self.was_shuffled = False
        self.loss_func = MixedLoss(self.var_types, orig_var_name_ordering=self.orig_var_name_ordering)

        for i in range(len(self.neurons_per_layer) - 1):
            linear_layer = MaskedLinear(neurons_per_layer[i], self.neurons_per_layer[i + 1])
            self.layers.append(linear_layer)
            if i < len(self.neurons_per_layer) - 2:
                self.activations.append(Swish())

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
        # Apply masks only to linear layers
        assert len(masks) == len(self.layers), "The number of masks must match the number of linear layers."
        for layer, mask in zip(self.layers, masks):
            layer.set_mask(mask)

    def forward(self, X: torch.Tensor, targets: Optional[torch.Tensor] = None, shuffling: bool = False) -> torch.Tensor:
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

        for i, (linear, activation) in enumerate(zip(self.layers, self.activations)):
            X = linear(X)
            X = activation(X)
            X = self.dropout_layers[i](X)
        X = self.layers[-1](X)

        if targets is None:
            return X
        else:
            loss, loss_tracker = self.loss_func(X, targets, shuffle_ordering)
            return X, loss, loss_tracker


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