import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from typing import Optional

def check_preds(model, dataset: torch.utils.data.Dataset, new_adj_matrix: Optional[np.ndarray] = None) -> None:
    """
    Evaluates predictions with a new adjacency matrix to simulate interventions on the causal system.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be used for evaluating the model.
        new_adj_matrix (Optional[np.ndarray]): New adjacency matrix to be used to update the model's masks, if provided.
    """
    model.eval()
    if new_adj_matrix is not None:# Convert the numpy array to tensor
        # Update the masks based on the new adjacency matrix
        new_masks = [torch.from_numpy(mask).float().to(torch.float64) for mask in utils.expand_adjacency_matrix(model.neurons_per_layer[1:], new_adj_matrix)]
        model.set_masks(new_masks)

    # Extract all data for plotting
    all_inputs = dataset[:][0]  # Assuming dataset is globally accessible and properly formatted

    with torch.no_grad():
        all_predictions = model(all_inputs, shuffle=False).numpy()

    if isinstance(all_inputs, torch.Tensor):
        actual_data = all_inputs.numpy()
    else:
        actual_data = all_inputs

    # Variables are assumed to be in columns: A, B, C, D
    variables = ['A', 'B', 'C', 'D']

    # Create scatter plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes = axes.ravel()  # Flatten the axis array

    for i, ax in enumerate(axes):
        ax.scatter(actual_data[:, i], all_predictions[:, i], alpha=0.5)
        ax.set_title(f'Predicted vs Actual for {variables[i]}')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.plot([actual_data[:, i].min(), actual_data[:, i].max()],
                [actual_data[:, i].min(), actual_data[:, i].max()], 'k--')  # Diagonal line

    plt.tight_layout()
    plt.show()
