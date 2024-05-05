import torch
import numpy as np
import networkx as nx
import utils


def remove_incoming_edges(graph, target_node='X'):
    """
    Creates a copy of the graph, removes all incoming edges to a specified node, and returns the modified graph.

    Args:
        graph (nx.DiGraph): A directed graph.
        target_node: The node for which all incoming edges should be removed.

    Returns:
        nx.DiGraph: A copy of the original graph with the specified modifications.
    """
    # Create a copy of the graph to avoid modifying the original
    modified_graph = graph.copy()

    # Check if the target node is in the graph
    if target_node not in modified_graph:
        print(f"Node {target_node} not found in the graph.")
        return None

    # List of all incoming edges to the target node
    incoming_edges = list(modified_graph.in_edges(target_node))

    # Remove incoming edges
    modified_graph.remove_edges_from(incoming_edges)

    print(f"All incoming edges to {target_node} have been removed.")
    return modified_graph


class CausalInference:
    """
    A template class for performing causal inference by modifying a specific variable 'X'
    in the dataset and observing the changes in the model's predictions.
    """
    def __init__(self, dataset, model):
        """
        Initializes the CausalInference class with a dataset and a model.

        Args:
            dataset (dict): The dataset as a dictionary.
            model (callable): The predictive model.
            dag (nxDigraph object): The DAG for the underlying DGP
        """
        self.dataset = dataset
        self.model = model

    def set_variable(self, var_name='X', value=1):
        """
        Sets a specific variable in the dataset to a given value.

        Args:
            var_name (str): The name of the variable to modify.
            value (int or float): The value to set the variable to.

        Returns:
            dict: A new dataset dictionary with the modified variable.
        """
        modified_dataset = {key: np.copy(val) if key != var_name else np.full_like(val, value) for key, val in self.dataset.items()}
        return modified_dataset

    def predict(self, modified_dataset, batch_size=100):
        """
        Predicts outcomes using the model for a modified dataset.

        Args:
            modified_dataset (dict): The dataset dictionary with modifications.

        Returns:
            np.array: The model's predictions.
        """

        modified_dag = remove_incoming_edges(self.model.dag)
        modified_adj_matrix = nx.to_numpy_array(modified_dag)

        modified_masks = [torch.from_numpy(mask).float().to(torch.float64) for mask in
                         utils.expand_adjacency_matrix(self.model.neurons_per_layer[1:], modified_adj_matrix)]

        self.model.set_masks(modified_masks)

        num_samples = len(next(iter(modified_dataset.values())))  # Assuming all fields have the same number of entries
        predictions = []

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch = {key: val[start:end] for key, val in modified_dataset.items()}

            # Convert batch dictionary to appropriate tensor format if necessary
            batch = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v for k, v in
                     batch.items()}

            # Model prediction for the current batch
            batch_predictions = self.model(batch)

            # Convert predictions to numpy if necessary and store
            if isinstance(batch_predictions, torch.Tensor):
                batch_predictions = batch_predictions.detach().numpy()  # Convert to numpy array if tensor
            predictions.append(batch_predictions)

        return np.concatenate(predictions)

    def infer_causal_effect(self, var_name='X'):
        """
        Infers the causal effect of setting a variable 'X' to 0 and 1.

        Args:
            var_name (str): The name of the variable to test.

        Returns:
            tuple: A tuple containing two arrays (predictions when 'X' is 0, predictions when 'X' is 1).
        """
        # Set the variable 'X' to 0
        dataset_x0 = self.set_variable(var_name, 0)
        predictions_x0 = self.predict(dataset_x0)

        # Set the variable 'X' to 1
        dataset_x1 = self.set_variable(var_name, 1)
        predictions_x1 = self.predict(dataset_x1)

        return predictions_x0, predictions_x1




class CausalMetrics:
    """
    A class to compute causal inference metrics, specifically the error in the Average Treatment Effect (ATE)
    and the error in the Precision in Estimation of Heterogeneous Effect (PEHE).

    Attributes:
        y0_true (np.array): Ground truth outcomes when X=0.
        y1_true (np.array): Ground truth outcomes when X=1.
        y0_pred (np.array): Predicted outcomes when X=0.
        y1_pred (np.array): Predicted outcomes when X=1.
    """

    def __init__(self, y0_true, y1_true, y0_pred, y1_pred):
        """
        Initializes the CausalMetrics class with ground truth and predicted outcomes.

        Args:
            y0_true (np.array): Ground truth outcomes for Y0.
            y1_true (np.array): Ground truth outcomes for Y1.
            y0_pred (np.array): Predicted outcomes for Y0.
            y1_pred (np.array): Predicted outcomes for Y1.
        """
        self.y0_true = np.array(y0_true)
        self.y1_true = np.array(y1_true)
        self.y0_pred = np.array(y0_pred)
        self.y1_pred = np.array(y1_pred)

    def calculate_ate_error(self):
        """
        Calculates the error in the Average Treatment Effect (ATE).

        Returns:
            float: The absolute error in the ATE estimate.
        """
        ate_true = np.mean(self.y1_true - self.y0_true)
        ate_pred = np.mean(self.y1_pred - self.y0_pred)
        return np.abs(ate_true - ate_pred)

    def calculate_pehe_error(self):
        """
        Calculates the error in the Precision in Estimation of Heterogeneous Effects (PEHE).

        Returns:
            float: The square root of the mean squared error of the individual treatment effect estimates.
        """
        ite_true = self.y1_true - self.y0_true
        ite_pred = self.y1_pred - self.y0_pred
        pehe = np.sqrt(np.mean((ite_true - ite_pred) ** 2))
        return pehe

    def evaluate_metrics(self):
        """
        Evaluates both ATE and PEHE errors and returns them.

        Returns:
            dict: A dictionary containing the ATE and PEHE errors.
        """
        ate_error = self.calculate_ate_error()
        pehe_error = self.calculate_pehe_error()
        return {'ATE Error': ate_error, 'PEHE Error': pehe_error}
# get metrics
#input = Y0, Y1, preds0, preds1

# compute eATE

# compute ePEHE