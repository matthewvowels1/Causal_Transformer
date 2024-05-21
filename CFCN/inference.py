import torch
import numpy as np
import networkx as nx
from utils import find_element_in_list
from typing import Union


def predict(model, data, device):
    data = torch.from_numpy(data).float().to(device)
    return model.forward(data)


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

    modified_graph.remove_edges_from(incoming_edges)
    return modified_graph


class CausalInference():
    def __init__(self, model, device):
        '''
        Using the CaT, this model iterates through the causally-constrained attention according to an intervention and some data
        :param model: causal transformer CaT pytorch model
        '''
        self.model = model
        self.dag = self.model.nxdag
        self.ordering = self.model.causal_ordering
        self.device = device

    def forward(self, data, intervention_nodes_vals=None):
        '''
        This function iterates through the causally-constrained attention according to the dag and a desired intervention
        :param data is dataset (numpy) to accompany the desired interventions (necessary for the variables which are not downstream of the intervention nodes). Assumed ordering is topological.
        :param intervention_nodes_vals: dictionary of variable names as strings for intervention with corresponding intervention values
        :return: an updated dataset with new values including interventions and effects
        '''

        if intervention_nodes_vals is not None:
            D0 = data.copy()

            # modify the dataset with the desired intervention values
            for var_name in intervention_nodes_vals.keys():
                val = intervention_nodes_vals[var_name]
                index = find_element_in_list(list(self.dag.nodes()), var_name)
                D0[:, index] = val

            # find all descendants of intervention variables which are not in intervention set
            all_descs = []
            for var in intervention_nodes_vals.keys():
                all_descs.append(list(nx.descendants(self.dag, var)))
            all_descs = [item for sublist in all_descs for item in sublist]
            vars_to_update = set(all_descs) - set(intervention_nodes_vals.keys())
            # get corresponding column indexes
            indices_to_update = []
            for var_name in vars_to_update:
                indices_to_update.append(find_element_in_list(list(self.dag.nodes()), var_name)[0])

            # iterate through the dataset / predictions, updating the input dataset each time, where appropriate
            min_int_order = min([self.ordering[var] for var in intervention_nodes_vals.keys()])

            for i, var in enumerate(list(self.dag.nodes())):
                if self.ordering[var] >= min_int_order:  # start at the causal ordering at least as high as the lowest order of the intervention variable
                    # generate predictions , updating the input dataset each time
                    preds = predict(model=self.model, data=D0, device=self.device)[:, i]  # get prediction for each variable
                    if i in indices_to_update:
                        D0[:, i] = preds.detach().cpu().numpy()
        else:
            D0 = predict(model=self.model, data=data, device=self.device)

        return D0



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

        if self.y0_pred.shape != self.y0_true.shape:
            raise ValueError("y0_pred must have the same dimensionality as y0_true")
        if self.y1_pred.shape != self.y1_true.shape:
            raise ValueError("y1_pred must have the same dimensionality as y1_true")

    def calculate_ate(self):
        """
        Calculates the error in the Average Treatment Effect (ATE).

        Returns:
            float: The absolute error in the ATE estimate.
        """
        ate_pred = np.mean(self.y1_pred - self.y0_pred)
        return ate_pred

    def calculate_ate_error(self, true_ate=None):
        """
        Calculates the error in the Average Treatment Effect (ATE).

        Returns:
            float: The absolute error in the ATE estimate.
        """
        if true_ate is None:
            true_ate = np.mean(self.y1_true - self.y0_true)
        ate_pred = np.mean(self.y1_pred - self.y0_pred)
        return np.abs(true_ate - ate_pred)

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