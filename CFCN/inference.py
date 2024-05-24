import torch
import numpy as np
import networkx as nx
from utils import find_element_in_list
from typing import Union


def predict(model, data, device):
    model.eval()
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
        self.ordering = self.model.causal_ordering
        self.dag = self.model.dag
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


