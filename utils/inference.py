import networkx as nx
import torch
import warnings

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
    def __init__(self, model, device, mask=None):
        '''
        Using the CaT, this model iterates through the causally-constrained attention according to an intervention and some data
        :param model: causal transformer CaT pytorch model
        '''
        self.model = model
        self.dag = self.model.nxdag
        self.ordering = self.model.causal_ordering
        self.device = device
        self.mask = mask

        if self.mask is None:
            warnings.warn(
                "No mask has been specified. If padding has been used, "
                "the absence of a mask may lead to incorrect results.",
                UserWarning
            )

    def get(self, data, var_name):
        index = list(self.dag.nodes()).index(var_name)
        return data[:, index]


    def forward(self, data, intervention_nodes_vals=None):
        '''
        This function iterates through the causally-constrained attention according to the dag and a desired intervention
        :param data is dataset (numpy) to accompany the desired interventions (necessary for the variables which are not downstream of the intervention nodes). Assumed ordering is topological.
        :param intervention_nodes_vals: dictionary of variable names as strings for intervention with corresponding intervention values
        :return: an updated dataset with new values including interventions and effects
        '''

        if intervention_nodes_vals is not None:
            Dprime = data.copy()

            # modify the dataset with the desired intervention values
            for var_name in intervention_nodes_vals.keys():
                val = intervention_nodes_vals[var_name]
                index = list(self.dag.nodes()).index(var_name)
                Dprime[:, index] = val

            if self.mask is not None:
                Dprime = Dprime * self.mask


            # find all descendants of intervention variables which are not in intervention set
            all_descs = []
            for var in intervention_nodes_vals.keys():
                descs = list(nx.descendants(self.dag, var))
                all_descs.append(descs)
            all_descs = [item for sublist in all_descs for item in sublist]
            vars_to_update = set(all_descs) - set(intervention_nodes_vals.keys())

            # iterate through the dataset / predictions, updating the input dataset each time, where appropriate
            for i, var in enumerate(list(self.dag.nodes())):
                if var in vars_to_update:
                    preds = predict(model=self.model, data=Dprime, device=self.device)[:,i]
                    Dprime[:, i] = preds.detach().cpu().numpy()
                    if self.mask is not None:  # apply the mask to the predictions
                        Dprime = Dprime * self.mask
        else:
            Dprime = predict(model=self.model, data=data, device=self.device).detach().cpu().numpy()
            if self.mask is not None:  # apply the mask to the predictions
                Dprime = Dprime * self.mask

        return Dprime
