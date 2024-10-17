import networkx as nx
import numpy as np
import torch
import warnings


class CausalInference:
    def __init__(self, dag, mask=None):
        '''
        This class implements utilities for handling dag datasets, and performing causal inference.
        :param dag: DagNX directed acyclic graph to use.
        :param mask: Optional mask to apply to the dataset.
        '''
        self.dag = dag
        self.mask = mask

        if self.mask is None:
            warnings.warn(
                "No mask has been specified. If padding has been used, "
                "the absence of a mask may lead to incorrect results.",
                UserWarning
            )

    def remove(self, data, var_name):
        index = list(self.dag.nodes()).index(var_name)
        if isinstance(data, np.ndarray):
            # If data is a NumPy array, use np.delete
            return np.delete(data, index, axis=1)
        elif isinstance(data, torch.Tensor):
            return data[:, torch.arange(data.size(1)) != index]
        else:
            raise TypeError("Input data must be a NumPy array or a PyTorch tensor.")

    def get(self, data, var_name):
        index = list(self.dag.nodes()).index(var_name)
        return data[:, index]

    def apply_intervention(self, data, intervention_nodes_vals: dict):
        data_copy = data.copy()
        if intervention_nodes_vals:
            for var_name, val in intervention_nodes_vals.items():
                index = list(self.dag.nodes()).index(var_name)
                data_copy[:, index] = val
            if self.mask is not None:
                data_copy = data_copy * self.mask
        return data_copy

    def forward(self, data, model, intervention_nodes_vals=None):
        '''
        This function iterates through the causally-constrained attention according to the dag and a desired intervention
        :param data: is dataset (numpy) to accompany the desired interventions (necessary for the variables which are not downstream of the intervention nodes). Assumed ordering is topological.
        :param model: is a nn.Module model to use for the forward method
        :param intervention_nodes_vals: dictionary of variable names as strings for intervention with corresponding intervention values
        :return: an updated dataset with new values including interventions and effects
        '''

        model.eval()

        if intervention_nodes_vals:
            Dprime = self.apply_intervention(data, intervention_nodes_vals)

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
                    preds = model.forward(torch.from_numpy(Dprime).float().to(model.device))
                    Dprime[:, i] = preds[:, i].detach().cpu().numpy()
                    if self.mask is not None:  # apply the mask to the predictions
                        Dprime = Dprime * self.mask
        else:
            preds = model.forward(torch.from_numpy(data).float().to(model.device))
            Dprime = preds.detach().cpu().numpy()
            if self.mask is not None:  # apply the mask to the predictions
                Dprime = Dprime * self.mask

        return Dprime
