
from trainer import predict
import networkx as nx

def find_element_in_list(input_list, target_string):
    matching_indices = []
    for index, element in enumerate(input_list):
        # Check if the element is equal to the target string
        if element == target_string:
            # If it matches, add the index to the matching_indices list
            matching_indices.append(index)
    return matching_indices


class CausalInference():
    def __init__(self, model, device):
        '''
        Using the CaT, this model iterates through the causally-constrained attention according to an intervention and some data
        :param model: causal transformer CaT pytorch model
        '''
        self.model = model
        self.dag = self.model.nxdag
        self.ordering = self.model.ordering
        self.device = device

    def forward(self, data, intervention_nodes_vals):
        '''
        This function iterates through the causally-constrained attention according to the dag and a desired intervention
        :param data is dataset (torch) to accompany the desired interventions (necessary for the variables which are not downstream of the intervention nodes). Assumed ordering is topological.
        :param intervention_nodes_vals: dictionary of variable names as strings for intervention with corresponding intervention values
        :return: an updated dataset with new values including interventions and effects
        '''

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

        return D0




