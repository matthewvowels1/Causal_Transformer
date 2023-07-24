

def find_element_in_list(input_list, target_string):
    matching_indices = []
    for index, element in enumerate(input_list):
        # Check if the element is equal to the target string
        if element == target_string:
            # If it matches, add the index to the matching_indices list
            matching_indices.append(index)
    return matching_indices


class CausalInference():
    def __init__(self, model):
        '''
        Using the CaT, this model iterates through the causally-constrained attention according to an intervention and some data
        :param model: causal transformer CaT pytorch model
        '''
        self.model = model
        self.dag = self.model.nxdag
        self.ordering = self.model.ordering

    def forward(self, data, intervention_nodes, intervention_vals):
        '''
        This function iterates through the causally-constrained attention according to the dag and a desired intervention
        :param data is dataset (torch) to accompany the desired interventions (necessary for the variables which are not downstream of the intervention nodes). Assumed ordering is topological.
        :param intervention_nodes: list of variable names as strings for intervention
        :param intervention_vals: np array of intervention values for each intervention (assumes same ordering as intervention_nodes)
        :return: an updated dataset with new values including interventions and effects
        '''

        assert len(intervention_nodes) == len(
            intervention_vals), 'Both intervention node names and values must be provided, and their lengths must match.'


        # modify the dataset with the desired intervention values
        for var_name, var_val in zip(intervention_nodes, intervention_vals):
            index = find_element_in_list(list(self.dag.nodes()), var_name)
            data[:, index] = var_val

        # use the ordering to work out where to start for the iterations
        # for all variables

        # recursively iterate through the dag to generate results which respond to the relevant interventions

        if intervention_vals is not None:
            assert len(intervention_nodes) == intervention_vals, 'Number of intervention nodes must be the same as the number of intervention values.'
        raise NotImplementedError("En route...!")




