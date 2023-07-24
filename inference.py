


class CausalInference():
	def __init__(self, model):
		'''
		Using the CaT, this model iterates through the causally-constrained attention according to an intervention and some data
		:param model: causal transformer pytorch model
		'''
		self.model = model
		self.dag = self.model.dag

	def forward(self, data, intervention_nodes=None, intervention_vals=None):
		'''
		This function iterates through the causally-constrained attention according to the dag and a desired intervention
		:param data is dataset (torch) to accompany the desired interventions (necessary for the variables which are not downstream of the intervention nodes)
		:param intervention_nodes: list of variable names as strings for intervention
		:param intervention_vals: np array of intervention values for each intervention (assumes same ordering as intervention_nodes)
		:return: an updated dataset with new values including interventions and effects
		'''

		# modify the dataset with the desired intervention values

		# use the dag to determine where to start in the causal ordering

		# recursively iterate through the dag to generate results which respond to the relevant interventions

		if intervention_vals is not None:
			assert len(intervention_nodes) == intervention_vals, 'Number of intervention nodes must be the same as the number of intervention values.'
		raise NotImplementedError("En route...!")




