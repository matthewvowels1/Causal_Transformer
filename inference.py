


class CausalInference():
	def __init__(self, model):
		'''
		Using the CaT, this model iterates through the causally-constrained attention according to an intervention and some data
		:param model: causal transformer pytorch model
		'''
		self.model = model
		self.dag = self.model.dag

	def forward(self, data, intervention_node=None):
		'''
		This function iterates through the causally-constrained attention according to the dag and a desired intervention
		:param intervention_node:
		:return:
		'''
		raise NotImplementedError("En route...!")




