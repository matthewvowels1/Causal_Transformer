## datasets.py
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def sigm(x):
	return 1 / (1 + np.exp(-x))


def inv_sigm(x):
	return np.log(x / (1 - x))



def reorder_dag(dag):
	'''Takes a networkx digraph object and returns a topologically sorted graph.'''

	assert nx.is_directed_acyclic_graph(dag), 'Graph needs to be acyclic.'

	old_ordering = list(dag.nodes())  # get old ordering of nodes
	adj_mat = nx.to_numpy_array(dag)  # get adjacency matrix of old graph

	index_old = {v: i for i, v in enumerate(old_ordering)}
	topological_ordering = list(nx.topological_sort(dag))  # get ideal topological ordering of nodes

	permutation_vector = [index_old[v] for v in topological_ordering]  # get required permutation of old ordering

	reordered_adj = adj_mat[np.ix_(permutation_vector, permutation_vector)]  # reorder old adj. mat

	dag = nx.from_numpy_array(reordered_adj, create_using=nx.DiGraph)  # overwrite original dag

	mapping = dict(zip(dag, topological_ordering))  # assign node names
	dag = nx.relabel_nodes(dag, mapping)

	return dag


def get_full_ordering(DAG):
	''' Note that the input DAG MUST be topologically sorted <before> using this function'''
	ordering_info = {}
	current_level = 0
	var_names = list(DAG.nodes)

	for i, var_name in enumerate(var_names):

		if i == 0:  # if first in list
			ordering_info[var_name] = 0

		else:
			# check if any parents
			parent_list = list(DAG.predecessors(var_name))

			# if no parents ()
			if len(parent_list) == 0:
				ordering_info[var_name] = current_level

			elif len(parent_list) >= 1:  # if some parents, find most downstream parent and add 1 to ordering
				for parent_var in parent_list:
					parent_var_order = ordering_info[parent_var]
					ordering_info[var_name] = parent_var_order + 1

	return ordering_info


def generate_data(N, seed, dataset, standardize=1):
	'''
	:param N: required sample size
	:param seed: random seed
	:param dataset: which dataset to use (only 'general' currently implemented)
	:return: data (DxN), a DAG (networkX), a list of variable names, Y0 and Y1 (vectors of counterfactual outcomes), a list of variable types

	Note that the data, the DAG, and the variable names are topologically sorted.
	'''

	np.random.seed(seed=seed)
	DAGnx = nx.DiGraph()
	if dataset == 'general':
		# confounders
		z1 = np.random.binomial(1, 0.5, (N, 1))
		z2 = np.random.binomial(1, 0.65, (N, 1))
		z3 = np.round(np.random.uniform(0, 4, (N, 1)), 0)
		z4 = np.round(np.random.uniform(0, 5, (N, 1)), 0)
		uz5 = np.random.randn(N, 1)
		z5 = 0.2 * z1 + uz5

		# risk vars:
		r1 = np.random.randn(N, 1)
		r2 = np.random.randn(N, 1)

		# instrumental vars:
		i1 = np.random.randn(N, 1)
		i2 = np.random.randn(N, 1)

		# treatment:
		ux = np.random.randn(N, 1)
		xp = sigm(-5 + 0.05 * z2 + 0.25 * z3 + 0.6 * z4 + 0.4 * z2 * z4 + 0.15 * z5 + 0.1 * i1 + 0.15 * i2 + 0.1 * ux)
		X = np.random.binomial(1, xp, (N, 1))

		# mediator:
		Um = np.random.randn(N, 1)
		m1 = 0.8 + 0.15 * Um
		m0 = 0.15 * Um

		if standardize:
			z1 = (z1 - z1.mean()) / z1.std()
			z2 = (z2 - z2.mean()) / z2.std()
			z3 = (z3 - z3.mean()) / z3.std()
			z4 = (z4 - z4.mean()) / z4.std()
			z5 = (z5 - z5.mean()) / z5.std()

			r1 = (r1 - r1.mean()) / r1.std()
			r2 = (r2 - r2.mean()) / r2.std()

			i1 = (i1 - i1.mean()) / i1.std()
			i2 = (i2 - i2.mean()) / i2.std()

			X = (X - X.mean()) / X.std()

			m1 = (m1 - m1.mean()) / m1.std()
			m0 = (m0 - m0.mean()) / m0.std()

		M = m1 * X + m0 * (1 - X)

		if standardize:
			M = (M - M.mean()) / M.std()
		# outcomes:
		Y1 = np.random.binomial(1, sigm(np.exp(-1 + m1 - 0.1 * z1 + 0.35 * z2 +
		                                       0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2)),
		                        (N, 1))
		Y0 = np.random.binomial(1,
		                        sigm(-1 + m0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2),
		                        (N, 1))
		Y = Y1 * X + Y0 * (1 - X)

		if standardize:
			Y = (Y - Y.mean()) / Y.std()

		# colliders:
		C = 0.6 * Y + 0.4 * X + 0.4 * np.random.randn(N, 1)

		all_data_dict = {'Z1': z1, 'Z2': z2, 'Z3': z3, 'Z4': z4, 'Z5': z5, 'X': X, 'M': M, 'I1': i1,
		                 'I2': i2, 'R1': r1, 'R2': r2, 'Y': Y, 'C': C}

		# types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
		var_types = {'Z1': 'cont', 'Z2': 'cont', 'Z3': 'cont', 'Z4': 'cont', 'Z5': 'cont', 'X': 'bin', 'M': 'cont',
		             'I1': 'cont',
		             'I2': 'cont', 'R1': 'cont', 'R2': 'cont', 'Y': 'bin', 'C': 'cont'}

		DAGnx.add_edges_from([('Z1', 'Z5'), ('Z2', 'X'), ('Z3', 'X'), ('Z4', 'X'), ('Z5', 'X'),
		                      ('Z2', 'Y'), ('Z3', 'Y'), ('Z4', 'Y'), ('Z5', 'Y'),
		                      ('R1', 'Y'), ('R2', 'Y'), ('M', 'Y'),
		                      ('I1', 'X'), ('I2', 'X'), ('X', 'M'), ('X', 'Y'), ('X', 'C'),
		                      ('Y', 'C')])


	elif dataset == 'simple_test':
		ux1 = np.random.randn(N, 1)
		ux2 = np.random.randn(N, 1)
		uy = np.random.randn(N, 1)

		X = ux1
		X2 = ux2
		Y = 0.6 * X - 0.5 * X2  # + uy

		if standardize:
			X = (X - X.mean()) / X.std()
			X2 = (X2 - X2.mean()) / X2.std()
			Y = (Y - Y.mean()) / Y.std()

		# outcomes:
		Y1 = 0.6 - 0.5 * X2  # + uy
		Y0 = -0.5 * X2  # + uy

		X_1 = np.full((len(Y1) // 2, 1), 1)  # TODO: very simple test dataset  (to be removed)
		X2_1 = np.full((len(Y1) // 2, 1), 2)
		Y_1 = np.full((len(Y1) // 2, 1), 3)

		X_2 = np.full((len(Y1) // 2, 1), 2)  # TODO: very simple test dataset  (to be removed)
		X2_2 = np.full((len(Y1) // 2, 1), 4)
		Y_2 = np.full((len(Y1) // 2, 1), 6)

		X = np.concatenate((X_1, X_2), 0)
		X2 = np.concatenate((X2_1, X2_2), 0)
		Y = np.concatenate((Y_1, Y_2), 0)

		all_data_dict = {'X': X, 'X2': X2, 'Y': Y}

		# types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
		var_types = {'X': 'cont', 'X2': 'cont', 'Y': 'cont'}

		DAGnx.add_edges_from([('X', 'Y'), ('X2', 'Y')])

	elif dataset == 'simple_test_v2':

		Ux = np.random.randn(N)
		X = Ux
		Ub = 0.1 * np.random.randn(N)
		B = 2 * X + Ub
		Uc = 0.1 * np.random.randn(N)
		C = 2 * B + Uc
		Uy = 0.1 * np.random.randn(N)
		Y = 2 * X + 2 * C + Uy

		B0 = Ub
		B1 = 2 + Ub

		C0 = 2 * B0 + Uc
		C1 = 2 * B1 + Uc

		Y0 = 2 * C0 + 0.1 * np.random.randn(N)
		Y1 = 2 + 2 * C1 + 0.1 * np.random.randn(N)

		all_data_dict = {'X': X, 'B': B, 'C': C, 'Y': Y}

		# types can be 'cat' (categorical) 'cont' (continuous) or 'bin' (binary)
		var_types = {'X': 'cont', 'B': 'cont', 'C': 'cont', 'Y': 'cont'}

		#         DAGnx.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'Y')])
		DAGnx.add_edges_from([('X', 'B'), ('B', 'C'), ('C', 'Y'), ('X', 'Y')])

	else:
		raise NotImplementedError

	DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
	causal_ordering = get_full_ordering(DAGnx)
	var_names = list(DAGnx.nodes())  # topologically ordered list of variables
	all_data = np.stack([all_data_dict[key] for key in var_names], axis=1)

	plt.title('general')
	pos = graphviz_layout(DAGnx, prog='dot')
	nx.draw_networkx(DAGnx, pos, with_labels=True, arrows=True)
	plt.savefig(f'{dataset}_graph.png')


	return all_data, DAGnx, var_types, var_names, causal_ordering, Y0, Y1