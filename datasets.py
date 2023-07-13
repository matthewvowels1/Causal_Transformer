import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def sigm(x):
    return 1/(1 + np.exp(-x))

def inv_sigm(x):
    return np.log(x/(1-x))


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


def generate_data(N, seed, dataset):
    np.random.seed(seed=seed)
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
        M = m1 * X + m0 * (1 - X)

        # outcomes:
        Y1 = np.random.binomial(1, sigm(np.exp(-1 + m1 - 0.1 * z1 + 0.35 * z2 +
                                               0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2)),
                                (N, 1))
        Y0 = np.random.binomial(1,
                                sigm(-1 + m0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2),
                                (N, 1))
        Y = Y1 * X + Y0 * (1 - X)

        # colliders:
        C = 0.6 * Y + 0.4 * X + 0.4 * np.random.randn(N, 1)

        DAGnx = nx.DiGraph()
        DAGnx.add_edges_from([('Z1', 'Z5'), ('Z2', 'X'), ('Z3', 'X'), ('Z4', 'X'), ('Z5', 'X'),
                              ('Z2', 'Y'), ('Z3', 'Y'), ('Z4', 'Y'), ('Z5', 'Y'),
                              ('R1', 'Y'), ('R2', 'Y'), ('M', 'Y'),
                              ('I1', 'X'), ('I2', 'X'), ('X', 'M'), ('X', 'Y'), ('X', 'C'),
                              ('Y', 'C')])

        DAGnx = reorder_dag(dag=DAGnx)  # topologically sorted dag
        var_names = list(DAGnx.nodes())

        all_data_dict = {'Z1': z1, 'Z2': z2, 'Z3': z3, 'Z4': z4, 'Z5': z5, 'X': X, 'M': M, 'I1': i1,
                         'I2': i2, 'R1': r1, 'R2': r2, 'Y': Y, 'C': C}

        all_data = (np.stack([all_data_dict[key] for key in var_names], axis=-1))[:, 0, :]

        plt.title('general')
        pos = graphviz_layout(DAGnx, prog='dot')
        nx.draw_networkx(DAGnx, pos, with_labels=True, arrows=True)
        plt.savefig('general_graph.png')

    return all_data, DAGnx, var_names, Y0, Y1