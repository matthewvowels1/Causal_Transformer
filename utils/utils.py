import numpy as np
import networkx as nx
from typing import List, Tuple


def pad_vectors_with_mask(vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pads list of input arrays according to the length of the highest-d vector.
    Returns array of padded vectors and corresponding mask.
    """
    # T is the length of the list (i.e., number of vars)
    T = len(vectors)

    # N is the number of rows (samples) in each vector (assumed consistent across all)
    N = vectors[0].shape[0]

    # Find the maximum number of columns (C) across all arrays
    max_c = max(vec.shape[1] for vec in vectors)  # Maximum C (number of features)

    # Initialize the padded array for vectors (N, T, C) and the mask (T, C)
    padded_vectors = np.zeros((N, T, max_c), dtype=np.float32)  # (N, T, C)
    mask = np.zeros((T, max_c),  dtype=np.float32)  # (T, C)

    # Pad vectors and create the mask
    for t, vec in enumerate(vectors):
        c = vec.shape[1]  # Number of columns for this particular time step
        padded_vectors[:, t, :c] = vec  # Copy the original vector values into padded array
        mask[t, :c] = 1.0  # Mark the valid positions in the mask

    return padded_vectors, mask



def assert_neuron_layers(layers, input_size):
    # Assert that the smallest number of neurons is never lower than input_size
    assert min(layers) >= input_size, "The smallest layer size must be at least the input size."

    # Assert that subsequent layers change either not at all, or by a factor of 2
    for i in range(1, len(layers)):
        previous_layer, current_layer = layers[i - 1], layers[i]
        is_same = current_layer == previous_layer
        is_double = current_layer == 2 * previous_layer
        is_half = current_layer == previous_layer / 2
        assert is_same or is_double or is_half, "Layer sizes must stay the same or change by a factor of 2."

    # Assert that the first layer is a multiple of 2 of the input_size
    assert layers[0] % input_size == 0 and ((layers[0] // input_size) & ((layers[0] // input_size) - 1)) == 0, \
        "The first layer must be a multiple of 2 of the input size."


def expand_matrix(original_matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Expands a matrix to a specified target shape by repeating its rows and columns.

    Parameters:
    - original_matrix (np.ndarray): The matrix to be expanded.
    - target_shape (Tuple[int, int]): A tuple specifying the desired number of rows and columns in the expanded matrix.

    Returns:
    - np.ndarray: The expanded matrix with the specified number of rows and columns.
    """
    original_rows, original_columns = original_matrix.shape
    target_rows, target_columns = target_shape

    # Calculate the number of times to repeat each row and column
    row_repeats = target_rows // original_rows
    column_repeats = target_columns // original_columns

    # Repeat rows and columns to expand the matrix
    expanded_matrix = np.repeat(np.repeat(original_matrix, row_repeats, axis=0), column_repeats, axis=1)

    return expanded_matrix


def calculate_weight_shape(initial_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculates the shape of a weight matrix transitioning from an initial shape to a target shape.

    Parameters:
    - initial_shape (Tuple[int, int]): The shape of the matrix at the starting layer.
    - target_shape (Tuple[int, int]): The desired shape of the matrix at the target layer.

    Returns:
    - Tuple[int, int]: A tuple representing the shape (columns, rows) of the weight matrix.
    """
    initial_rows, initial_columns = initial_shape
    target_rows, target_columns = target_shape

    # Calculate the number of rows and columns in the weight matrices
    weight_rows = target_columns
    weight_columns = initial_columns

    return (weight_columns, weight_rows)


def expand_adjacency_matrix(neurons_per_layer: List[int], adjacency_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Expands an adjacency matrix according to the number of neurons per layer, adjusting its shape for each layer.

    Parameters:
    - neurons_per_layer (List[int]): A list of integers where each integer represents the number of neurons in that layer.
    - adjacency_matrix (np.ndarray): The initial adjacency matrix.

    Returns:
    - List[np.ndarray]: A list of expanded adjacency matrices for each layer.
    """
    # Calculate target shapes for each layer based on the number of neurons
    target_shapes = [(1, neurons) for neurons in neurons_per_layer]

    # Initialize the expanded adjacency matrix list
    expanded_matrices = []

    initial_shape = (1, adjacency_matrix.shape[1])
    identity_matrix = np.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[1], dtype=np.float32)

    for i, target_shape in enumerate(target_shapes):
        # Calculate the weight shape for the current layer
        weight_shape = calculate_weight_shape(initial_shape, target_shape)
        initial_shape = target_shape
        # Expand the adjacency matrix to match the weight shape
        expanded_matrix = expand_matrix(adjacency_matrix, weight_shape)

        if i >= 1:
            expanded_identity = expand_matrix(identity_matrix, weight_shape)
            expanded_matrix += expanded_identity

        # Append the expanded matrix to the list
        expanded_matrices.append(expanded_matrix)

    return expanded_matrices


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
