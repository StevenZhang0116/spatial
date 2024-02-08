import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import torch
import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D

import multiprocessing as mp

def line_line_distance(seg1, seg2):
    """
    Calculate the shortest distance between two line segments in 3D.

    Parameters:
    seg1: Tuple of numpy arrays representing the start and end points of the first line segment.
    seg2: Tuple of numpy arrays representing the start and end points of the second line segment.

    Reference: 
    https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf
    https://stackoverflow.com/questions/38637542/finding-the-shortest-distance-between-two-3d-line-segments
    """
    p1, q1 = seg1
    p2, q2 = seg2

    # Direction vectors for the line segments
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    # Squared magnitudes of the direction vectors
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    f = a * c - b * b

    # Check if the lines are parallel
    if f != 0:
        s = (b * e - c * d) / f
        t = (a * e - b * d) / f
    else:
        # Lines are parallel, use arbitrary point on seg1
        s = 0
        t = d / b if b > 0 else 0

    # Cap s and t between 0 and 1 to stay within the line segments
    s = max(0, min(1, s))
    t = max(0, min(1, t))

    # Calculate the closest points
    closest_point_on_seg1 = p1 + d1 * s
    closest_point_on_seg2 = p2 + d2 * t

    # Return the distance between the closest points
    return np.linalg.norm(closest_point_on_seg1 - closest_point_on_seg2)

def stick_pairwise_distances(sticks):
    """
    Calculate pairwise (minimum distance) for stick structure
    """
    num = sticks.shape[0]
    W = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            stick1 = sticks[i]
            stick2 = sticks[j]
            W[i,j] = line_line_distance(stick1, stick2)
    return W

def point_pairwise_distances(sticks):
    """
    Trivial 3D Euclidean distance calculation: only useful when sticks are reduced to points; for sanity check
    """
    num = sticks.shape[0]
    W = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            # the start and end point will be same in that case
            pt1 = sticks[i][0]
            pt2 = sticks[j][0]
            W[i,j] = np.linalg.norm(pt2 - pt1)
    return W


def generate_random_sticks(n, length, x_range, y_range, z_range):
    """
    Generate a set of random sticks (line segments) in 3D space.
    Each stick is represented by its start and end points.
    """
    sticks = np.zeros((n, 2, 3))

    for i in range(n):
        # Generate a random start point within the specified ranges
        start = np.array([np.random.uniform(x_range[0], x_range[1]),
                          np.random.uniform(y_range[0], y_range[1]),
                          np.random.uniform(z_range[0], z_range[1])])
        
        # Generate a random direction vector and normalize it
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        # Scale the direction vector to the specified length and calculate the end point
        end = start + direction * length

        # Store the start and end points of the stick
        sticks[i] = [start, end]

    return sticks

def generate_equal_length_stick_structure(mean_range, short_std, long_std):
    """
    Generate points for a stick-like structure with a random orientation and equal length using multivariate normal distribution.
    """
    # Creating a random rotation matrix for the covariance
    random_rotation = scipy.linalg.orth(np.random.rand(3, 3))

    # Fixed covariance along one axis (to create the stick-like shape), then rotating it
    # The variance along the major axis is set to a fixed value to maintain equal lengths
    base_cov = np.diag([short_std, short_std, long_std])  # Smaller variance in X and Y to keep the stick thin
    rotated_cov = random_rotation @ base_cov @ random_rotation.T

    return [multivariate_normal(mean_range, rotated_cov), mean_range, rotated_cov]

def generate_grid_points(num_per_dimension, x_range, y_range=None, z_range=None):
    """
    Generate points based on teh grid points
    """
    x_points = np.linspace(x_range[0], x_range[1], num_per_dimension)
    
    # 3D
    if z_range is not None and y_range is not None:
        y_points = np.linspace(y_range[0], y_range[1], num_per_dimension)
        z_points = np.linspace(z_range[0], z_range[1], num_per_dimension)

        x_grid, y_grid, z_grid = np.meshgrid(x_points, y_points, z_points, indexing='ij')
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
    # 2D
    elif z_range is None and y_range is not None:
        y_points = np.linspace(y_range[0], y_range[1], num_per_dimension)

        x_grid, y_grid = np.meshgrid(x_points, y_points, indexing='ij')
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    # 1D
    else:
        # provide interface for [generate_graph_from_adj_matrix] 
        grid_points = [[pt, 0] for pt in x_points]

    pos_dictionary = {}
    for i in range(len(grid_points)):
        sample = grid_points[i]
        pos_dictionary[i] = [sample[l] for l in range(len(sample))]
    
    return pos_dictionary

def generate_graph_from_adj_matrix(adj_matrix, positions=None):
    """
    Generate networkx object based on adjency matrix
    """
    G = nx.Graph()
    
    if not isinstance(adj_matrix, np.ndarray):
        adj_matrix = np.array(adj_matrix)

    # ADD NODE BEFORE ADD EDGES
    if positions is not None: 
        for node, pos in positions.items():
            G.add_node(node, pos=pos)

    if positions is None: 
        G.add_nodes_from(range(adj_matrix.shape[0]))  
    
    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):  
            if adj_matrix[i, j] == 1:  
                G.add_edge(i, j)
    
    return G
    

def generate_adjacency_matrix_uniform(n, dimensionality):
    """
    Generate corresponding adjacency matrix based on the uniform grids
    """

    total_points = n ** dimensionality
    print(f"total_points: {total_points}")
    adj_matrix = np.zeros((total_points, total_points))

    if dimensionality == 1:
        for i in range(n):
            # Right neighbor
            if i + 1 < n:
                adj_matrix[i, i + 1] = 1
            # Left neighbor
            if i - 1 >= 0:
                adj_matrix[i, i - 1] = 1
    
    elif dimensionality == 2:
        for i in range(n):
            for j in range(n):
                index = i * n + j
                # Right
                if j + 1 < n:
                    adj_matrix[index, i * n + (j + 1)] = 1
                # Left
                if j - 1 >= 0:
                    adj_matrix[index, i * n + (j - 1)] = 1
                # Up
                if i + 1 < n:
                    adj_matrix[index, (i + 1) * n + j] = 1
                # Down
                if i - 1 >= 0:
                    adj_matrix[index, (i - 1) * n + j] = 1
                
    elif dimensionality == 3:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    index = i * n * n + j * n + k
                    # Along x-axis
                    if i + 1 < n:
                        adj_matrix[index, (i + 1) * n * n + j * n + k] = 1
                    if i - 1 >= 0:
                        adj_matrix[index, (i - 1) * n * n + j * n + k] = 1
                    # Along y-axis
                    if j + 1 < n:
                        adj_matrix[index, i * n * n + (j + 1) * n + k] = 1
                    if j - 1 >= 0:
                        adj_matrix[index, i * n * n + (j - 1) * n + k] = 1
                    # Along z-axis
                    if k + 1 < n:
                        adj_matrix[index, i * n * n + j * n + (k + 1)] = 1
                    if k - 1 >= 0:
                        adj_matrix[index, i * n * n + j * n + (k - 1)] = 1

    return adj_matrix.astype(int)

def generate_fixed_edges_graph(n, m):
    """
    Gererate Erdos-Renyi-like but guarantee the total number of connections in single trail 
    """
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = min(m, len(all_possible_edges))
    selected_edges = random.sample(all_possible_edges, int(m))    
    G = nx.Graph()
    G.add_nodes_from(range(int(n)))  
    G.add_edges_from(selected_edges)
    return G

def count_chains_of_any_length(graph, chain_length):
    """
    Count chain with length n -> Nodes and lengths are different (Eulerian), otherwise just use W^n
    """
    chain_count = 0
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                # Adjusting the cutoff to be exactly the chain_length
                for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=chain_length):
                    if len(path) == chain_length + 1:  # Ensures the path is exactly the desired length
                        chain_count += 1
    return chain_count // 2

def flex_plot(fig, G, pos):
    """
    Similar to nx.draw() but accept the position information to be 3D (instead of maximum 2D)
    Have plot configuration, including plt.figrue(), plt.show() etc., outside the function. 
    """
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*[pos[node] for node in G.nodes()])
    ax.scatter(xs, ys, zs)

    for edge in G.edges():
        x_coords, y_coords, z_coords = zip(*[pos[node] for node in edge])
        ax.plot(x_coords, y_coords, z_coords, "b-") 

    return fig 

