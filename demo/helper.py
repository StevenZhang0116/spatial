import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kneed import KneeLocator
import seaborn as sns
import sys 
sys.path.append("../")
from analysis import *
import csv

def plot_single_neuron(soma, pre_loc=np.array([]), post_loc=np.array([])):
    """
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    soma = soma[0]
    ax.scatter(soma[0], soma[2], soma[1], label="soma", color="red", s=100) 

    pre_loc = np.array(pre_loc)
    post_loc = np.array(post_loc)
    if len(pre_loc) > 0:
        ax.scatter(pre_loc[:, 0], pre_loc[:, 2], pre_loc[:, 1], label="pre", color="blue")
    if len(post_loc) > 0:
        ax.scatter(post_loc[:, 0], post_loc[:, 2], post_loc[:, 1], label="post", color="green")

    plt.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.set_title(f'Neuron: {np.round(soma[0], 2), np.round(soma[2], 2), np.round(soma[1], 2)}')
    plt.show()

def plot_double_neuron(soma1, soma2, loc1, loc2):
    """
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    soma1 = soma1[0]
    soma2 = soma2[0]
    ax.scatter(soma1[0], soma1[2], soma1[1], label="soma1", color="red", s=100) 
    ax.scatter(soma2[0], soma2[2], soma2[1], label="soma2", color="red", s=100) 
    ax.scatter(loc1[:, 0], loc1[:, 2], loc1[:, 1], label="loc1", color="blue")
    ax.scatter(loc2[:, 0], loc2[:, 2], loc2[:, 1], label="loc2", color="green")

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    plt.show()


def extract_neuron_synaptic_info(cell_id, flag, cell_table, synapse_table):
    """
    Extract synaptic information about a neuron based on a flag indicating
    whether to retrieve presynaptic or postsynaptic data.
    """
    # Validate flag value
    if flag not in ['pre', 'post']:
        raise ValueError("Flag must be 'pre' or 'post'.")

    # Extract basic neuron information
    soma_loc = ((cell_table.loc[cell_table['pt_root_id'] == cell_id, "pt_position"].values)[0]).reshape(1, 3)
    neuron_type = (cell_table.loc[cell_table['pt_root_id'] == cell_id, "cell_type"].values)[0]

    # Use flag to dynamically adjust column names for extracting synaptic information
    serve_as_id = synapse_table.loc[synapse_table[f'{flag}_pt_root_id'] == cell_id, 'id'].values
    serve_as_loc = synapse_table.loc[synapse_table[f'{flag}_pt_root_id'] == cell_id, 'ctr_pt_position'].values
    serve_as_syn_size = synapse_table.loc[synapse_table[f'{flag}_pt_root_id'] == cell_id, 'size'].values
    
    # Prepare synaptic locations
    serve_as_loc = np.array([i for i in serve_as_loc])

    # it is possible, especially for postsynaptic location, that no synapse is found
    if len(serve_as_loc) > 0:
        serve_as_rela_loc = serve_as_loc - soma_loc
        serve_as_rela_loc_norm = np.linalg.norm(serve_as_rela_loc, axis=1)
    else:
        serve_as_rela_loc = []
        serve_as_rela_loc_norm = []

    return [soma_loc, neuron_type, serve_as_loc, serve_as_syn_size, serve_as_rela_loc_norm]

def gaussian_decay(x, A, mu, sigma):
    """
    Calculate the Gaussian decay for a given x with specified amplitude, mean, and standard deviation.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def check_EI(type):
    """
    Check neuron type if excititory or inhibitory
    """
    cc = 0
    if type[-1] == 'C':
        cc += 1
    return cc

def create_graph(X, epsilon):
    """
    Creates a graph from 3D coordinates with edges based on a distance threshold
    and calculates the minimum distance between the last node and all other nodes.
    """
    G = nx.Graph()
    
    for i, coord in enumerate(X):
        G.add_node(i, pos=coord)
    
    for i in range(len(X)):
        for j in range(i + 1, len(X)):  
            dist = euclidean(X[i], X[j])
            # print(f"{X[i]}; {X[j]}; {dist}")
            if dist <= epsilon:  
                G.add_edge(i, j, weight=dist)
    
    # Generate minimum spanning tree
    if nx.is_connected(G):
        G = nx.minimum_spanning_tree(G, weight='weight')
       
        last_node = len(X) - 1
        min_distances = nx.shortest_path_length(G, source=last_node, weight='weight')
        min_distances = dict(min_distances)
        min_distances = {k: min_distances[k] for k in sorted(min_distances)}

    return G, min_distances

def epsilon_select(X):
    dmat = cdist(X, X, 'euclidean')
    np.fill_diagonal(dmat, np.inf)
    minval = np.min(dmat, axis=0)
    return np.max(minval)

def plot_graph_3d(G):
    """
    Plots a 3D graph using Matplotlib.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    pos = nx.get_node_attributes(G, 'pos')
    
    for node, coord in pos.items():
        if node == len(pos.items()) - 1:
            ax.scatter(*coord, s=100, c='g')  
        else:
            ax.scatter(*coord, s=10, c='r')  
    
    for edge in G.edges:
        points = np.array([pos[edge[0]], pos[edge[1]]])
        ax.plot(points[:,0], points[:,1], points[:,2], 'b-', lw=2)  
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def rotate_around_y_with_pivot(points, pivot, angle=None):
    """
    Rotate an N*3 ndarray around the y-axis and a pivot point.
    """
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    
    R_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    translated_points = points - pivot
    rotated_points = np.dot(translated_points, R_y.T)  
    rotated_translated_points = rotated_points + pivot

    converted_array = np.array([[sub_array[0] for sub_array in row] for row in rotated_translated_points])
    
    return converted_array

def generate_connection(cell_table, synapse_table, file_path):
    """
    Use synapse table to generate connectivity matrix between cells
    """
    all_neuron_ids = cell_table["pt_root_id"].unique()
    num_neuron = len(all_neuron_ids)

    W = np.zeros((num_neuron, num_neuron))
    totalSyn = np.zeros((num_neuron, num_neuron))
    synCount = np.zeros((num_neuron, num_neuron))

    id_to_index = {neuron_id: index for index, neuron_id in enumerate(all_neuron_ids)}

    for index, row in synapse_table.iterrows():
        if row['pre_pt_root_id'] in id_to_index and row['post_pt_root_id'] in id_to_index:
            i = id_to_index[row['pre_pt_root_id']]
            j = id_to_index[row['post_pt_root_id']]
            syn = row["size"]
            if i != j:
                W[i, j] = 1
                totalSyn[i, j] += syn
                synCount[i, j] += 1

    plt.figure()
    sns.heatmap(W)
    plt.savefig("connectivity.png")

    plt.figure()
    sns.heatmap(totalSyn)
    plt.savefig("synapse.png")
    
    all_neuron_ids = np.arange(1, W.shape[0] + 1) 

    data = [
        {'neuron1': all_neuron_ids[i], 'neuron2': all_neuron_ids[j], 'connectivity': W[i, j], 'total_synapse': totalSyn[i, j], 'syn_count': synCount[i, j]}
        for i in range(W.shape[0]) for j in range(W.shape[1])
    ]

    conn_df = pd.DataFrame(data)
    np.save('conn_info.npy', conn_df)

    write_matrix_to_file(W, f"{file_path[:-4]}_complete.txt")

    # check if any neuron is completely "detached" from the graph
    rows_to_delete = np.where(~W.any(axis=1))[0]
    cols_to_delete = np.where(~W.any(axis=0))[0]

    indices_to_delete = np.intersect1d(rows_to_delete, cols_to_delete)

    W = np.delete(W, indices_to_delete, axis=0)  
    W = np.delete(W, indices_to_delete, axis=1)  

    write_matrix_to_file(W, f"{file_path[:-4]}_modified.txt")


def write_matrix_to_file(W, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                val = int(W[i,j])
                if val != 0:
                    writer.writerow([i, j])


    
    


