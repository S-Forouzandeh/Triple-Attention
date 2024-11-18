
import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import csr_matrix



def process_data(folder_path):
    ground_truth_ratings = None
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
        'movie_directors.xlsx': ['movieID', 'directorID'],
        'movie_actors.xlsx': ['movieID', 'actorID']    
    }

    unique_values = {column: set() for column in file_columns.keys()}

    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path, usecols=columns)
            for column in columns:
                if column in unique_values and column != 'rating':
                    unique_values[column].update(df[column].unique())
            if 'rating' in columns:
                if ground_truth_ratings is None:
                    ground_truth_ratings = df
                else:
                    ground_truth_ratings = pd.concat([ground_truth_ratings, df], ignore_index=True)
        else:
            print(f"File not found: {file_name}")

    return ground_truth_ratings

def create_heterogeneous_graph(folder_path):
    # Create an empty graph
    G = nx.Graph()
    # Create dictionaries to store the number of nodes for each node type
    node_counts = {'userID': 0, 'movieID': 0, 'directorID': 0, 'actorID': 0}

    # Create a dictionary to store mapping between nodes and their attributes
    node_attributes = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create dictionaries to store the number of nodes and edges for each type of relationship
    relationship_counts = {}

    # Create a dictionary to map each file to its corresponding columns
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
        'movie_directors.xlsx': ['movieID', 'directorID'],
        'movie_actors.xlsx': ['movieID', 'actorID']
    }

    # Iterate through the files and read them to populate the graph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Add nodes and edges to the graph based on the file's content
            if 'userID' in columns:
                for _, row in df.iterrows():
                    user_node = f"userID:{row['userID']}"
                    movie_node = f"movieID:{row['movieID']}"
                    rating = row['rating']

                    # Add nodes only if they don't exist
                    if user_node not in G:
                        G.add_node(user_node, type='userID')
                        node_counts['userID'] += 1

                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    G.add_edge(user_node, movie_node, weight=rating)

            if 'directorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    director_node = f"directorID:{row['directorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if director_node not in G:
                        G.add_node(director_node, type='directorID')
                        node_counts['directorID'] += 1

                    G.add_edge(movie_node, director_node)

            if 'actorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    actor_node = f"actorID:{row['actorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if actor_node not in G:
                        G.add_node(actor_node, type='actorID')
                        node_counts['actorID'] += 1

                    G.add_edge(movie_node, actor_node)

    # Print the number of nodes and edges for the graph and the node counts
    print("Graph information:")
    print("Nodes:", len(G.nodes()))
    print("Edges:", len(G.edges()))
    for node_type, count in node_counts.items():
        print(f"Number of {node_type} nodes: {count}")

    return G

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-User--------------------------------------------
#****************************************************************************************
def hypergraph_MU(folder_path):

    # Create an empty hypergraph
    hyper_MU = {}
    relationship_counts = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MU = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create a dictionary to map the 'user_movies.xlsx' file to its corresponding columns
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
    }

    # Iterate through the files and read them to populate the hypergraph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hypergraph and relationship counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                user_node = f"userID:{str(row['userID'])}"
                rating = row['rating']

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MU:
                    hyper_MU[movie_node] = []

                # Add the user node to the hypergraph if it doesn't exist
                if user_node not in hyper_MU:
                    hyper_MU[user_node] = []

                # Add the user node to the movie hyperedge
                hyper_MU[movie_node].append(user_node)

                # Set the type attribute in att_MU
                att_MU[user_node] = {'type': 'userID'}
                att_MU[movie_node] = {'type': 'movieID'}

                edge_weights[(movie_node, user_node)] = rating

                # Count nodes and edges for the userID-movieID relationship
                relationship = 'userID-movieID'
                relationship_counts[relationship] = relationship_counts.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts[relationship]['nodes'] += 2  # Two nodes (movie and user)
                relationship_counts[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MU = {k: v for k, v in hyper_MU.items() if v}
    
    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MU.values())

    print("Hypergraph information of MU:")
    print("Number of hyperedges of MU (nodes):", len(hyper_MU))
    print("Number of edges of MU:", num_edges)

    return hyper_MU, att_MU

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Director--------------------------------------------
#****************************************************************************************
def hypergraph_MD(folder_path):
 
    # Create an empty hyper_MD
    hyper_MD = {}
    relationship_counts_MD = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MD = {}
    
    # Create a dictionary to map the 'director_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_directors.xlsx': ['movieID', 'directorID'],
    }

    # Iterate through the files and read them to populate the hyper_MD
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MD and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                director_node = f"directorID:{str(row['directorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MD:
                    hyper_MD[movie_node] = []

                # Add the director node to the hyper_MD if it doesn't exist
                if director_node not in hyper_MD:
                    hyper_MD[director_node] = []

                # Add the director node to the movie hyperedge
                hyper_MD[movie_node].append(director_node)

                # Set the type attribute in att_MD
                att_MD[director_node] = {'type': 'directorID'}
                att_MD[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the directorID-movieID relationship
                relationship = 'directorID-movieID'
                relationship_counts_MD[relationship] = relationship_counts_MD.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MD[relationship]['nodes'] += 2  # Two nodes (movie and director)
                relationship_counts_MD[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MD = {k: v for k, v in hyper_MD.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MD.values())

    print("Hypergraph information of MD:")
    print("Number of hyperedges of MD (nodes):", len(hyper_MD))
    print("Number of edges of MD:", num_edges)

    return hyper_MD, att_MD

def generate_incidence_matrices_MD(hyper_MD, att_MD):
    """
    Generates incidence matrices for movies and directors.

    Args:
        hyper_MD (dict): Hypergraph representing connections between movies and directors.
        att_MD (dict): Dictionary containing attributes for nodes.

    Returns:
        tuple: A tuple containing the movie-director incidence matrix and its transpose.
    """
    movie_nodes = [node for node in att_MD if att_MD[node]['type'] == 'movieID']
    director_nodes = [node for node in att_MD if att_MD[node]['type'] == 'directorID']

    num_movies = len(movie_nodes)
    num_directors = len(director_nodes)
    incidence_matrix_MD = np.zeros((num_directors, num_movies), dtype=float)  # Swap dimensions

    for movie_index, movie_node in enumerate(movie_nodes):
        directors_connected = hyper_MD.get(movie_node, [])
        for director_node in directors_connected:
            if director_node in director_nodes:
                director_index = director_nodes.index(director_node)
                incidence_matrix_MD[director_index, movie_index] = 1  # Swap indices
    
    print("incidence_matrix_MD Shape", incidence_matrix_MD.shape)
    
    return incidence_matrix_MD

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Actor--------------------------------------------
#****************************************************************************************
def hypergraph_MA(folder_path):
    """
    Generate a hypergraph based on the files found in the specified folder path.

    Args:
    - folder_path (str): Path to the folder containing the files.

    Returns: 
    - hyper_MA (dict): Dictionary representing the hypergraph.
    - att_MA (dict): Dictionary containing attributes of nodes in the hypergraph.
    """
    # Create an empty hyper_MA
    hyper_MA = {}
    relationship_counts_MA = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MA = {}
    
    # Create a dictionary to map the 'actor_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_actors.xlsx': ['movieID', 'actorID'],
    }

    # Iterate through the files and read them to populate the hyper_MA
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MA and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                actor_node = f"actorID:{str(row['actorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MA:
                    hyper_MA[movie_node] = []

                # Add the actor node to the hyper_MA if it doesn't exist
                if actor_node not in hyper_MA:
                    hyper_MA[actor_node] = []

                # Add the actor node to the movie hyperedge
                hyper_MA[movie_node].append(actor_node)

                # Set the type attribute in att_MA
                att_MA[actor_node] = {'type': 'actorID'}
                att_MA[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the actorID-movieID relationship
                relationship = 'actorID-movieID'
                relationship_counts_MA[relationship] = relationship_counts_MA.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MA[relationship]['nodes'] += 2  # Two nodes (movie and actor)
                relationship_counts_MA[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MA = {k: v for k, v in hyper_MA.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MA.values())

    print("Hypergraph information of MA:")
    print("Number of hyperedges of MA (nodes):", len(hyper_MA))
    print("Number of edges of MA:", num_edges)

    return hyper_MA, att_MA

#****************************************************************************************
#----------------------------------- Dynamic Hypergraph with Attention based Simialrity-----------------------------
#****************************************************************************************
def generate_incidence_matrices_MU(hyper_MU, att_MU):
    movie_nodes = [node for node in att_MU if att_MU[node]['type'] == 'movieID']
    user_nodes = [node for node in att_MU if att_MU[node]['type'] == 'userID']

    num_movies = len(movie_nodes)
    num_users = len(user_nodes)
    incidence_matrix_MU = np.zeros((num_users, num_movies), dtype=float)

    for movie_index, movie_node in enumerate(movie_nodes):
        users_connected = hyper_MU.get(movie_node, [])
        for user_node in users_connected:
            if user_node in user_nodes:
                user_index = user_nodes.index(user_node)
                incidence_matrix_MU[user_index, movie_index] = 1

    print("Incidence Matrix MU Shape:", incidence_matrix_MU.shape)
    return incidence_matrix_MU


def generate_incidence_matrices_MD(hyper_MD, att_MD):
    movie_nodes = [node for node in att_MD if att_MD[node]['type'] == 'movieID']
    director_nodes = [node for node in att_MD if att_MD[node]['type'] == 'directorID']

    num_movies = len(movie_nodes)
    num_directors = len(director_nodes)
    incidence_matrix_MD = np.zeros((num_directors, num_movies), dtype=float)

    for movie_index, movie_node in enumerate(movie_nodes):
        directors_connected = hyper_MD.get(movie_node, [])
        for director_node in directors_connected:
            if director_node in director_nodes:
                director_index = director_nodes.index(director_node)
                incidence_matrix_MD[director_index, movie_index] = 1

    print("incidence_matrix_MD Shape", incidence_matrix_MD.shape)
    
    return incidence_matrix_MD

def generate_incidence_matrices_MA(hyper_MA, att_MA):
    movie_nodes = [node for node in att_MA if att_MA[node]['type'] == 'movieID']
    actor_nodes = [node for node in att_MA if att_MA[node]['type'] == 'actorID']

    num_movies = len(movie_nodes)
    num_actors = len(actor_nodes)
    incidence_matrix_MA = np.zeros((num_actors, num_movies), dtype=float)

    for movie_index, movie_node in enumerate(movie_nodes):
        actors_connected = hyper_MA.get(movie_node, [])
        for actor_node in actors_connected:
            if actor_node in actor_nodes:
                actor_index = actor_nodes.index(actor_node)
                incidence_matrix_MA[actor_index, movie_index] = 1

    print("incidence_matrix_MA Shape", incidence_matrix_MA.shape)
    
    return incidence_matrix_MA

def calculate_hyperedge_weights(incidence_matrix):
    if isinstance(incidence_matrix, tuple):
        raise ValueError("Expected numpy array but received a tuple.")
    
    print("Incidence Matrix Shape:", incidence_matrix.shape)
    
    if incidence_matrix.ndim != 2:
        raise ValueError("Incidence matrix must be 2D.")
    
    row_sums = np.sum(incidence_matrix, axis=0)
    if np.max(row_sums) == 0:
        raise ValueError("Maximum sum of rows is zero, cannot divide by zero.")
    
    hyperedge_weights = row_sums / np.max(row_sums)
    return hyperedge_weights

def generate_sparsified_neighbors(cosine_sim_matrix, num_neighbors=10):
    num_nodes = cosine_sim_matrix.shape[0]
    sparsified_neighbors = {}
    
    for i in range(num_nodes):
        sim_scores = cosine_sim_matrix[i]
        neighbor_indices = np.argsort(-sim_scores)[:num_neighbors]
        sparsified_neighbors[i] = neighbor_indices.tolist()
    
    return sparsified_neighbors

def sparse_attention(cosine_sim, sparsified_neighbors):
    num_nodes = cosine_sim.shape[0]
    sparse_attention_weights = np.zeros_like(cosine_sim)
    
    for i in range(num_nodes):
        neighbors = sparsified_neighbors[i]
        for j in neighbors:
            sparse_attention_weights[i, j] = cosine_sim[i, j]
    
    activated_sparse_attention = leaky_relu(sparse_attention_weights)
    attention_weights_sparse = np.exp(activated_sparse_attention) / np.sum(np.exp(activated_sparse_attention), axis=1, keepdims=True)
    
    return attention_weights_sparse

def attention_cosine_similarity_multi(incidence_matrix, relationship_weights, hyperedge_weights, sparsified_neighbors):
    if incidence_matrix.shape[0] == 0:
        raise ValueError("Input incidence matrix has zero samples.")
    
    num_relationships = len(relationship_weights)
    
    incidence_matrix_sparse = csr_matrix(incidence_matrix)  # Convert to sparse matrix for efficiency
    cosine_sim = cosine_similarity(incidence_matrix_sparse)
    activated_cosine_sim = leaky_relu(cosine_sim)
    attention_weights_multi = np.exp(activated_cosine_sim) / np.sum(np.exp(activated_cosine_sim), axis=1, keepdims=True)
    
    # Compute high-order attention weights
    hyperedge_weights_matrix = incidence_matrix_sparse @ np.diag(hyperedge_weights)
    highorder_attention = cosine_similarity(hyperedge_weights_matrix)
    activated_highorder_attention = leaky_relu(highorder_attention)
    attention_weights_highorder = np.exp(activated_highorder_attention) / np.sum(np.exp(activated_highorder_attention), axis=1, keepdims=True)
    
    # Compute sparse attention weights
    attention_weights_sparse = sparse_attention(cosine_sim, sparsified_neighbors)
    
    # Compute final attention weights
    final_attention_weights = (
        np.sum([relationship_weights[r] * attention_weights_multi[r] for r in range(num_relationships)], axis=0)
        + attention_weights_highorder
        + attention_weights_sparse
    )
    
    return final_attention_weights

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def dynamic_laplacian(incidence_matrix, attention_weights):
    weight_matrix = np.diag(np.sum(attention_weights, axis=1)) - attention_weights
    dynamic_Laplacian = incidence_matrix.T @ weight_matrix @ incidence_matrix
    dynamic_Laplacian = dynamic_Laplacian / np.max(dynamic_Laplacian)
    return dynamic_Laplacian

def compute_hypergraph_laplacian_MU(hyper_MU, att_MU, relationship_weights, num_neighbors):
    incidence_matrix_MU = generate_incidence_matrices_MU(hyper_MU, att_MU)
    hyperedge_weights = calculate_hyperedge_weights(incidence_matrix_MU)  # Ensure this function is defined
    attention_weights = attention_cosine_similarity_multi(incidence_matrix_MU, relationship_weights, hyperedge_weights, num_neighbors)
    Dyn_Laplacian_MU = dynamic_laplacian(incidence_matrix_MU, attention_weights)
    return Dyn_Laplacian_MU

def compute_hypergraph_laplacian_MU(hyper_MU, att_MU, relationship_weights, num_neighbors):
    incidence_matrix_MU = generate_incidence_matrices_MU(hyper_MU, att_MU)
    hyperedge_weights = calculate_hyperedge_weights(incidence_matrix_MU)
    
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(incidence_matrix_MU)
    
    # Generate sparsified neighbors
    sparsified_neighbors = generate_sparsified_neighbors(cosine_sim, num_neighbors)
    
    attention_weights = attention_cosine_similarity_multi(
        incidence_matrix_MU,
        relationship_weights,
        hyperedge_weights,
        sparsified_neighbors
    )
    
    Dyn_Laplacian_MU = dynamic_laplacian(incidence_matrix_MU, attention_weights)
    return Dyn_Laplacian_MU

def compute_hypergraph_laplacian_MD(hyper_MD, att_MD, relationship_weights, num_neighbors):
    incidence_matrix_MD = generate_incidence_matrices_MD(hyper_MD, att_MD)
    hyperedge_weights = calculate_hyperedge_weights(incidence_matrix_MD)  # Ensure this function is defined
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(incidence_matrix_MD)
    
    # Generate sparsified neighbors
    sparsified_neighbors = generate_sparsified_neighbors(cosine_sim, num_neighbors)
    
    attention_weights = attention_cosine_similarity_multi(
        incidence_matrix_MD,
        relationship_weights,
        hyperedge_weights,
        sparsified_neighbors
    )
    
    Dyn_Laplacian_MD = dynamic_laplacian(incidence_matrix_MD, attention_weights)
    
    return Dyn_Laplacian_MD

def compute_hypergraph_laplacian_MA(hyper_MA, att_MA, relationship_weights, num_neighbors):
    incidence_matrix_MA = generate_incidence_matrices_MA(hyper_MA, att_MA)
    hyperedge_weights = calculate_hyperedge_weights(incidence_matrix_MA)  # Ensure this function is defined
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(incidence_matrix_MA)
    
    # Generate sparsified neighbors
    sparsified_neighbors = generate_sparsified_neighbors(cosine_sim, num_neighbors)
    
    attention_weights = attention_cosine_similarity_multi(
        incidence_matrix_MA,
        relationship_weights,
        hyperedge_weights,
        sparsified_neighbors
    )
    
    Dyn_Laplacian_MA = dynamic_laplacian(incidence_matrix_MA, attention_weights)

    return Dyn_Laplacian_MA

def resize_matrix(matrix, new_shape):
    resized_matrix = np.zeros(new_shape)
    min_rows = min(matrix.shape[0], new_shape[0])
    min_cols = min(matrix.shape[1], new_shape[1])
    resized_matrix[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
    return resized_matrix

class AdaptiveEdgeDropping(nn.Module):
    def __init__(self):
        super(AdaptiveEdgeDropping, self).__init__()
        self.drop_param = nn.Parameter(torch.randn(1))
        self.gamma = nn.Parameter(torch.randn(1))
    
    def forward(self, matrix, drop_ratio):
        matrix = np.asarray(matrix)
        dropped_matrix = matrix.copy()
        edges = np.transpose(np.nonzero(matrix))
        num_edges = len(edges)
        
        drop_probabilities = 1 / (1 + np.exp(-self.drop_param.item() * (1 - matrix[edges[:, 0], edges[:, 1]]) + self.gamma.item() * matrix[edges[:, 0], edges[:, 1]]))
        
        drop_indices = np.random.choice(num_edges, int(num_edges * drop_ratio), replace=False, p=drop_probabilities/np.sum(drop_probabilities))
        for index in drop_indices:
            dropped_matrix[edges[index][0], edges[index][1]] = 0
        
        return dropped_matrix

def fuse_laplacian_matrices(hyper_MU, att_MU, hyper_MD, att_MD, hyper_MA, att_MA, method='concatenate'):

    # Define relationship weights (learnable parameters)
    relationship_weights = [0.3, 0.5, 0.2]  

    #Call the functions with the appropriate hypergraph and attention matrices
    num_neighbors =5
    Dyn_Laplacian_MU = compute_hypergraph_laplacian_MU(hyper_MU, att_MU, relationship_weights, num_neighbors )
    Dyn_Laplacian_MD = compute_hypergraph_laplacian_MD(hyper_MD, att_MD, relationship_weights, num_neighbors )
    Dyn_Laplacian_MA = compute_hypergraph_laplacian_MA(hyper_MA, att_MA, relationship_weights, num_neighbors ) 
    
    # Determine the new shape for resizing
    new_shape = (
        max(Dyn_Laplacian_MU.shape[0], Dyn_Laplacian_MD.shape[0], Dyn_Laplacian_MA.shape[0]),
        max(Dyn_Laplacian_MU.shape[1], Dyn_Laplacian_MD.shape[1], Dyn_Laplacian_MA.shape[1])
    )
    
    # Resize all matrices to the new shape
    Dyn_Laplacian_MU = resize_matrix(Dyn_Laplacian_MU, new_shape)
    Dyn_Laplacian_MD = resize_matrix(Dyn_Laplacian_MD, new_shape)
    Dyn_Laplacian_MA = resize_matrix(Dyn_Laplacian_MA, new_shape)
    
    # Fuse the matrices by concatenating along the columns
    fused_MUD = np.concatenate((Dyn_Laplacian_MU, Dyn_Laplacian_MD), axis=1)
    fused_MUA = np.concatenate((Dyn_Laplacian_MU, Dyn_Laplacian_MA), axis=1)
    
    # Normalize the fused matrices
    fused_MUD = fused_MUD / np.max(fused_MUD) if np.max(fused_MUD) != 0 else fused_MUD
    fused_MUA = fused_MUA / np.max(fused_MUA) if np.max(fused_MUA) != 0 else fused_MUA

    return Dyn_Laplacian_MU, fused_MUD, fused_MUA

def generate_gcl_augmentations(Dyn_Laplacian_MU, fused_MUD, fused_MUA, drop_param, gamma, drop_ratio=0.1):
    # Instantiate the model with learnable parameters
    edge_dropping = AdaptiveEdgeDropping()
    edge_dropping.drop_param.data = torch.tensor([drop_param], dtype=torch.float32)
    edge_dropping.gamma.data = torch.tensor([gamma], dtype=torch.float32)
    
    # Apply adaptive edge dropping
    Aug_Dyn_Laplacian_MU_dropped = edge_dropping(Dyn_Laplacian_MU, drop_ratio)
    Aug_fused_MUD_dropped = edge_dropping(fused_MUD, drop_ratio)
    Aug_fused_MUA_dropped = edge_dropping(fused_MUA, drop_ratio)
   
    # Normalize dimensions if needed
    Aug_Dyn_Laplacian_MU_dropped = normalize_dimensions(Aug_Dyn_Laplacian_MU_dropped, Dyn_Laplacian_MU.shape)
    Aug_fused_MUD_dropped = normalize_dimensions(Aug_fused_MUD_dropped, fused_MUD.shape)
    Aug_fused_MUA_dropped = normalize_dimensions(Aug_fused_MUA_dropped, fused_MUA.shape)

    return (Dyn_Laplacian_MU, Aug_Dyn_Laplacian_MU_dropped), \
           (fused_MUD, Aug_fused_MUD_dropped), \
           (fused_MUA, Aug_fused_MUA_dropped)
           
def normalize_dimensions(matrix, target_shape):
    if matrix.size == 0:
        return np.zeros(target_shape)
    if matrix.shape != target_shape:
        # Reshape the matrix to the target shape
        matrix = np.resize(matrix, target_shape)
    return matrix

def contrastive_loss(z, g, tau):
    if z.shape[1] != g.shape[1]:
        raise ValueError(f"Shape mismatch: z shape {z.shape} and g shape {g.shape} cannot be multiplied.")

    num_nodes = z.shape[0]
    
    # Compute the similarity matrix using PyTorch
    sim_matrix = torch.mm(z, g.t()) / tau
    sim_matrix = torch.clamp(sim_matrix, -50, 50)  # Clip the values to avoid overflow
    exp_sim_matrix = torch.exp(sim_matrix)
    exp_sim_matrix_sum = torch.sum(exp_sim_matrix, dim=1) + 1e-9  # Add epsilon to avoid division by zero
    
    loss = -torch.log(torch.diagonal(exp_sim_matrix) / exp_sim_matrix_sum + 1e-9)  # Add epsilon to avoid log(0)
    return torch.mean(loss)

def local_contrastive_loss(z, tau):
    num_nodes = z.shape[0]
    
    # Compute the similarity matrix using PyTorch
    sim_matrix = torch.mm(z, z.t()) / tau
    sim_matrix = torch.clamp(sim_matrix, -50, 50)
    exp_sim_matrix = torch.exp(sim_matrix)
    exp_sim_matrix_sum = torch.sum(exp_sim_matrix, dim=1) + 1e-9
    loss = -torch.log(torch.diagonal(exp_sim_matrix) / exp_sim_matrix_sum + 1e-9)
    return torch.mean(loss)

def hierarchical_global_contrastive_loss(z, layers, tau):
    total_loss = 0
    for l in range(len(layers)):
        for m in range(len(layers[l])):
            z_layer = layers[l][m]
            if z_layer.size(0) == 0:
                continue
            if z_layer.dim() != 2:
                continue

            sim_matrix = torch.mm(z_layer, z_layer.t()) / tau
            sim_matrix = torch.clamp(sim_matrix, -50, 50)
            exp_sim_matrix = torch.exp(sim_matrix)
            exp_sim_matrix_sum = torch.sum(exp_sim_matrix, dim=1) + 1e-9
            loss = -torch.log(torch.diagonal(exp_sim_matrix) / exp_sim_matrix_sum + 1e-9)
            total_loss += torch.mean(loss)
    return total_loss

def gcl_with_augmentations(Dyn_Laplacian_MU, fused_MUD, fused_MUA, Aug_Dyn_Laplacian_MU_dropped, Aug_fused_MUD_dropped, Aug_fused_MUA_dropped, tau=0.5):
    # Calculate local contrastive losses
    local_loss_MU = (
        local_contrastive_loss(Dyn_Laplacian_MU, tau) +
        local_contrastive_loss(Aug_Dyn_Laplacian_MU_dropped, tau))
    local_loss_MUD = (
        local_contrastive_loss(fused_MUD, tau) +
        local_contrastive_loss(Aug_fused_MUD_dropped, tau))
    local_loss_MUA = (
        local_contrastive_loss(fused_MUA, tau) +
        local_contrastive_loss(Aug_fused_MUA_dropped, tau))
    
    # Calculate global contrastive losses
    global_loss_MU = hierarchical_global_contrastive_loss(Dyn_Laplacian_MU, [Aug_Dyn_Laplacian_MU_dropped], tau)
    global_loss_MUD = hierarchical_global_contrastive_loss(fused_MUD, [Aug_fused_MUD_dropped], tau)
    global_loss_MUA = hierarchical_global_contrastive_loss(fused_MUA, [Aug_fused_MUA_dropped], tau)

    local_loss = local_loss_MU + local_loss_MUD + local_loss_MUA 
    global_loss = global_loss_MU + global_loss_MUD + global_loss_MUA 

    # Combined loss
    lambda1, lambda2, lambda3 = 0.5, 0.5, 0.2  # weights
    combined_loss = lambda1 * local_loss + lambda2 * global_loss
    return combined_loss

def train(num_epochs, learning_rate, Dyn_Laplacian_MU, fused_MUD, fused_MUA, 
          Aug_Dyn_Laplacian_MU_dropped, Aug_fused_MUD_dropped, Aug_fused_MUA_dropped, tau=0.5):
    # Convert matrices to tensors
    tensors = {
        "Dyn_Laplacian_MU": torch.tensor(Dyn_Laplacian_MU, dtype=torch.float32, requires_grad=True),
        "fused_MUD": torch.tensor(fused_MUD, dtype=torch.float32, requires_grad=True),
        "fused_MUA": torch.tensor(fused_MUA, dtype=torch.float32, requires_grad=True),
        "Aug_Dyn_Laplacian_MU_dropped": torch.tensor(Aug_Dyn_Laplacian_MU_dropped, dtype=torch.float32, requires_grad=True),
        "Aug_fused_MUD_dropped": torch.tensor(Aug_fused_MUD_dropped, dtype=torch.float32, requires_grad=True),
        "Aug_fused_MUA_dropped": torch.tensor(Aug_fused_MUA_dropped, dtype=torch.float32, requires_grad=True),
    }
    
    # Define the optimizer
    optimizer = optim.Adam([
        {"params": tensors["Dyn_Laplacian_MU"]},
        {"params": tensors["fused_MUD"]},
        {"params": tensors["fused_MUA"]}
    ], lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute total contrastive loss
        total_loss = gcl_with_augmentations(
            tensors["Dyn_Laplacian_MU"], tensors["fused_MUD"], tensors["fused_MUA"], tensors["Aug_Dyn_Laplacian_MU_dropped"], tensors["Aug_fused_MUD_dropped"], tensors["Aug_fused_MUA_dropped"], tau)
        
        # Backpropagation
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")

    print("Training completed.")

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)  # Ensure output is between 0 and 1
        return x
# Custom Dataset
class UserItemDataset(Dataset):
    def __init__(self, fused_embedding, binary_matrix):
        self.fused_embedding = fused_embedding
        self.binary_matrix = binary_matrix

    def __len__(self):
        return len(self.binary_matrix)

    def __getitem__(self, idx):
        user_embedding = self.fused_embedding[idx]
        label = self.binary_matrix[idx]
        return torch.FloatTensor(user_embedding), torch.FloatTensor(label)

def create_binary_ground_truth_matrix(folder_path):
    file_name = 'user_movies.xlsx'
    columns = ['userID', 'movieID', 'rating']

    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path, usecols=columns)

    unique_users = sorted(df['userID'].unique())
    unique_movies = sorted(df['movieID'].unique())

    num_users = len(unique_users)
    num_movies = len(unique_movies)
    binary_ground_truth = np.zeros((num_users, num_movies), dtype=int)

    for _, row in df.iterrows():
        user_idx = unique_users.index(row['userID'])
        movie_idx = unique_movies.index(row['movieID'])
        binary_ground_truth[user_idx, movie_idx] = 1

    return binary_ground_truth, unique_users, unique_movies

def fuse_embeddings(embeddings):
    fused_embedding = np.concatenate(embeddings, axis=1)
    return fused_embedding

def split_and_save_data(binary_ground_truth, test_size=0.2, random_state=42):
    num_users = binary_ground_truth.shape[0]
    indices = np.arange(num_users)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_data = binary_ground_truth[train_indices]
    test_data = binary_ground_truth[test_indices]
    return train_data, test_data

def calculate_hit_rate(true_items, predicted_items, k):
    """
    Calculate Hit Rate @ k
    Args:
        true_items: Binary vector of actual interactions
        predicted_items: Predicted scores for all items
        k: Number of top items to consider
    Returns:
        1 if at least one true positive item is in top-k, 0 otherwise
    """
    # Get indices of items that user actually interacted with (true positives)
    true_item_indices = np.where(true_items > 0)[0]
    
    # If user has no interactions in the test set, return 0
    if len(true_item_indices) == 0:
        return 0
        
    # Get top k predicted items
    predicted_item_indices = np.argsort(predicted_items)[-k:]
    
    # Check if any of the true items appear in top k predictions
    hits = np.isin(predicted_item_indices, true_item_indices)
    return 1 if np.any(hits) else 0

def calculate_precision_recall_f1(true_items, predicted_items, k):
    """
    Calculate Precision, Recall, and F1 Score @ k with fixes for numerical stability
    """
    # Get indices of items that user actually interacted with
    true_item_indices = np.where(true_items > 0)[0]
    
    # If user has no interactions in the test set, return 0 for all metrics
    if len(true_item_indices) == 0:
        return 0, 0, 0
        
    # Get top k predicted items (ensure we're getting highest scores)
    predicted_item_indices = np.argsort(predicted_items)[-k:][::-1]  # Reverse to get highest first
    
    # Calculate number of true positives
    true_positives = len(np.intersect1d(predicted_item_indices, true_item_indices))
    
    # Calculate precision@k and recall@k
    precision = true_positives / min(k, len(predicted_item_indices))  # Handle case where we have fewer than k predictions
    recall = true_positives / len(true_item_indices) if len(true_item_indices) > 0 else 0
    
    # Calculate F1 score with numerical stability
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1

def evaluate_model(model, test_loader):
    model.eval()
    metrics_at_k = {k: {'hits': 0, 'precision': [], 'recall': [], 'f1': []} 
                    for k in [10, 20, 50]}
    total_users = 0
    valid_users = 0
    
    with torch.no_grad():
        for batch_idx, (user_embedding, label) in enumerate(test_loader):
            predictions = model(user_embedding)
            batch_size = label.shape[0]
            total_users += batch_size
            
            # Process each user in the batch
            for i in range(batch_size):
                user_true = label[i].numpy()
                user_pred = predictions[i].numpy()
                
                # Only count users who have at least one interaction in test set
                if np.sum(user_true) > 0:
                    valid_users += 1
                    
                    for k in metrics_at_k.keys():
                        # Calculate Hit Rate
                        hit = calculate_hit_rate(user_true, user_pred, k)
                        metrics_at_k[k]['hits'] += hit
                        
                        # Calculate Precision, Recall, and F1
                        precision, recall, f1 = calculate_precision_recall_f1(
                            user_true, user_pred, k
                        )
                        metrics_at_k[k]['precision'].append(precision)
                        metrics_at_k[k]['recall'].append(recall)
                        metrics_at_k[k]['f1'].append(f1)
    
    print("\nRecommendation Metrics:")
    print(f"Total users in test set: {total_users}")
    print(f"Users with interactions: {valid_users}")
    
    for k in metrics_at_k.keys():
        if len(metrics_at_k[k]['precision']) > 0:
            hit_rate = (metrics_at_k[k]['hits'] / valid_users) * 100
            avg_precision = np.mean(metrics_at_k[k]['precision']) * 100
            avg_recall = np.mean(metrics_at_k[k]['recall']) * 100
            avg_f1 = np.mean(metrics_at_k[k]['f1']) * 100
            
            print(f"\nMetrics @{k}:")
            print(f"Hit Rate: {hit_rate:.2f}%")
            print(f"Precision: {avg_precision:.2f}%")
            print(f"Recall: {avg_recall:.2f}%")
            print(f"F1 Score: {avg_f1:.2f}%")
            print(f"Number of evaluated users: {len(metrics_at_k[k]['precision'])}")

    return metrics_at_k

def main():
    folder_path = 'C:\\IMDB'
    
    # Assuming the required functions and variables are defined and generated elsewhere:
    graph = create_heterogeneous_graph(folder_path)
    hyper_MU, att_MU = hypergraph_MU(folder_path)
    hyper_MD, att_MD = hypergraph_MD(folder_path)
    hyper_MA, att_MA = hypergraph_MA(folder_path)

    
    # Define relationship weights (learnable parameters)
    relationship_weights = [0.3, 0.5, 0.2]  
    num_neighbors=10

    #Call the functions with the appropriate hypergraph and attention matrices
    Dyn_Laplacian_MU = compute_hypergraph_laplacian_MU(hyper_MU, att_MU, relationship_weights, num_neighbors )
    Dyn_Laplacian_MD = compute_hypergraph_laplacian_MD(hyper_MD, att_MD, relationship_weights, num_neighbors )
    Dyn_Laplacian_MA = compute_hypergraph_laplacian_MA(hyper_MA, att_MA, relationship_weights, num_neighbors ) 

    Dyn_Laplacian_MU, fused_MUD, fused_MUA = fuse_laplacian_matrices(
        hyper_MU, att_MU, hyper_MD, att_MD, hyper_MA, att_MA )

    drop_param = 0.5  
    gamma = 0.5      

    augmented_views = generate_gcl_augmentations(Dyn_Laplacian_MU, fused_MUD, fused_MUA, drop_param, gamma)
    (Dyn_Laplacian_MU, Aug_Dyn_Laplacian_MU_dropped), \
    (fused_MUD, Aug_fused_MUD_dropped), \
    (fused_MUA, Aug_fused_MUA_dropped) = augmented_views

    embeddings = [Dyn_Laplacian_MU, fused_MUD, fused_MUA, 
                  Aug_Dyn_Laplacian_MU_dropped, Aug_fused_MUD_dropped,
                  Aug_fused_MUA_dropped]
    fused_embedding = fuse_embeddings(embeddings)
    binary_ground_truth, unique_users, unique_movies = create_binary_ground_truth_matrix(folder_path)

    # Modify the data splitting to ensure we have enough test data
    train_data, test_data = split_and_save_data(binary_ground_truth, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = UserItemDataset(fused_embedding, train_data)
    test_dataset = UserItemDataset(fused_embedding, test_data)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Model definition and training remains the same
    input_dim = fused_embedding.shape[1]
    hidden_dim = 128
    output_dim = binary_ground_truth.shape[1]
    
    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user_embedding, label in train_loader:
            optimizer.zero_grad()
            output = model(user_embedding)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()