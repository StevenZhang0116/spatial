from sklearn.manifold import Isomap
from kneed import KneeLocator
import umap 
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.decomposition import PCA, KernelPCA
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
import scipy
from scipy.stats import bernoulli, multivariate_normal
from scipy.integrate import nquad
import seaborn as sns 
from scipy.linalg import sqrtm, det

# isomap analysis is time-consuming when sample size is large
def isomap_test(X, grange):
    dim = []
    re_err = []
    for n_comp in range(1, grange):
        embedding = Isomap(n_components=n_comp)
        X_transformed = embedding.fit_transform(X)
        dim.append(n_comp)
        re_err.append(embedding.reconstruction_error())

    # select the knee locator
    kn_isomap = KneeLocator(dim, re_err, curve='convex', direction='decreasing').knee
    # use that number of PC to embed
    embedding = Isomap(n_components=kn_isomap)
    X_transformed = embedding.fit_transform(X)

    return dim, re_err, X_transformed, kn_isomap

# umap analysis
def umap_test(X, grange):
    dim = []
    re_err = []
    for n_comp in range(1, grange):
        # UMAP embedding
        reducer = umap.UMAP(n_components=n_comp)
        X_transformed = reducer.fit_transform(X)

        # Inverse transform is not directly available in UMAP,
        # so we need to fit a separate inverse transformation model
        # For example, using a supervised learning model
        # Here, we use a Nearest Neighbors approach
        knn = KNeighborsRegressor()
        knn.fit(X_transformed, X)
        X_reconstructed = knn.predict(X_transformed)

        # Calculate reconstruction error
        error = mean_squared_error(X, X_reconstructed)
        
        dim.append(n_comp)
        re_err.append(error)

    return dim, re_err

# Kernel PCA analysis
def kernel_pca_test(X, grange, kernel='rbf', n_neighbors=5):
    dim = []
    re_err = []
    for n_comp in range(1, grange):
        # Kernel PCA
        embedding = KernelPCA(n_components=n_comp, kernel=kernel)
        X_transformed = embedding.fit_transform(X)

        # Use K-Nearest Neighbors to estimate the inverse transformation
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_transformed, X)
        X_reconstructed = knn.predict(X_transformed)

        # Calculate reconstruction error
        error = mean_squared_error(X, X_reconstructed)

        dim.append(n_comp)
        re_err.append(error)

    return dim, re_err

# lle analysis
def lle_test(X, grange):
    dim = []
    re_err = []
    for n_comp in range(1, grange):
        # Apply LLE
        embedding = LocallyLinearEmbedding(n_components=n_comp)
        X_transformed = embedding.fit_transform(X)

        # Reconstruct the high-dimensional data points
        nbrs = NearestNeighbors(n_neighbors=n_comp+1, algorithm='auto').fit(X_transformed)
        distances, indices = nbrs.kneighbors(X_transformed)
        reconstructed_X = np.zeros(X.shape)

        for i in range(X_transformed.shape[0]):
            # Compute the reconstructed point as a weighted average of neighbors
            weights = distances[i] / np.sum(distances[i])
            for j in range(1, len(indices[i])):
                reconstructed_X[i] += weights[j] * X[indices[i][j]]

        # Calculate reconstruction error
        error = mean_squared_error(X, reconstructed_X)
        
        dim.append(n_comp)
        re_err.append(error)
        
    return dim, re_err

def eigenspectrum(adj):
    nec = np.linalg.eigvalsh(adj)

    mean = np.mean(nec)
    std_dev = np.std(nec)      

    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    newnec = nec[(nec >= lower_bound) & (nec <= upper_bound)]
    within_ratio = len(newnec)/len(nec)
    unique_ratio = len(set(newnec))/len(newnec)

    return [nec[::-1], newnec, within_ratio, unique_ratio]

def calculate_overlap_volume(mean1, cov1, mean2, cov2, bounds):
    """
    Time costly
    """
    # Define the product of the two Gaussian PDFs
    def pdf_product(x, y, z):
        return multivariate_normal.pdf([x, y, z], mean=mean1, cov=cov1) * \
               multivariate_normal.pdf([x, y, z], mean=mean2, cov=cov2)

    # Compute the overlap volume
    overlap_volume, error = nquad(pdf_product, bounds)
    return overlap_volume, error

def prob_bind(A, index=0):
    """
    Consider the pairwise connectivity matrix as binding probability.
    If index=1, only run Bernoulli on the upper triangular matrix and symmetrize the result.
    """
    bind_matrix = A.copy()
    n = A.shape[0]

    if index == 0:
        # Original behavior: Run Bernoulli on entire matrix
        for i in range(n):
            for j in range(n):
                prob = A[i][j]
                sample = bernoulli.rvs(prob, size=1)[0] 
                bind_matrix[i][j] = sample
    elif index == 1:
        # Run Bernoulli only on upper triangular matrix and symmetrize
        for i in range(n):
            for j in range(i+1, n):  # Start from i+1 to exclude diagonal
                prob = A[i][j]
                sample = bernoulli.rvs(prob, size=1)[0]
                bind_matrix[i][j] = sample
                bind_matrix[j][i] = sample  # Symmetrize

    return bind_matrix


def is_symmetric(matrix):
    """
    Check if a matrix is symmetric.
    """
    assert np.allclose(matrix, matrix.T) == True
    return 

def clamp_matrix(matrix, clamp_value=1):
    """
    Clamp the values of a matrix such that any value greater than the clamp_value is set to the clamp_value.
    """
    return np.minimum(matrix, clamp_value)