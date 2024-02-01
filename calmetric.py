import tensorflow as tf 
import scipy
import bct
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def cal_comm_weight_matrix(A, comms_factor=1):
    """
    Taken from Achterberg's SNN
    Version of SE1 regulariser which combines the spatial and communicability parts in loss function.
    """
    # Convert sparse matrix to dense if necessary
    if isinstance(A, tf.sparse.SparseTensor):
        A = tf.sparse.to_dense(A)
    elif isinstance(A, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        A = A.todense()

    abs_weight_matrix = tf.math.abs(A)

    # Calculate weighted communicability
    stepI = tf.math.reduce_sum(abs_weight_matrix, axis=1)
    stepII = tf.math.pow(stepI, -0.5)
    stepIII = tf.linalg.diag(stepII)
    stepIV = tf.linalg.expm(stepIII @ abs_weight_matrix @ stepIII)
    # comms_matrix = tf.linalg.set_diag(stepIV, tf.zeros(stepIV.shape[0:-1]))
    comms_matrix = tf.linalg.set_diag(stepIV, tf.zeros(stepIV.shape[0:-1], dtype=stepIV.dtype))

    # Multiply absolute weights with communicability weights
    comms_matrix = comms_matrix ** comms_factor
    comms_weight_matrix = tf.math.multiply(abs_weight_matrix, comms_matrix)
    return comms_weight_matrix

def cal_modularity(A):
    """
    Taken from Achterberg's SNN
    Calculate modularity value
    """
    _, q_stat = bct.modularity_und(A, gamma=1)
    return q_stat

def binaryize_mat(A, theq):
    """
    Binarize the weight connectivity matrix
    """
    binary_weight_matrix = A.copy()
    thresh = np.quantile(A, q=theq)
    matrix_mask = A > thresh
    binary_weight_matrix[matrix_mask] = 1.0
    binary_weight_matrix[~matrix_mask] = 0.0
    return binary_weight_matrix

def cal_smallwordness(A, neuron_num):
    """
    Taken from Achterberg's SNN
    Calculate 
    """
    clu = np.mean(bct.clustering_coef_bu(A))
    pth = bct.efficiency_bin(A)
    # Run nperm null models
    nperm = 1000
    cluperm = np.zeros((nperm,1))
    pthperm = np.zeros((nperm,1))
    for perm in range(nperm):
        Wperm = np.random.rand(neuron_num, neuron_num)
        # Make it into a matrix
        Wperm = np.matrix(Wperm)
        # Make symmetrical
        Wperm = Wperm+Wperm.T
        Wperm = np.divide(Wperm,2)
        # Binarise
        threshold, upper, lower = .7,1,0
        Aperm = np.where(Wperm>threshold,upper,lower)
        # Take null model
        cluperm[perm] = np.mean(bct.clustering_coef_bu(Aperm))
        pthperm[perm] = bct.efficiency_bin(Aperm)
    # Take the average of the nulls
    clunull = np.mean(cluperm)
    pthnull = np.mean(pthperm)
    # Compute the small worldness
    smw = np.divide(np.divide(clu,clunull),np.divide(pth,pthnull))
    return smw

def bhattacharyya_distance(mu1, Sigma1, mu2, Sigma2):
    """
    Calculate the Bhattacharyya distance between two 3D multivariate Gaussian distributions.
    """
    Sigma = 0.5 * (Sigma1 + Sigma2)
    
    diff_mu = mu2 - mu1
    inv_Sigma = np.linalg.inv(Sigma)
    term_mu = 0.125 * np.dot(np.dot(diff_mu.T, inv_Sigma), diff_mu)
    
    sqrt_det = np.sqrt(np.linalg.det(Sigma1) * np.linalg.det(Sigma2))
    det_Sigma = np.linalg.det(Sigma)
    term_sigma = 0.5 * np.log(det_Sigma / sqrt_det)
    
    D_B = term_mu + term_sigma
    
    return D_B

def combine_gaussians(mean1, cov1, mean2, cov2):
    """
    Combine two multivariate Gaussian distributions.
    """
    # Compute the precision matrices
    precision1 = np.linalg.inv(cov1)
    precision2 = np.linalg.inv(cov2)
    
    # Compute the combined covariance matrix
    combined_cov = np.linalg.inv(precision1 + precision2)
    
    # Compute the combined mean
    combined_mean = combined_cov @ (precision1 @ mean1 + precision2 @ mean2)
    
    return combined_mean, combined_cov

def wasserstein_distance(mean1, cov1, mean2, cov2):
    """
    Compute the squared 2-Wasserstein distance between two multivariate Gaussian distributions.
    """
    mean_diff = mean1 - mean2
    mean_dist_sq = np.dot(mean_diff, mean_diff)
    
    cov_mid = sqrtm(np.dot(np.dot(sqrtm(cov1), cov2), sqrtm(cov1)))
    if np.iscomplexobj(cov_mid):
        cov_mid = cov_mid.real
    cov_dist = cov1 + cov2 - 2 * cov_mid
    
    return mean_dist_sq + np.trace(cov_dist)