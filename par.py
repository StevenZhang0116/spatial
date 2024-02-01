import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
import os
import time
from scipy.io import savemat

from toy_data import *
from analysis import *
from auxiliary import * 
from calmetric import *
from motif import *

np.random.seed(20010116)


def process_iteration(jjj, repeat, neuron_num, num_points_per_dim, bound, shift, short_std, long_std, mean_range, step, sample_space, nnnn, defpath, closed, metric_choice):
    print(f"Thread: {jjj}, {long_std}, {short_std}, {mean_range}")

    dend_pdf, axon_pdf = [], []
    mean1lst, mean2lst = [], []
    cov1lst, cov2lst = [], []

    for _ in range(neuron_num):
        dend_mean = np.random.uniform(mean_range[0], mean_range[1], 3)

        # dend_dist = multivariate_normal(dend_mean, diag_cov)
        [dend_dist, mean1, cov1] = generate_equal_length_stick_structure(dend_mean, short_std, long_std)
        mean1lst.append(mean1)
        cov1lst.append(cov1)

        opt_shift = np.random.uniform(-shift, shift, 3)
        axon_mean = np.add(dend_mean, opt_shift)

        # axon_dist = multivariate_normal(axon_mean, diag_cov)
        [axon_dist, mean2, cov2] = generate_equal_length_stick_structure(axon_mean, short_std, long_std) # another direction
        mean2lst.append(mean2)
        cov2lst.append(cov2)

        # Monte Carlo Integration

        if closed == 0:
            res1 = dend_dist.pdf(sample_space)
            dend_pdf.append(res1)
            res2 = axon_dist.pdf(sample_space)
            axon_pdf.append(res2)
            print(f"Sanity Check: {np.sum(res2) * (step ** 3)}")

    # connectivity matrix
    W = np.zeros((neuron_num, neuron_num))
    # bhattacharyya distance
    BDm = np.zeros((neuron_num, neuron_num))
    # wasserstein metric
    WDm = np.zeros((neuron_num, neuron_num))

    # ==== axon and dendrite are "the same" ====
    axon_pdf = dend_pdf 
    mean1lst = mean2lst
    cov1lst = cov2lst

    if closed == 0: 
        for i in range(neuron_num):
            for j in range(neuron_num):
                integral_value = np.sum(dend_pdf[i] * axon_pdf[j]) * (step ** 3)
                W[i, j] = integral_value
                BDm[i, j] = bhattacharyya_distance(mean1lst[i], cov1lst[i], mean2lst[j], cov2lst[j])
                WDm[i, j] = wasserstein_distance(mean1lst[i], cov1lst[i], mean2lst[j], cov2lst[j]) 

    elif closed == 1:
        for i in range(neuron_num):
            for j in range(neuron_num):
                muprod, covprod = combine_gaussians(mean1lst[i], cov1lst[i], mean2lst[j], cov2lst[j])
                print(f"{muprod}, {covprod}")
                new_dist = multivariate_normal(muprod, covprod)
                intpres = new_dist.pdf(sample_space)
                overlap = np.sum(intpres) * (step ** 3)
                W[i, j] = overlap
                if i == j:
                    print(f"overlap: {overlap}")

    
    J = np.ones([neuron_num, neuron_num])

    # connData = {'connectivity': W}
    # path = "./data/"
    # savemat(f'{path}{neuron_num}_{short_std}_{long_std}_{jjj}_connectivity.mat', connData)
    
    W = clamp_matrix(W / W[1,1]) # standardize, maximum val is 1
    is_symmetric(W)

    # covariance 
    if metric_choice == 1: 
        C = np.cov(W)
        D = J - C
    # volumetric
    elif metric_choice == 0:
        D = J - W

    W_old = W

    W = prob_bind(W, 1) 
    np.fill_diagonal(W, 0) # no self connection

    pp1 = np.sum(W) / (W.shape[0] ** 2)
    print(f"PP1: {pp1}")

    if jjj % 10 == 0: 
        heatmap = sns.heatmap(W)
        plt.title(f"Prob: {pp1}")
        plt.savefig(f"{defpath}/conn_{neuron_num}_{mean_range[0]}_{mean_range[1]}_longstd_{long_std}_std_{short_std}_{jjj}.png")
        plt.clf()

    # calculate motif-related quantities
    mus = motif_moment_chain(W, nnnn)
    print(f"mus: {mus}")
    G = generate_graph_from_adj_matrix(W)
    mus2 = motif_moment_chain_sample(G, nnnn)
    print(f"mus2: {mus2}")

    print(f"max mu diff: {np.max(abs(mus-mus2))}")

    kappa1 = kappa_chain_comb(mus, nnnn)
    print(f"kappa1: {kappa1}")
    kappa2 = kappa_chain(W, nnnn)
    print(f"kappa2: {kappa2}")

    print(f"max kappa diff: {np.max(abs(kappa1-kappa2))}")

    munn_cy = motif_moment_cycle(W, nnnn)
    kappa_cy = kappa_cycle(W, nnnn)
    print(f"kappa_cy: {kappa_cy}")
    cycle_sanity_check(munn_cy, kappa_cy, kappa1)

    # run spectrum on connectivity matrix
    [eigenval, newnec, within_ratio, unique_ratio] = eigenspectrum(W)


    # cw = cal_comm_weight_matrix(D)
    # print(f"Comm Weight Matrix: {cw}")

    # biD = binaryize_mat(D, 0.9)
    # print(f"Binaryized Matrix: {biD}")

    # mod = cal_modularity(biD)
    # print(f"Modularity: {mod}")

    # smw = cal_smallwordness(biD, neuron_num)
    # print(f"Small Worldness: {smw}")

    X, err_isomap = isomap_test(W_old, 2)

    kn_isomap = KneeLocator(X, err_isomap, curve='convex', direction='decreasing').knee

    err = {
        "isomap": err_isomap
    }

    kn = {
        "isomap": kn_isomap
    }

    return jjj, err, kn, kappa1, kappa_cy, pp1, eigenval
