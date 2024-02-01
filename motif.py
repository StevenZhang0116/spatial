import numpy as np
import networkx as nx
from scipy.special import factorial
import itertools
import matplotlib.pyplot as plt
import random
import concurrent.futures
import time
import os


def eigen_power(W, n):
    """
    Use eigenvalue decomposition to do the matrix power calculation: using np.linalg.matrix_power would easily out of stack
    """
    eigenvalues, eigenvectors = np.linalg.eig(W)
    D_power_n = np.diag(eigenvalues ** n)
    W_power_n = eigenvectors @ D_power_n @ np.linalg.inv(eigenvectors)
    return W_power_n


def count_motif(graph, motif):
    """
    Count the occurrences of a specific motif in the graph, based on structural pattern 
    """
    motif_count = 0
    for sub_nodes in itertools.combinations(graph.nodes(), len(motif.nodes())):
        subg = graph.subgraph(sub_nodes)
        if nx.is_isomorphic(subg, motif):
            motif_count += 1
    return motif_count


def motif_moment_chain(W, nmax, ind=0):
    """
    Packing
    """
    def motif_moment_test(W, n):
        """
        Calculate \mu_n motif moment [Text above Equation 4] 
        Using eigen decomposition for efficiency with large n
        """
        assert W.shape[0] == W.shape[1]
        N = W.shape[0]
        if ind == 0:
            W_power_n = np.linalg.matrix_power(W, n)
        else: 
            W_power_n = eigen_power(W, n)
        thesum = np.sum(W_power_n) / pow(N, n+1)
        return thesum

    return np.array([motif_moment_test(W, n) for n in range(1, nmax+1)])


def motif_moment_cycle(W, nmax, ind=0):
    """
    Packing
    """
    def motif_moment_cycle_test(W, n):
        """
        Calculate motif moments for n-cycles [Equation 19]
        """
        assert n >= 1
        assert W.shape[0] == W.shape[1]
        N = W.shape[0]
        if ind == 0:
            W_power_n = np.linalg.matrix_power(W, n)
        else: 
            W_power_n = eigen_power(W, n)
        stepI = np.trace(W_power_n)
        return pow(N, -n) * stepI

    return [motif_moment_cycle_test(W, n) for n in range(1, nmax+1)]


def sample_chain(G, n):
    """
    Iteration
    """
    sampled_nodes = [random.choice(list(G.nodes())) for _ in range(n + 1)]
    chain_exists = all(G.has_edge(
        sampled_nodes[i], sampled_nodes[i + 1]) for i in range(n))
    return 1 if chain_exists else 0


def motif_moment_chain_sample(G, nmax, m=10000, num_threads=4):
    def motif_moment_sample_test(G, n, m, num_threads=4):
        """
        Calculate \mu_n motif moment by lcoal sampling of connectivity [Appendix G]
        Notice here the input is G [networkx graph object] instead of W -- transform in advance
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(sample_chain, G, n) for _ in range(m)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return np.mean(results)
    return np.array([motif_moment_sample_test(G, n, m) for n in range(1, nmax+1)])


def kappa_chain(W, nmax, ind=0):
    """
    Packing
    """
    
    def calculate_kappa_test(W, n):
        """
        Calculate the motif cumulant kappa_n for a given adjacency matrix W, number of nodes N, and motif size n [Equation 5]
        """
        N = W.shape[0]
        if n >= 2:
            e = np.ones((N, 1)) / np.sqrt(N)
            theta = np.eye(N) - e @ e.T
            W_circle_theta = theta @ W
            if ind == 0:
                W_circle_theta_power = np.linalg.matrix_power(W_circle_theta, n-1)
            else:
                W_circle_theta_power = eigen_power(W_circle_theta, n-1)
            kappa_n = (1 / N**n) * (e.T @ W @ W_circle_theta_power @ e)
            return kappa_n.item()

        elif n == 1:
            return 1/(N ** 2) * np.sum(W)

    return [calculate_kappa_test(W, n) for n in range(1, nmax+1)]


def kappa_cycle(W, nmax, ind=0):
    """
    Packing
    """

    def kappa_cycle_test(W, n):
        """
        Calculate the cycle motif cumulants [Text above Equation 21]
        """
        N = W.shape[0]
        e = np.ones((N, 1)) / np.sqrt(N)
        theta = np.eye(N) - e @ e.T
        stepI = theta @ W

        if ind == 0:
            stepII = np.linalg.matrix_power(stepI, n)
        else: 
            stepII = eigen_power(stepI, n)

        stepIII = np.trace(stepII)
        return pow(N, -n) * stepIII

    return [kappa_cycle_test(W, n) for n in range(1, nmax+1)]

def cycle_sanity_check(mus, kappas_cc, kappas):
    """
    Cycle motif power/cumulants sanity check [Equation 20]
    """
    orders = len(kappas_cc) + 1
    errlst = []
    for n in range(1, orders):
        permutation = generate_compositions(n)
        this_mu = mus[n-1]
        this_kappa = kappas_cc[n-1]
        hyp = 0 # hypothetical cycle moment
        stepI = 0 
        for perm in permutation:
            prod = 1
            t = len(perm)
            for ind in perm:
                ind = int(ind)
                prod *= kappas[ind-1]
            stepI += prod * n / t
        hyp =  stepI + this_kappa
        errlst.append(abs(this_mu-hyp))
    mean_err = np.mean(errlst)
    print(f"Cycle Sanity Check: {mean_err}")



def generate_compositions(n):
    """
    Generate all compositions (ordered partitions) of a given integer n
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[1]]

    compositions = []
    for i in range(1, n+1):
        for rest in generate_compositions(n-i):
            compositions.append([i] + rest)

    return compositions


def kappa_chain_comb(mus, nmax):
    """
    Combinatorical way to calculate the cumulants; iterative solver [Equation 4]
    """
    assert len(mus) == nmax
    kappas = np.zeros(len(mus))
    kappas[0] = mus[0]  # \mu_1 = \kappa_1
    for it in range(2, len(mus)+1):
        perm = generate_compositions(it)
        result_holder = []
        for term in perm[:-1]:  # the iterative unknown parameter will always be the last
            term_res = []
            for kk in term:
                kk = int(kk)
                if kk < it:
                    term_res.append(kappas[kk-1])
            res = np.prod(np.array(term_res))
            result_holder.append(res)
        kappas[it-1] = mus[it-1] - np.sum(np.array(result_holder))

    # kappas = [float(f"{x:.10f}") for x in kappas]
    return kappas


def laplace_filter(index, s):
    """
    Return the result after nodal filters (exponential or decaying-osciallatory filter)
    """
    if index == 1:
        alpha = 0.2

        def exp_fil(s):
            return 1/(s+alpha)
        return exp_fil(s)
    # decaying-oscillatory filter [Equation J2]
    elif index == 2:
        alpha = 0.2
        v = 2*np.pi/7

        def os_fil(s):
            return (s+alpha)/((s+alpha)**2 + v**2)
        return os_fil(s)


def cal_transfer_func(W, s_range, index=1, dispt=1000):
    """
    Calculating network transfer function by solving the differential equation [Equation 3]
    """
    ss = generate_uniform_complex(s_range, dispt)
    result = []
    for s in ss:
        N = W.shape[0]
        B = np.ones((N, 1)) / np.sqrt(N)
        C = np.ones((N, 1)) / np.sqrt(N)
        hs = laplace_filter(index, s)
        stepI = np.eye(N) - hs * W
        stepII = np.linalg.inv(stepI)
        stepIII = B * hs
        stepIV = C.T
        res = stepIV @ stepII @ stepIII
        result.append(res.item())
    return result


def cal_transfer_func_motif(W, kappa_lst, s_range, index=1, dispt=1000):
    """
    Calculate network transfer function using motif
    """
    ss = generate_uniform_complex(s_range, dispt)
    result = []
    for s in ss:
        N = W.shape[0]
        max_ord = len(kappa_lst)
        stepI = 0
        for i in range(len(kappa_lst)):
            n = i+1
            kappa_n = kappa_lst[i]
            hs = laplace_filter(index, s)
            hsn = hs ** n
            stepI += (N**n) * kappa_n * hsn
        stepII = 1 - stepI
        stepIII = 1 / stepII
        res = stepIII * hs
        result.append(res)
    return result


def generate_uniform_complex(s_range, dispt):
    """
    Generate uniformly distributed complex numbers.
    """
    real_part = np.linspace(s_range[0], s_range[1], dispt)
    imag_part = np.linspace(s_range[0], s_range[1], dispt)
    real_grid, imag_grid = np.meshgrid(real_part, imag_part)
    complex_grid = real_grid + 1j * imag_grid
    return complex_grid.flatten()