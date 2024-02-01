from toy_data import *
from analysis import *
from motif import *

from numpy.linalg import matrix_power
import networkx as nx

num_stick = 2
length_range = [0.0]
repeat = 1
up_dim = 10
bound = 1

savepath = "toy_image/"

nnnn = 10

plt.figure(figsize=(6,6))

for thelength in length_range:
    length = thelength
    x_range = (0, bound)
    y_range = (0, bound)
    z_range = (0, bound)
    # err_lst = []
    # kn_lst = []
    kappalst = []
    kappacylst = []

    kappaerlst = []
    kappacyerlst = []

    chain_un = []
    chain_er = []

    for iiii in range(repeat):
        # generate random points in the space

        # points = generate_random_sticks(num_stick, length, x_range, y_range, z_range)
        # pair_distance = stick_pairwise_distances(points)
        # W = 1 / pair_distance 
        # np.fill_diagonal(W, 0)

        dim = 3

        grids = generate_grid_points(num_stick, x_range, y_range, z_range)
        W = generate_adjacency_matrix_uniform(num_stick, 1)
        G = generate_graph_from_adj_matrix(W, grids)
        totpt = num_stick ** dim
        num_conn = np.sum(W) / 2
        p = (2 * num_conn) / ((totpt - 1) * totpt)

        # G_er = nx.erdos_renyi_graph(totpt, p)
        G_er = generate_fixed_edges_graph(totpt, num_conn)
        W_er = nx.adjacency_matrix(G_er).todense()
        
        # max_value = np.max(W)
        # standW = W / max_value
        # W = prob_bind(standW, 1)

        if iiii+1 == repeat:
            plt.figure()
            nx.draw(G,pos=grids)
            plt.savefig(f"{savepath}{dim}D_graph_un_{num_stick}_{repeat}_run={iiii}.png")

            plt.figure()
            nx.draw(G_er)
            plt.savefig(f"{savepath}{dim}D_graph_er_{num_stick}_{repeat}_run={iiii}.png")

            plt.figure()
            heatmap = sns.heatmap(W)
            plt.savefig(f"{savepath}{dim}D_conn_un_{num_stick}_{repeat}_run={iiii}.png")
            plt.clf()

            plt.figure()
            heatmap2 = sns.heatmap(W_er)
            plt.savefig(f"{savepath}{dim}D_conn_er_{num_stick}_{repeat}_run={iiii}.png")
            plt.clf()


        mc_un, mc_er = [], []
        top = 8
        for i in range(1, top):
            # count1 = count_chains_of_any_length(G, i)
            count1 = np.sum(matrix_power(W, i))
            # count2 = count_chains_of_any_length(G_er, i)
            count2 = np.sum(matrix_power(W_er, i))
            mc_un.append(count1)
            mc_er.append(count2)

        chain_un.append(mc_un)
        chain_er.append(mc_er)

        mus = motif_moment_chain(W, nnnn)
        mus2 = motif_moment_chain_sample(G, nnnn)
        print(f"mu max error: {np.max(abs(mus-mus2))}")
        kappa1 = kappa_chain_comb(mus, nnnn)
        kappa2 = kappa_chain(W, nnnn)
        print(f"kappa max error: {np.max(abs(kappa1-kappa2))}")
        munn_cy = motif_moment_cycle(W, nnnn)
        kappa_cy = kappa_cycle(W, nnnn)
        cycle_sanity_check(munn_cy, kappa_cy, kappa1)

        mus_er = motif_moment_chain(W_er, nnnn)
        mus2_er = motif_moment_chain_sample(G_er, nnnn)
        print(f"mu er max error: {np.max(abs(mus_er-mus2_er))}")
        kappa1_er = kappa_chain_comb(mus_er, nnnn)
        kappa2_er = kappa_chain(W_er, nnnn)
        print(f"kappa er max error: {np.max(abs(kappa1_er-kappa2_er))}")
        munn_cy_er = motif_moment_cycle(W_er, nnnn)
        kappa_cy_er = kappa_cycle(W_er, nnnn)
        cycle_sanity_check(munn_cy_er, kappa_cy_er, kappa1)

        kappalst.append(kappa2)
        kappacylst.append(kappa_cy)

        kappaerlst.append(kappa2_er)
        kappacyerlst.append(kappa_cy_er)

        print(mus)
        print(mus_er)
        print(kappa2)
        print(kappa2_er)

        
        # comp, err = isomap_test(pair_distance, up_dim)
        # kn = KneeLocator(comp, err, curve='convex', direction='decreasing')
        # err_lst.append(err)
        # kn_lst.append(kn.knee)

    # err_lst = np.array(err_lst)
    # the_err = np.mean(err_lst, axis=0)
    # mean_kn = np.mean(np.array(kn_lst))

    # plt.plot(comp, the_err, label=f"{length}={mean_kn}")

# plt.legend()
# plt.savefig(f"err_num_{num_stick}_repeat_{repeat}_bound_{bound}.png")

kappa1_mean = np.mean(np.array(kappalst), axis=0)
kappacy_mean = np.mean(np.array(kappacylst), axis=0)
kappa1_er_mean = np.mean(np.array(kappaerlst), axis=0)
kappacy_er_mean = np.mean(np.array(kappacyerlst), axis=0)

kappa1_mean = np.abs(kappa1_mean)
kappa1_er_mean = np.abs(kappa1_er_mean)

chain_un_mean = np.mean(np.array(chain_un), axis=0)
chain_er_mean = np.mean(np.array(chain_er), axis=0)

plottt = 1
if plottt == 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([i for i in range(1, len(kappa1_mean)+1)], kappa1_mean, marker='o', linestyle='-', label="un")
    ax.plot([i for i in range(1, len(kappa1_er_mean)+1)], kappa1_er_mean, marker='o', linestyle='-', label="er")
    ax.set_title(f"Kappa; Order: {nnnn}")
    ax.legend(loc='best')  # Explicitly using ax.legend()
    ax.set_yscale('log')  # Set the y-axis to a logarithmic scale
    fig.savefig(f"{savepath}1D_snn_unif_er_kappa_{num_stick}_{repeat}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([i for i in range(1, len(kappa1_mean))], np.abs([kappa1_mean[i]/kappa1_mean[i+1] for i in range(0, len(kappa1_mean)-1)]), marker='o', linestyle='-', label="un")
    ax.plot([i for i in range(1, len(kappa1_er_mean))], np.abs([kappa1_er_mean[i]/kappa1_er_mean[i+1] for i in range(0, len(kappa1_er_mean)-1)]), marker='o', linestyle='-', label="er")
    ax.set_title(f"Kappa; Order: {nnnn}")
    ax.legend(loc='best')  # Explicitly using ax.legend()
    ax.set_yscale('log')  # Set the y-axis to a logarithmic scale
    fig.savefig(f"{savepath}1D_snn_unif_er_kappa_decay_{num_stick}_{repeat}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([i for i in range(1, len(kappacy_mean)+1)], kappacy_mean, marker='o', linestyle='-', label="un")
    ax.plot([i for i in range(1, len(kappacy_er_mean)+1)], kappacy_er_mean, marker='o', linestyle='-', label="er")
    ax.set_title(f"Kappa_cy; Order: {nnnn}")
    ax.legend(loc='best')  
    fig.savefig(f"{savepath}1D_snn_unif_er_kappacy_{num_stick}_{repeat}.png")

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.hist(kappa1_er_mean, density=False)
    # ax.set_title(f"Kappa; Order: {nnnn}; {kappa1_er_mean}")
    # fig.savefig(f"snn_er_kappa_{num_stick}_{repeat}.png")

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.hist(kappacy_er_mean, density=False)
    # ax.set_title(f"Kappa_cy; Order: {nnnn}; {kappacy_er_mean}")
    # fig.savefig(f"snn_er_kappacy_{num_stick}_{repeat}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([i+1 for i in range(len(chain_un_mean))], chain_un_mean, marker='o', linestyle='-', label="un")
    ax.plot([i+1 for i in range(len(chain_er_mean))], chain_er_mean, marker='o', linestyle='-', label="er")
    ax.set_title(f"n-chain count")
    ax.set_xlabel("Length of Chain")
    ax.set_ylabel("Count")
    ax.legend(loc='best')  
    ax.set_yscale('log')  # Set the y-axis to a logarithmic scale
    fig.savefig(f"{savepath}1D_snn_unif_er_{num_stick}_{repeat}.png")