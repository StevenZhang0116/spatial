from motif import *

n_lst = [1e3]

p = 0.5
samnum = 100000
repeat = 1000

nnnn = 5

kappa1lst = []
kappacylst = []

for _ in range(repeat):
    N = int(n_lst[0])
    G = nx.erdos_renyi_graph(N, p)

    adjacency_matrix = nx.adjacency_matrix(G)
    W = adjacency_matrix.todense()

    mus = motif_moment_chain(W, nnnn)

    kappa1 = kappa_chain_comb(mus, nnnn)
    kappa2 = kappa_chain(W, nnnn)
    # print(f"kappa1: {kappa1}")
    # print(f"kappa2: {kappa2}")
    print(f"max: {np.max(abs(kappa1-kappa2))}")
    kappa1lst.append(kappa1)

    munn_cy = motif_moment_cycle(W, nnnn)
    kappa_cy = kappa_cycle(W, nnnn)
    kappacylst.append(kappa_cy)
    cycle_sanity_check(munn_cy, kappa_cy, kappa1)


kappa1_mean = np.mean(np.array(kappa1lst), axis=0)
kappacy_mean = np.mean(np.array(kappacylst), axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(kappa1_mean, density=False)
ax.set_title(f"Kappa; Order: {nnnn}")
fig.savefig(f"er_kappa_{N}_{repeat}_{p}.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(kappacy_mean, density=False)
ax.set_title(f"Kappa_cy; Order: {nnnn}")
fig.savefig(f"er_kappacy_{N}_{repeat}_{p}.png")
