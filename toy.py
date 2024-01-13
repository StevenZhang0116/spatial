from toy_data import *
from toy_analysis import *

num_stick = 200
length_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
repeat = 1000
up_dim = 10
bound = 4

plt.figure(figsize=(6,6))

for thelength in length_range:
    length = thelength
    x_range = (0, bound)
    y_range = (0, bound)
    z_range = (0, bound)
    err_lst = []
    kn_lst = []

    for _ in range(repeat):
        sticks = generate_random_sticks(num_stick, length, x_range, y_range, z_range)
        pair_distance = parallel_pairwise_distances(sticks)

        comp, err = isomap_test(pair_distance, up_dim)
        # normalized
        # err = [elem / num_stick for elem in err]
        kn = KneeLocator(comp, err, curve='convex', direction='decreasing')
        err_lst.append(err)
        kn_lst.append(kn.knee)
        print(kn.knee)

    err_lst = np.array(err_lst)
    the_err = np.mean(err_lst, axis=0)
    mean_kn = np.mean(np.array(kn_lst))
    print(the_err)

    plt.plot(comp, the_err, label=f"{length}={mean_kn}")
    # plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

plt.legend()
plt.savefig(f"err_num_{num_stick}_repeat_{repeat}_bound_{bound}.png")