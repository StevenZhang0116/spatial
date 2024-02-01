from par import * # multitherading module

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

repeat = 100
neuron_num = 100
num_points_per_dim = 200
bound = [0, 10]
step = (bound[1] - bound[0])/num_points_per_dim
shift = 0
short_std = 0.05
# long_std_lst = [i for i in np.arange(0.1, 0.5, 0.05)]
long_std_lst = [0.4]

# specify the total deviation of mean range
# divlst = [i for i in np.arange(0.05, 0.5, 0.05)]
divlst = [0.5]
ubb = 2

x = np.linspace(bound[0], bound[1], num_points_per_dim)
y = np.linspace(bound[0], bound[1], num_points_per_dim)
z = np.linspace(bound[0], bound[1], num_points_per_dim)

X, Y, Z = np.meshgrid(x, y, z)
sample_space = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# choice of metric being used
metric_choice = 0 

# number of motif power/cumulants that need to be calculated
nnnn = 10

# how to calculate the overlap volume
closed = 0

defpath = "images"

methods = ['Isomap'] # after check, only consider isomap at this moment

bind_prob = []

for long_std in long_std_lst: 
    for div in divlst: 
        # secondary parameter calculation
        meanpt = (bound[0]+bound[1])/2
        mean_range = [meanpt-div, meanpt+div]

        # data stack
        all_err = np.zeros((repeat, ubb - 1, len(methods)))
        all_kn = np.zeros((repeat, len(methods)))
        all_kappa = np.zeros((repeat, nnnn))
        all_kappa_cy = np.zeros((repeat, nnnn))
        bind_pp = np.zeros((repeat, 1))
        eigenvals = []

        for repeat_ind in range(repeat):
            jjj, err, kn, kappa1, kappa_cy, pp1, eigen = process_iteration(repeat_ind, repeat, neuron_num, num_points_per_dim, bound, shift, short_std, long_std, mean_range, step, sample_space, nnnn, defpath, closed, metric_choice)
            all_err[jjj, :, 0] = err["isomap"]
            all_kn[jjj, 0] = kn["isomap"]
            all_kappa[jjj, :] = kappa1
            all_kappa_cy[jjj, :] = kappa_cy
            bind_pp[jjj, 0] = pp1
            eigenvals.append(eigen)

        eigenvals = np.array(eigenvals)

        # plt.figure(figsize=(20, 10))  

        # for i, method in enumerate(methods):
        #     the_err = np.mean(all_err[:, :, i], axis=0)
        #     mean_kn = np.mean(all_kn[:, i], axis=0)
        #     X_label = [i for i in range(len(the_err))]
        #     plt.plot(X_label, the_err, linewidth=3, markersize=12, label=f"{method}={mean_kn}")
        #     plt.legend()

        # plt.savefig(f"{defpath}/toy_method_compare_neuron_{neuron_num}_repeat_{repeat}_std_{short_std}_longstd_{long_std}_mean_{mean_range[0]}_{mean_range[1]}.png")

        kappa1_mean = np.mean(np.array(all_kappa), axis=0)
        kappacy_mean = np.mean(np.array(all_kappa_cy), axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(kappa1_mean, density=False)
        ax.set_title(f"Kappa")
        fig.savefig(f"{defpath}/kappa_{neuron_num}_{long_std}_{short_std}.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(kappacy_mean, density=False)
        ax.set_title(f"Kappa Cycle")
        fig.savefig(f"{defpath}/kappacy_{neuron_num}_{mean_range[0]}_{mean_range[1]}_longstd_{long_std}_std_{short_std}.png")

        bind_prob.append(np.mean(bind_pp))

plt.figure()
mean_ev = np.mean(eigenvals, axis=0)
skewness = stats.skew(mean_ev)
plt.hist(mean_ev, bins=50)
plt.title(f"Average Eigenvalue Spectrum: skewness={round(skewness,3)}; prob={round(np.mean(bind_prob),3)}")
plt.savefig(f"./spec_image/{neuron_num}_{mean_range[0]}_{mean_range[1]}_longstd_{long_std}_std_{short_std}.png")

# model = LinearRegression()
# bind_prob = np.array(bind_prob).reshape(-1, 1)
# divlst = np.array(divlst).reshape(-1,1)
# model.fit(divlst, bind_prob)

# def predict_x_given_y(y, model):
#     slope = model.coef_[0]
#     intercept = model.intercept_
#     return (y - intercept) / slope

# predicted_x = predict_x_given_y(0.379636, model)

# plt.figure()
# plt.plot(divlst, bind_prob)
# plt.plot(divlst, model.predict(divlst), color='b', linestyle='-.')
# plt.axhline(y = 0.379636, color='r', linestyle='-') 
# print(bind_prob)
# plt.title(f"Intersection point: x={predicted_x}")
# plt.savefig("testimg_div.png")