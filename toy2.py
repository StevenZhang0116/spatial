from toy_data import *
from toy_analysis import *
from auxiliary import *

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
import os

repeat = 10
neuron_num = 100
num_points_per_dim = 1000
bound = [0, 6]
shift = 0.1
std = 0.01 # keep it fixed now
long_std = 0.5
mean_range = [2.5, 3.5]
diag_cov = np.diag([std, std, std])

defpath = "images"

# little redundant but for clarity
x = np.linspace(bound[0], bound[1], num_points_per_dim)
y = np.linspace(bound[0], bound[1], num_points_per_dim)
z = np.linspace(bound[0], bound[1], num_points_per_dim)

X, Y, Z = np.meshgrid(x, y, z)
sample_space = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

plt.figure(figsize=(6,6))

err_lst = []
kn_lst = []

for _ in range(repeat):
    dend_pdf, axon_pdf = [], []

    for _ in range(neuron_num):
        dend_mean = np.random.uniform(mean_range[0], mean_range[1], 3)

        # dend_dist = multivariate_normal(dend_mean, diag_cov)
        dend_dist = generate_equal_length_stick_structure(dend_mean, std, long_std)

        opt_shift = np.random.uniform(-shift, shift, 3)
        axon_mean = np.add(dend_mean, opt_shift)

        # axon_dist = multivariate_normal(axon_mean, diag_cov)
        axon_dist = generate_equal_length_stick_structure(axon_mean, std, long_std)

        # Monte Carlo Integration
        pdf1 = dend_dist.pdf(sample_space)
        dend_pdf.append(pdf1)
        pdf2 = axon_dist.pdf(sample_space)
        axon_pdf.append(pdf2)

    W = np.zeros([neuron_num, neuron_num])

    for i in range(neuron_num):
        for j in range(neuron_num):
            pdf1, pdf2 = dend_pdf[i], axon_pdf[j]
            product_of_densities = pdf1 * pdf2

            average_product_density = np.mean(product_of_densities)
            sample_space_volume = (bound[1] - bound[0]) ** 3  
            
            integral_value = average_product_density * sample_space_volume
            W[i][j] = integral_value

    C = np.cov(W)
    J = np.ones([neuron_num, neuron_num])
    D = J - C

    comp, err = isomap_test(D, 8)
    print(err)
    kn = KneeLocator(comp, err, curve='convex', direction='decreasing')
    print(kn.knee)

    err_lst.append(err)
    kn_lst.append(kn.knee)


err_lst = np.array(err_lst)
the_err = np.mean(err_lst, axis=0)
mean_kn = np.mean(np.array(kn_lst))

plt.plot(comp, the_err, label=f"mean_kn={mean_kn}")
plt.legend()
plt.savefig(f"{defpath}/toy2_test_neuron_{neuron_num}_repeat_{repeat}_std_{std}_longstd_{long_std}.png")

# delete_pycache(os.getcwd())
