# this script is used to recover/reconstruct plots from the output

import ast
import numpy as np 
import matplotlib.pyplot as plt 

filename = "opt2_longshift.txt"

tend_lst = []
knee_lst = []
errcnt = 0

# Open the file using a with statement
with open(filename, 'r') as file:
    cnt = 0
    try: 
        for line in file:
            res = line.strip()
            if cnt == 0:
                tend_lst.append(ast.literal_eval(res))
            else:
                knee_lst.append(int(res))
            cnt = abs(cnt - 1)
    except ValueError:
        errcnt += 1

mean_kn = np.mean(knee_lst)
tend_array = np.array(tend_lst)
tend = np.mean(tend_array, axis=0)
xx = [i for i in range(len(tend))]

plt.plot(xx, tend, label=f"mean_kn={mean_kn}")
plt.legend()
plt.savefig("result.png")