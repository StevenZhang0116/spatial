import os
import numpy as np
import re

import os
import numpy as np
import re

folder_path = './data/prepost'  
print(f"FilePath: {folder_path}")

all_files = os.listdir(folder_path)

npz_files = [file for file in all_files if file.endswith('.npz')]

categorized_data = {}

file_name_pattern = re.compile(r"(\d+P)_num_(\d+)_sigma_(\d+)_diffusion_(\d+)_repeat_(\d+)_(\w+).npz")

for file_name in npz_files:
    match = file_name_pattern.match(file_name)
    if match:
        neuron_type, num, sigma, diffusion, repeat, data_type = match.groups()
        
        category_key = (neuron_type, num, sigma, diffusion, repeat)
        
        if category_key not in categorized_data:
            categorized_data[category_key] = {}
        
        file_path = os.path.join(folder_path, file_name)
        with np.load(file_path) as data:
            categorized_data[category_key][data_type] = [data[key] for key in data]

optpc = 1

if optpc == 1:
    optpc_data = {}

    for category_key, data_dict in categorized_data.items():
        if "optpc" in data_dict:
            optpc_data[category_key] = data_dict["optpc"]

    for key in optpc_data.keys():
        print(key)
        data = optpc_data[key]
        # print(data)
        meanval = sum(arr.item() for arr in data) / len(data)
        print(meanval)

weight = 0

if weight == 1:
    weight_data = {}

    for category_key, data_dict in categorized_data.items():
        if "W" in data_dict:
            weight_data[category_key] = data_dict["W"]
