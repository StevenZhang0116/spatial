import os
import numpy as np
import re
import matplotlib.pyplot as plt

folder_path = './data/'
subfolder_path = 'prepost'  
print(f"FilePath: {folder_path}{subfolder_path}")

all_files = os.listdir(folder_path+subfolder_path)

npz_files = [file for file in all_files if file.endswith('.npz')]

categorized_data = {}

file_name_pattern = re.compile(r'(\d+P|BC|NGC)_num_(\d+)_sigma_(\d+)_diffusion_(\d+)_repeat_(\d+)_([a-z]+)\.npz')


for file_name in npz_files:
    match = file_name_pattern.match(file_name)
    if match:
        neuron_type, num, sigma, diffusion, repeat, data_type = match.groups()
        
        category_key = (neuron_type, num, sigma, diffusion, repeat)
        
        if category_key not in categorized_data:
            categorized_data[category_key] = {}
        
        file_path = os.path.join(folder_path, subfolder_path, file_name)
        with np.load(file_path) as data:
            categorized_data[category_key][data_type] = [data[key] for key in data]

optpc = 1

if optpc == 1:
    optpc_data = {}
    mean_pc_data = {}

    for category_key, data_dict in categorized_data.items():
        if "optpc" in data_dict:
            optpc_data[category_key] = data_dict["optpc"]

    for key in optpc_data.keys():
        print(key)
        data = optpc_data[key]
        meanval = sum(arr.item() for arr in data) / len(data)
        print(meanval)
        mean_pc_data[key] = meanval

weight = 0

if weight == 1:
    weight_data = {}

    for category_key, data_dict in categorized_data.items():
        if "W" in data_dict:
            weight_data[category_key] = data_dict["W"]
            print(weight_data)

loss = 1

if loss == 1:
    loss_data = {}

    for category_key, data_dict in categorized_data.items():
        if "loss" in data_dict:
            loss_data[category_key] = data_dict["loss"]
    
    num = len(loss_data.keys())

    num_rows = int(np.ceil(num / 2)) 
    num_cols = 2 if num > 1 else 1  

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))
    axs = np.array(axs)  

    if num > 1:
        axs = axs.flatten()

    for i, (key, data) in enumerate(loss_data.items()):
        coll = np.array(data)
        mean_loss = np.mean(coll, axis=0)
        std_loss = np.std(coll, axis=0)
        
        if num > 1:
            ax = axs[i]
        else:
            ax = axs  
        
        ax.plot(mean_loss, label=f'Mean Loss for {key}', marker='o', linestyle='dashed')
        ax.fill_between(range(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        ax.set_title(f'{key}: {np.round(mean_pc_data[key], 3)}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        # ax.legend()

    plt.tight_layout()
    plt.savefig(f"{subfolder_path}_image.png")
        

    