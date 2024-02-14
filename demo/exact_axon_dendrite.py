from helper import *

# use exact information of synapses

if __name__ == "__main__":
    cell_table = pd.read_feather("./microns_cell_tables/pre_cell_table_microns_mm3.feather")
    synapse_table = pd.read_feather("./microns_cell_tables/synapse_table_microns_mm3.feather")

    neuron_type_lst = cell_table['cell_type'].unique()

    cells_index = cell_table['pt_root_id']
    nn = len(cells_index)

    conn = np.zeros((nn, nn))
    ttpp = np.zeros((nn, nn))
    binttpp = np.zeros((nn, nn))

    thesigma = 5
    savepath = "./images/"

    neuron_type_record = []
    ei_neuron_type_record = []

    n1 = "pre"
    n2 = "pre"

    for i in range(nn):
        print(f"i: {i}")
        for j in range(nn):
            cell_id1 = cells_index[i]
            cell_id2 = cells_index[j]
            if cell_id1 != cell_id2: 
                cnt += 1
                [soma1_loc, neuron1_type, pre1_loc, _, _] = extract_neuron_synaptic_info(cell_id1, n1, cell_table, synapse_table)
                [soma2_loc, neuron2_type, pre2_loc, _, _] = extract_neuron_synaptic_info(cell_id2, n2, cell_table, synapse_table)

                # search for neuron type (not just IE)
                neuron1_type_index = np.where(neuron_type_lst == neuron1_type)[0]
                neuron2_type_index = np.where(neuron_type_lst == neuron2_type)[0]
                ttpp[i,j] = neuron1_type_index + neuron2_type_index
                # check EI
                binttpp[i,j] = check_EI(neuron1_type) + check_EI(neuron2_type)

                if cnt == 1:
                    neuron_type_record.append(int(neuron1_type_index))
                    ei_neuron_type_record.append(check_EI(neuron1_type))

                # plot_double_neuron(soma1_loc, soma2_loc, pre1_loc, pre2_loc)s
                rec = 0

                # sample and generate sphere
                for index in range(pre1_loc.shape[0]):
                    sample_pt = pre1_loc[index, :]
                    dist_to_pt = pre2_loc - sample_pt
                    dist_to_pt_norm = np.linalg.norm(dist_to_pt, axis=1)

                    # sphere_filter = 20
                    # filter_norm = dist_to_pt_norm[dist_to_pt_norm < sphere_filter]
                    filter_norm = dist_to_pt_norm

                    if len(filter_norm) > 0:
                        gaussian_norm = gaussian_decay(filter_norm, A=1, mu=0, sigma=thesigma)
                        rec += np.sum(gaussian_norm)

            else:
                rec = 0

            conn[i,j] = rec

    np.save(f'exact_{n1}_{n2}_conn.npy', conn)
    np.save(f'exact_{n1}_{n2}_ttpp.npy', ttpp)

    reloadindex = 1
    if reloadindex == 1:
        conn = np.load("conn.npy")
        ttpp = np.load("ttpp.npy")

        [dim, re_err, X_transformed, kk] = isomap_test(conn, 20)
        
        plt.figure()
        conn_log = np.log10(conn + 1)  
        sns.heatmap(conn_log, cbar_kws={'label': 'Log 10 Scale'})
        plt.savefig(f"{savepath}conn_log_{thesigma}.png")

        if kk == 2:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  
            axs[0].scatter(X_transformed[:, 0], X_transformed[:, 1], c=neuron_type_record, cmap='viridis')
            axs[0].set_title("EI Categorization")
            axs[1].scatter(X_transformed[:, 0], X_transformed[:, 1], c=ei_neuron_type_record, cmap='viridis')
            axs[0].set_title("Neuron Type Categorization")
            plt.tight_layout()
            plt.savefig(f"{savepath}/x_trans_{thesigma}.png")

        plt.figure()
        plt.plot(dim,re_err,marker='o', linestyle='-')
        plt.axvline(x = kk, color = 'b', linestyle='--')
        plt.savefig(f"{savepath}isomap_result_{thesigma}.png")