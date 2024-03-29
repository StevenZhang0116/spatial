from helper import *

# tossing across different neuronal structure
# sample arbitrary neuronal structure from a given type with different placement

if __name__ == "__main__":
    cell_table = pd.read_feather("./microns_cell_tables/pre_cell_table_microns_mm3.feather")
    synapse_table = pd.read_feather("./microns_cell_tables/synapse_table_microns_mm3.feather")

    # permute hyperparameters
    neuron_type_lst = cell_table['cell_type'].unique()
    sigmalst = [5, 10, 15, 20]
    samplenum_lst = [20, 50, 100]
    diffusionlst = [100, 200, 500]
    repeatlst = [10]

    parameter_combinations = list(product(neuron_type_lst, sigmalst, samplenum_lst, diffusionlst, repeatlst))
    print("Neuron Type || Sigma || NeuronNum || Diffusion || Repeat")

    for cnt in range(len(parameter_combinations)):
        [neuron_type, thesigma, sample_num, diffusion, repeat] = parameter_combinations[cnt]
        print(f"{parameter_combinations[cnt]}; {(cnt+1)/len(parameter_combinations)*100}%")

        photopath = "./images/"
        datapath = "./data"

        choice = 1
        if choice == 1:
            subpath = "prepre"
        elif choice == 2:
            subpath = "prepost"

        # neuron list from the selected type
        neuron_lst = cell_table[cell_table['cell_type'] == neuron_type]

        conn_lst = []
        loss_lst = []
        optpc_lst = []
        violation_lst = []

        for _ in range(repeat):
            # neuron synapse location
            neuron_location_lst = []
            neuron_location_post_lst = [] # consider postsynaptic as well
            # iterate 
            for _ in range(sample_num):
                # pick one 
                random_row = neuron_lst.sample(n=1)
                soma_loc = np.array(random_row['pt_position'])
                # generate random shift of soma (and all its synapse points)
                coordinate = np.random.randint(0, diffusion, size=3)
                # "new" soma location: only shift, no rotation
                tran_soma_loc = soma_loc + coordinate
                # which neuron to cross-reference
                neuron_indicator = np.array(random_row['pt_root_id'])[0]
                [_, _, syn_loc, syn_size, _] = extract_neuron_synaptic_info(neuron_indicator, "pre", cell_table, synapse_table)
                tran_syn_loc = syn_loc + coordinate
                ro_new_syn_loc = rotate_around_y_with_pivot(tran_syn_loc, tran_soma_loc, 0)
                neuron_location_lst.append(ro_new_syn_loc)
                if choice == 2:
                    [_, _, syn_loc2, syn_size2, _] = extract_neuron_synaptic_info(neuron_indicator, "post", cell_table, synapse_table)
                    # what if no synapse has been found
                    if len(syn_loc2) > 0:
                        tran_syn_loc2 = syn_loc2 + coordinate
                        ro_new_syn_loc2 = rotate_around_y_with_pivot(tran_syn_loc2, tran_soma_loc, 0)
                        neuron_location_post_lst.append(ro_new_syn_loc2)
                    else:
                        neuron_location_post_lst.append([])
                
            W = np.zeros((sample_num, sample_num))
            
            # construct connectivity matrix
            for i in range(sample_num):
                for j in range(sample_num):
                    if i != j:
                        neuron1_syn = neuron_location_lst[i]
                        if choice == 1:
                            # comapre between pre-pre
                            neuron2_syn = neuron_location_lst[j]
                        elif choice == 2:
                            # compare between pre-post
                            neuron2_syn = neuron_location_post_lst[j]
                        rec = 0
                        # ensure >=1 synapses could be found
                        if len(neuron1_syn) > 0 and len(neuron2_syn) > 0:
                            for index in range(neuron1_syn.shape[0]):
                                sample_pt = neuron1_syn[index, :]
                                dist_to_pt = neuron2_syn - sample_pt
                                dist_to_pt_norm = np.linalg.norm(dist_to_pt, axis=1) 
                                # no filter
                                gaussian_norm = gaussian_decay(dist_to_pt_norm, A=1, mu=0, sigma=thesigma)
                                rec += np.sum(gaussian_norm)
                        W[i,j] = rec
            
            # column_sums = W.sum(axis=0) + np.finfo(float).tiny
            # norm_W = W / column_sums
            # D = 1 - norm_W

            conn_lst.append(W)

            violation_result = triangle_inequality_violation(W)

            [dim, reconstruction_error, X_transformed, optimal_dimension] = isomap_test(W, 10)

            optpc_lst.append(optimal_dimension)
            loss_lst.append(reconstruction_error)
            violation_lst.append(violation_result)

            print(f"vio: {violation_result}; opt: {optimal_dimension};")
            
        np.savez(f'{datapath}/{subpath}/{neuron_type}_num_{sample_num}_sigma_{thesigma}_diffusion_{diffusion}_repeat_{repeat}_W.npz', *conn_lst)
        np.savez(f'{datapath}/{subpath}/{neuron_type}_num_{sample_num}_sigma_{thesigma}_diffusion_{diffusion}_repeat_{repeat}_optpc.npz', *optpc_lst)
        np.savez(f'{datapath}/{subpath}/{neuron_type}_num_{sample_num}_sigma_{thesigma}_diffusion_{diffusion}_repeat_{repeat}_loss.npz', *loss_lst)
        np.savez(f'{datapath}/{subpath}/{neuron_type}_num_{sample_num}_sigma_{thesigma}_diffusion_{diffusion}_repeat_{repeat}_violation.npz', *violation_lst)



    

        