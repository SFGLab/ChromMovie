#################################################################
########### Krzysztof Banecki, Warsaw 2025 ######################
#################################################################

import yaml
import os
from ChromMovie import *


def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def validate_input_yaml_parameters(m_config, s_config, f_config):
    """Validate input YAML parameters."""
    # TODO
    return 0

def main(config_path: str='config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Access configuration values
    main_config = config['general']
    sim_config = config['simulation']
    force_config = config['forcefield']
    
    # Creating input numpy heatmaps
    heatmaps = None
    contact_dfs = None
    if main_config["input"] is not None:
        files = os.listdir(main_config["input"])
        files = [file for file in files if file.endswith(".csv")]
        files.sort()
        contact_dfs = [pd.read_csv(os.path.join(main_config["input"], file)) for file in files]
        contact_dfs = [df[(df.iloc[:, 0]==main_config["chrom"]) & (df.iloc[:, 3]==main_config["chrom"])] for df in contact_dfs]
        for df in contact_dfs:
            df["x"] = [int((s+e)/2) for s, e in zip(df.iloc[:,1], df.iloc[:,2])]
            df["y"] = [int((s+e)/2) for s, e in zip(df.iloc[:,4], df.iloc[:,4])]
            df["chrom"] = [main_config["chrom"]]*df.shape[0]
        contact_dfs = [df[["chrom", "x", "y"]] for df in contact_dfs]
    elif main_config["artificial_structure"] == 1:
        structure_set = get_structure_01(frames=main_config["n"], m=main_config["m"])
        heatmaps = get_hicmaps(structure_set, n_contacts=main_config["n_contacts"])
    elif main_config["artificial_structure"] == 2:
        structure_set = get_structure_02(frames=main_config["n"], m=main_config["m"])
        heatmaps = get_hicmaps(structure_set, n_contacts=main_config["n_contacts"])
    else:
        raise(Exception("Neither input nor artificial structure were correctly specified."))
    
    # Saving the ground truth structure (if applicable):
    if main_config["input"] is None:
        path_init = os.path.join(main_config["output"], "ground_truth")
        if os.path.exists(path_init): shutil.rmtree(path_init)
        if not os.path.exists(path_init): os.makedirs(path_init)
        for frame in range(main_config["n"]):
            write_mmcif(structure_set[frame], os.path.join(path_init, f"frame_{str(frame).zfill(3)}.cif"))

    # Run ChromMovie
    md = MD_simulation(main_config=main_config, sim_config=sim_config, heatmaps=heatmaps, contact_dfs=contact_dfs, output_path=main_config["output"], 
                       N_steps=sim_config["N_steps"], burnin=sim_config["burnin"], MC_step=sim_config["MC_step"], 
                       platform=sim_config["platform"], force_params=force_config)
    md.run_pipeline(write_files=True, sim_step=sim_config["sim_step"])


if __name__ == "__main__":
    main('config.yaml')