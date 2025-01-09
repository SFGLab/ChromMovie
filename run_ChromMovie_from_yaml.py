import yaml
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
    if main_config["input"] is not None:
        # TODO
        pass
    elif main_config["artificial_structure"] == 1:
        structure_set = get_structure_01(frames=main_config["n"], m=main_config["m"])
        heatmaps = get_hicmaps(structure_set, n_contacts=main_config["n_contacts"])
    elif main_config["artificial_structure"] == 2:
        structure_set = get_structure_02(frames=main_config["n"], m=main_config["m"])
        heatmaps = get_hicmaps(structure_set, n_contacts=main_config["n_contacts"])
    
    # Saving the ground truth structure:
    path_init = os.path.join(main_config["output"], "ground_truth")
    if os.path.exists(path_init): shutil.rmtree(path_init)
    if not os.path.exists(path_init): os.makedirs(path_init)
    for frame in range(main_config["n"]):
        save_points_as_pdb(structure_set[frame], os.path.join(path_init, f"frame_{str(frame).zfill(3)}.pdb"))

    # Run ChromMovie
    f_params = [force_config["ev_min_dist"], force_config["ev_coef"],
                force_config["bb_opt_dist"], force_config["bb_coef"],
                force_config["sc_opt_dist"], force_config["sc_coef"],
                force_config["ff_opt_dist"], force_config["ff_coef"]
                ]
    
    md = MD_simulation(heatmaps, output_path=main_config["output"], 
                       N_steps=sim_config["N_steps"], burnin=sim_config["burnin"], MC_step=sim_config["MC_step"], 
                       platform=sim_config["platform"], force_params=f_params)
    md.run_pipeline(write_files=True, plots=True, sim_step=sim_config["sim_step"])


if __name__ == "__main__":
    main('config.yaml')