#################################################################
########### Krzysztof Banecki, Warsaw 2025 ######################
#################################################################

import yaml
import os
import pandas as pd
import shutil
from ChromMovie_core import *


def load_config(config_file: str):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def validate_input_yaml_parameters(m_config: dict, s_config: dict, f_config: dict) -> None:
    """Validate input YAML parameters."""
    # Main config validation
    if m_config["input"] is not None:
        if not os.path.isdir(m_config["input"]):
            raise ValueError("Simulation 'input' parameter is not a valid directory")
        files = os.listdir(m_config["input"])
        files = [file for file in files if file.endswith(".csv")]
        if len(files) == 0:
            raise ValueError("No csv files were found in the input directory")
    
    if m_config["genome"] not in [file.split(".")[0] for file in os.listdir("chrom_sizes")]:
        raise ValueError(f"Cannot find chromosome sizes for genome assembly: "+str(m_config["genome"]))

    if not isinstance(m_config["pdf_report"], (bool)):
        raise TypeError("Invalid 'pdf_report' parameter type. Expected 'bool', got "+str(type(m_config["input"])))
    
    # Simulation config validation
    try:
        resolutions = [float(x) for x in s_config["resolutions"].split(",")]
    except ValueError:
        print("Incorrect 'resolutions' format. Epected are comma separated int or float numbers")
    if any([res<1e-6 for res in resolutions]):
        raise ValueError("Resolution have to be greater than 1bp")
    if any([res>1e5 for res in resolutions]):
        raise ValueError("One of the resolutions is now greater than 100Gb. Please provide resolutions in the units of Mb")

    if not isinstance(s_config["N_steps"], (int)):
        raise TypeError("Invalid 'N_steps' parameter type. Expected 'int', got "+str(type(s_config["input"])))
    if s_config["N_steps"] <= 0:
        raise ValueError("N_steps must be greater than 0")

    if not isinstance(s_config["burnin"], (int)):
        raise TypeError("Invalid 'burnin' parameter type. Expected 'int', got "+str(type(s_config["input"])))
    if s_config["burnin"] < 0:
        raise ValueError("burnin must be greater or equal 0")
    
    if not isinstance(s_config["MC_step"], (int)):
        raise TypeError("Invalid 'MC_step' parameter type. Expected 'int', got "+str(type(s_config["input"])))
    if s_config["MC_step"] <= 0:
        raise ValueError("MC_step must be greater than 0")
    
    if not isinstance(s_config["sim_step"], (int)):
        raise TypeError("Invalid 'sim_step' parameter type. Expected 'int', got "+str(type(s_config["input"])))
    if s_config["sim_step"] <= 0:
        raise ValueError("sim_step must be greater than 0")
    
    # Force field config validation
    if f_config["ev_formula"] not in ["harmonic"]:
        raise ValueError("Could not recognize ev_formula. Currently available formulas: 'harmonic'")
    if not isinstance(f_config["ev_min_dist"], (int, float)):
        raise TypeError("Invalid 'ev_min_dist' parameter type. Expected 'int' or 'float', got "+str(type(f_config["ev_min_dist"])))
    if f_config["ev_min_dist"] <= 0:
        raise ValueError("ev_min_dist must be greater than 0")
    if not isinstance(f_config["ev_coef"], (int, float)):
        raise TypeError("Invalid 'ev_coef' parameter type. Expected 'int' or 'float', got "+str(type(f_config["ev_coef"])))
    if f_config["ev_coef"] <= 0:
        raise ValueError("ev_coef must be greater than 0")
    if not isinstance(f_config["ev_coef_evol"], (bool)):
        raise TypeError("Invalid 'ev_coef_evol' parameter type. Expected 'bool', got "+str(type(f_config["ev_coef_evol"])))
    
    if f_config["bb_formula"] not in ["harmonic", "gaussian"]:
        raise ValueError("Could not recognize bb_formula. Currently available formulas: 'harmonic', 'gaussian'")
    if not isinstance(f_config["bb_opt_dist"], (int, float)):
        raise TypeError("Invalid 'bb_opt_dist' parameter type. Expected 'int' or 'float', got "+str(type(f_config["bb_opt_dist"])))
    if f_config["bb_opt_dist"] <= 0:
        raise ValueError("bb_opt_dist must be greater than 0")
    if not isinstance(f_config["bb_lin_thresh"], (int, float)):
        raise TypeError("Invalid 'bb_lin_thresh' parameter type. Expected 'int' or 'float', got "+str(type(f_config["bb_lin_thresh"])))
    if f_config["bb_lin_thresh"] <= 0:
        raise ValueError("bb_lin_thresh must be greater than 0")
    if f_config["bb_lin_thresh"] < f_config["bb_opt_dist"]:
        raise ValueError("bb_lin_thresh must be greater or equal to bb_opt_dist. For no linear part set those parameters to equal")
    if not isinstance(f_config["bb_coef"], (int, float)):
        raise TypeError("Invalid 'bb_coef' parameter type. Expected 'int' or 'float', got "+str(type(f_config["bb_coef"])))
    if f_config["bb_coef"] <= 0:
        raise ValueError("bb_coef must be greater than 0")
    if not isinstance(f_config["bb_coef_evol"], (bool)):
        raise TypeError("Invalid 'bb_coef_evol' parameter type. Expected 'bool', got "+str(type(f_config["bb_coef_evol"])))
    
    if f_config["sc_formula"] not in ["harmonic", "gaussian"]:
        raise ValueError("Could not recognize sc_formula. Currently available formulas: 'harmonic', 'gaussian'")
    if not isinstance(f_config["sc_opt_dist"], (int, float)):
        raise TypeError("Invalid 'sc_opt_dist' parameter type. Expected 'int' or 'float', got "+str(type(f_config["sc_opt_dist"])))
    if f_config["sc_opt_dist"] <= 0:
        raise ValueError("sc_opt_dist must be greater than 0")
    if not isinstance(f_config["sc_lin_thresh"], (int, float)):
        raise TypeError("Invalid 'sc_lin_thresh' parameter type. Expected 'int' or 'float', got "+str(type(f_config["sc_lin_thresh"])))
    if f_config["sc_lin_thresh"] <= 0:
        raise ValueError("sc_lin_thresh must be greater than 0")
    if f_config["sc_lin_thresh"] < f_config["sc_opt_dist"]:
        raise ValueError("sc_lin_thresh must be greater or equal to sc_opt_dist. For no linear part set those parameters to equal")
    if not isinstance(f_config["sc_coef"], (int, float)):
        raise TypeError("Invalid 'sc_coef' parameter type. Expected 'int' or 'float', got "+str(type(f_config["sc_coef"])))
    if f_config["sc_coef"] <= 0:
        raise ValueError("sc_coef must be greater than 0")
    if not isinstance(f_config["sc_coef_evol"], (bool)):
        raise TypeError("Invalid 'sc_coef_evol' parameter type. Expected 'bool', got "+str(type(f_config["sc_coef_evol"])))

    if f_config["ff_formula"] not in ["harmonic", "gaussian"]:
        raise ValueError("Could not recognize ff_formula. Currently available formulas: 'harmonic', 'gaussian'")
    if not isinstance(f_config["ff_opt_dist"], (int, float)):
        raise TypeError("Invalid 'ff_opt_dist' parameter type. Expected 'int' or 'float', got "+str(type(f_config["ff_opt_dist"])))
    if f_config["ff_opt_dist"] <= 0:
        raise ValueError("ff_opt_dist must be greater than 0")
    if not isinstance(f_config["ff_lin_thresh"], (int, float)):
        raise TypeError("Invalid 'ff_lin_thresh' parameter type. Expected 'int' or 'float', got "+str(type(f_config["ff_lin_thresh"])))
    if f_config["ff_lin_thresh"] <= 0:
        raise ValueError("ff_lin_thresh must be greater than 0")
    if f_config["ff_lin_thresh"] < f_config["ff_opt_dist"]:
        raise ValueError("ff_lin_thresh must be greater or equal to ff_opt_dist. For no linear part set those parameters to equal")
    if not isinstance(f_config["ff_coef"], (int, float)):
        raise TypeError("Invalid 'ff_coef' parameter type. Expected 'int' or 'float', got "+str(type(f_config["ff_coef"])))
    if f_config["ff_coef"] <= 0:
        raise ValueError("ff_coef must be greater than 0")
    if not isinstance(f_config["ff_coef_evol"], (bool)):
        raise TypeError("Invalid 'ff_coef_evol' parameter type. Expected 'bool', got "+str(type(f_config["ff_coef_evol"])))


def heatmap2df(heatmap: np.ndarray, chrom: str, chrom_size: int) -> pd.DataFrame:
    df = pd.DataFrame(columns=["chrom", "x", "y"])
    m = heatmap.shape[0]
    bin_w = chrom_size/m
    for i in range(m):
        for j in range(i, m):
            if heatmap[i, j] > 0:
                for _ in range(int(heatmap[i, j])):
                    pos1 = int(bin_w*i + np.random.rand()*bin_w)
                    pos2 = int(bin_w*j + np.random.rand()*bin_w)
                    df.loc[df.shape[0], :] = [chrom, pos1, pos2]
    return df


def identify_header(path, n=10, th=0.9):
    """Function that specifies what header parameter to use automatically. Helps with reading csv files with header or without"""
    df1 = pd.read_csv(path, header='infer', nrows=n)
    df2 = pd.read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return 'infer' if sim < th else None


def ChromMovie_from_yaml(config_path: str='config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Access configuration values
    main_config = config['general']
    sim_config = config['simulation']
    force_config = config['forcefield']

    # Parameter validation
    validate_input_yaml_parameters(main_config, sim_config, force_config)
    
    # Creating results folder and copying config file
    if not os.path.exists(main_config["output"]):
        os.makedirs(main_config["output"])
    shutil.copy(config_path, os.path.join(main_config["output"], os.path.basename(config_path)))

    # Creating input pandas DataFrames with contact information:
    files = os.listdir(main_config["input"])
    files = [file for file in files if file.endswith(".csv")]
    files.sort()
    contact_dfs = [pd.read_csv(os.path.join(main_config["input"], file), header=identify_header(os.path.join(main_config["input"], file))) for file in files]
    chrom_regex = r'chr([1-9]|1[0-9]|2[0-2]|X|Y)(?:-P|-M)?'
    contact_dfs = [df[(df.iloc[:, 0].str.fullmatch(chrom_regex)) & (df.iloc[:, 3].str.fullmatch(chrom_regex))] for df in contact_dfs]
    for df in contact_dfs:
        df["chrom1"] = df.iloc[:, 0]
        df["pos1"] = [int((s+e)/2) for s, e in zip(df.iloc[:,1], df.iloc[:,2])]
        df["chrom2"] = df.iloc[:, 3]
        df["pos2"] = [int((s+e)/2) for s, e in zip(df.iloc[:,4], df.iloc[:,4])]
    contact_dfs = [df[["chrom1", "pos1", "chrom2", "pos2"]] for df in contact_dfs]
    
    # Run ChromMovie
    md = MD_simulation(main_config=main_config, sim_config=sim_config, contact_dfs=contact_dfs, output_path=main_config["output"], 
                       N_steps=sim_config["N_steps"], burnin=sim_config["burnin"], MC_step=sim_config["MC_step"], 
                       platform=sim_config["platform"], force_params=force_config)
    md.run_pipeline(write_files=True, sim_step=sim_config["sim_step"])


if __name__ == "__main__":
    ChromMovie_from_yaml('config.yaml')