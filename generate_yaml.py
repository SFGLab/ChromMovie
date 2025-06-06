import yaml
import numpy as np


# Default configuration with descriptions
CONFIG_PARAMETERS = {
    'general':{
        'input': {
            'value': 'examples/example1_cell_cycle',
            'description': 'Folder containing input scHi-C contacts in csv format. If \'None\' simulated scHi-C maps are going to be used.'
        },
        'output': {
            'value': 'results',
            'description': 'Output folder for storing simulation results'
        },
        'genome': {
            'value': 'mm10',
            'description': 'Genome assembly of the input data. Currently supported assemblies: hg19, hg38, mm10, GRCm39.'
        },
        'pdf_report': {
            'value': True,
            'description': 'Whether to save the simulation diagnostics in a pdf format.'
        },
        'remove_problematic': {
            'value': False,
            'description': 'A flag indicating whether at each resolution round problematic contacts that the simulation was unable to resolve, should be removed.'
        }
    },

    'simulation': {
        'platform': {
            'value': 'OpenCL',
            'description': 'Available platoforms: CPU, CUDA and OpenCL.'
        },
        'resolutions': {
            'value': '5,2',
            'description': 'Resolutions to be used for hierarchical modeling. Expected to be in the form of comma separated integer of float numbers in the units of Mb.'
        },
        'N_steps': {
            'value': 100,
            'description': 'Number of simulation steps to take at every resolution'
        },
        'burnin': {
            'value': 0,
            'description': 'Number of simulation steps before starting collecting the simulatin diagnostic data'
        },
        'MC_step': {
            'value': 1,
            'description': 'Simulation diagnostic data is going to be collected every MC_step'
        },
        'sim_step': {
            'value': 20,
            'description': 'The simulation step of Langevin integrator'
        },
        
    },

    'forcefield': {
        'ev_formula': {
            'value': 'harmonic',
            'description': 'Type of the Excluded Volume (EV) repulsion. Available types: harmonic'
        },
        'ev_min_dist': {
            'value': 1.0,
            'description': 'Excluded Volume (EV) minimal distance'
        },
        'ev_coef': {
            'value': 50.0,
            'description': 'Excluded Volume (EV) force coefficient'
        },
        'ev_coef_evol': {
            'value': True,
            'description': 'Enable or disable the changing EV coefficient value.\nIf True the coefficient will start as 0 at the beginning of the simulation and reach ev_coef at the end.\nIf False the coefficient will have stable value of ev_coef.'
        },

        'bb_formula': {
            'value': 'harmonic',
            'description': 'Type of the Backbone (BB) potential. Available types: harmonic, gaussian'
        },
        'bb_opt_dist': {
            'value': 1.0,
            'description': 'Backbone (BB) optimal distance'
        },
        'bb_lin_thresh': {
            'value': 2.0,
            'description': 'Backbone (BB) distance after which the potential grows linearly.\nMust be strictly greater than bb_opt_dist.\nOnly applicable if bb_formula is harmonic'
        },
        'bb_coef': {
            'value': 1000.0,
            'description': 'Backbone (BB) force coefficient'
        },
        'bb_coef_evol': {
            'value': False,
            'description': 'Enable or disable the changing BB coefficient value.\nIf True the coefficient will start as 0 at the beginning of the simulation and reach bb_coef at the end.\nIf False the coefficient will have stable value of bb_coef.'
        },
        

        'sc_formula': {
            'value': 'harmonic',
            'description': 'Type of the Single cell contact (SC) potential. Available types: harmonic, gaussian'
        },
        'sc_opt_dist': {
            'value': 1.0,
            'description': 'Single cell contact (SC) optimal distance'
        },
        'sc_lin_thresh': {
            'value': 2.0,
            'description': 'Single cell contact (SC) distance after which the potential grows linearly.\nMust be strictly greater than sc_opt_dist.\nOnly applicable if sc_formula is harmonic'
        },
        'sc_coef': {
            'value': 100.0,
            'description': 'Single cell contact (SC) force coefficient'
        },
        'sc_coef_evol': {
            'value': False,
            'description': 'Enable or disable the changing SC coefficient value.\nIf True the coefficient will start as 0 at the beginning of the simulation and reach sc_coef at the end.\nIf False the coefficient will have stable value of sc_coef.'
        },

        'ff_formula': {
            'value': 'harmonic',
            'description': 'Type of the Frame force (FF) potential. Available types: harmonic, gaussian'
        },
        'ff_opt_dist': {
            'value': 1.0,
            'description': 'Frame force (FF) optimal distance'
        },
        'ff_lin_thresh': {
            'value': 2.0,
            'description': 'Frame force (FF) distance after which the potential grows linearly.\nMust be strictly greater than ff_opt_dist.\nOnly applicable if ff_formula is harmonic'
        },
        'ff_coef': {
            'value': 100.0,
            'description': 'Frame force (FF) force coefficient'
        },
        'ff_coef_evol': {
            'value': False,
            'description': 'Enable or disable the changing FF coefficient value.\nIf True the coefficient will start as 0 at the beginning of the simulation and reach ff_coef at the end.\nIf False the coefficient will have stable value of ff_coef.'
        }
    }
}


def merge_with_defaults(user_config):
    """Merge user-specified parameters with default configuration."""
    merged_config = {}
    descriptions = {}
    
    for section, params in CONFIG_PARAMETERS.items():
        merged_config[section] = {}
        descriptions[section] = {}
        
        for key, details in params.items():
            # Use user value if available, otherwise default
            if section in user_config and key in user_config[section]:
                merged_config[section][key] = user_config[section][key]
            else:
                merged_config[section][key] = details['value']
            
            # Always add descriptions from defaults
            descriptions[section][key] = details['description']
    
    return {'descriptions': descriptions, **merged_config}


def generate_yaml_config(output_file, user_config=None):
    """Generate a YAML configuration file with user-specified and default parameters."""
    if user_config is None:
        user_config = {}
    
    final_config = merge_with_defaults(user_config)
    
    # Write to YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(final_config, yaml_file, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration file '{output_file}' generated successfully.")


if __name__ == "__main__":
    # Example of user-specified parameters:
    user_specified_config = {
        'general': {
            'input': "examples/example3_cell_cycle_sc",
            'output': 'results/tests/test_00',
            'genome': 'mm10',
            'pdf_report': False
        },
        'simulation': {
            'resolutions': '5,2,0.5,0.2,0.1',
            'burnin': 99
        },
        'forcefield': {
            'ev_coef': 100.0,
            'ev_coef_evol': True,
            'bb_coef': 100.0,
            'sc_lin_thresh': 1.5,
            'sc_coef': 50.0,
            'ff_lin_thresh': 1.5,
            'ff_coef': 50.0
        }
    }
    
    #preparing configs for frame force study:
    # ff_coefs = np.linspace(-3, 3, 100)
    # ff_coefs = [10**v for v in ff_coefs]
    # for i, ff_coef in enumerate(ff_coefs):
    #     user_specified_config['forcefield']['ff_coef'] = round(float(ff_coef), 6)
    #     user_specified_config['general']['output'] = f'results/tests/test_{str(i).zfill(3)}'
    #     generate_yaml_config(f'data/cell_cycle/configs/config_{str(i).zfill(3)}.yaml', user_specified_config)
    
    #preparing configs for frame force study:
    # n = 15
    # for i in range(n):
    #     user_specified_config['general']['input'] = f"examples/example4_{str(i+1).zfill(2)}"
    #     user_specified_config['general']['output'] = f'results/tests_benchmark/test_{str(i).zfill(2)}'
    #     generate_yaml_config(f'data/cell_cycle/configs/config_{str(i).zfill(3)}.yaml', user_specified_config)
    
    for i in [1,5,10,25,50,100,200]:
        user_specified_config['general']['input'] = f"examples/example5_{str(i).zfill(3)}"
        user_specified_config['general']['output'] = f'results/tests_benchmark2/test_{str(i).zfill(3)}'
        generate_yaml_config(f'data/cell_cycle/configs/config_{str(i).zfill(3)}.yaml', user_specified_config)

