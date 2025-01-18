import yaml

# Default configuration with descriptions
CONFIG_PARAMETERS = {
    'general':{
        'input': {
            'value': None,
            'description': 'Folder containing input scHi-C contacts in csv format. If \'None\' simulated scHi-C maps are going to be used.'
        },
        'output': {
            'value': 'results',
            'description': 'Output folder for storing simulation results'
        },
        'n': {
            'value': 10,
            'description': 'Number of scHi-C frames to be generated. Only applicable if input==None.'
        },
        'm': {
            'value': 100,
            'description': 'Size of scHi-C frames to be generated (number of beads). Only applicable if input==None.'
        },
        'n_contacts': {
            'value': 50,
            'description': 'Number of contacts to be drawn from each in silico structure. Only applicable if input==None.'
        },
        'artificial_structure': {
            'value': 1,
            'description': 'Index of the in silico structure to be generated as simulation input.\nOnly applicable if input==None. Available structure indices: 1, 2'
        },
        'genome': {
            'value': 'mm10',
            'description': 'Genome assembly of the input data. Currently supported assemblies: hg19, hg38, mm10, GRCm39.'
        },
        'chrom': {
            'value': 'chr1',
            'description': 'Chromosome to be modeled. The input files are going to be filtered for intra-chromosomal contacts within this chromosome.'
        },
        'pdf_report': {
            'value': True,
            'description': 'Whether to save the simulation diagnostics in a pdf format.'
        }
    },

    'simulation': {
        'platform': {
            'value': 'OpenCL',
            'description': 'Available platoforms: CPU, CUDA, OpenCL, HIP'
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
            'value': 5,
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
            'value': 1000.0,
            'description': 'Excluded Volume (EV) force coefficient'
        },
        'ev_coef_evol': {
            'value': False,
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
            'value': 1000.0,
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
            'value': 1000.0,
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
            'input': "examples/example1_cell_cycle",
            'output': 'results',
            'chrom': 'chr1'
        },
        'simulation': {
            'ev_min_dist': 2
        },
        'forcefield': {
            
        }
    }
    
    generate_yaml_config('config.yaml', user_specified_config)
