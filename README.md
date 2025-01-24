# ChromMovie
ChromMovie: a molecular dynamics approach at simultaneous modeling of chromatin conformation changes from multiple sets of single-cell Hi-C data.

![ChromMovie_idea](https://github.com/user-attachments/assets/3a4af3e4-7db9-45cc-8eb6-a5c4696ae5f7)

ChromMovie is an Openmm based molecular dynamics simulation model for modeling 3D chromatin structure evolution up to the chromosome level and at a single cell resolution. Employs hierarchical modeling scheme and can be used with either CPU or GPU acceleration.

![k562_chr12](https://github.com/user-attachments/assets/1a07ff34-edd8-4749-a73b-455ed3017c15)

## Installation

ChromMovie software was primarily tested on Unix-based systems and with Python version 3.10.0. Before running ChromMovie please install the required packages listed in `requirements.txt`.

## Input Data

The input to ChromMovie consists of the ordered list of single cell interaction data. The data should reflect the cellular process of interest (cell cycle, cell maturation, etc.). The data are expected to be placed in a separate folder `input`. The files are expected to be in alphabetical order (see `examples`). Each file should contain the single cell contacts in `.csv` format with or without the header. First few rows of an example file may look like this:

```text
chrom1,start1,end1,chrom2,start2,end2
chr3,128668059,128668108,chr3,159257092,159257218
chr3,50107348,50107498,chr3,51161425,51161468
chr3,53660133,53660244,chr3,54171138,54171157
```

If you are running ChromMovie for the first time, we encourage you to try some of our examples in `examples` folder.

## Parameter specification

The simulation parameters used by ChromMovie are stored in YAML file. Example specification is provided in `config.yaml` file. The path to the YAML configuration file is a main input parameter of ChromMovie script.

## Running ChromMovie

After preparing the data and YAML configuration file, ChromMovie algorithm can be used with the following command:

```python
python3 -m ChromMovie --input config.yaml
```

## Output

Simulation results will be saved in the folder `output`. A typical output of ChromMovie simulation consists of the following data:

* `config.yaml` - Configuration file used for this particular simulation. This is a copy of the input configuration file.
* `energy.csv` - File containing diagnostic information about Energy and Temperature of the simulation.
* `simulation_reportXXX.pdf` - `.pdf` file with diagnostic information and figures created for each resolution XXX used in the simulation. Only generated if `pdf_report==True`.
* `struct_00_init.cif` - `.cif` file containing merged information about the initial structure for each simulation frame. Typically initial structure is a self-avoiding random walk.
* `struct_XX_resYYY_init.cif` - `.cif` file containing merged information about the initial structure at each resolution of the simulation (after initial energy minimization).
* `struct_XX_resYYY_ready.cif` - `.cif` file containing merged information about the final structure at each resolution of the simulation.
* `frames_cif` - Folder containing all of the structures in `.cif` format for different steps and frames of the simulation and the lowest of the resolutions of the simulation. 
* `frames_npy` - Folder containing contact heatmaps in `.npy` format created from the input contact data for the lowest of the resolutions of the simulation.


## Simulation Arguments

The YAML configuration file for ChromMovie uses a number of configurable parameters.
List of parameter default values recommended by the authors and their descriptions are provided in the table below:

| Argument Name | Type          | Value         | Description   |
|---------------|---------------|---------------|---------------|
|input          |str            |examples/example1_cell_cycle|Folder containing input scHi-C contacts in csv format. If 'None' simulated scHi-C maps are going to be used.|
|output         |str            |results        |Output folder for storing simulation results|
|n              |int            |10             |Number of scHi-C frames to be generated. Only applicable if input==None.|
|m              |int            |100            |Size of scHi-C frames to be generated (number of beads). Only applicable if input==None.|
|n_contacts     |int            |50             |Number of contacts to be drawn from each in silico structure. Only applicable if input==None.|
|artificial_structure|int            |1              |Index of the in silico structure to be generated as simulation input.|
|genome         |str            |mm10           |Genome assembly of the input data. Currently supported assemblies: hg19, hg38, mm10, GRCm39.|
|chrom          |str            |chr1           |Chromosome to be modeled. The input files are going to be filtered for intra-chromosomal contacts within this chromosome.|
|pdf_report     |bool           |True           |Whether to save the simulation diagnostics in a pdf format.|
|remove_problematic|bool           |False           |A flag indicating whether at each resolution round problematic contacts that the simulation was unable to resolve, should be removed.|
|platform       |str            |OpenCL         |Available platoforms: CPU, CUDA, OpenCL, HIP|
|resolutions    |str            |5,2            |Resolutions to be used for hierarchical modeling. Expected to be in the form of comma separated integer of float numbers in the units of Mb.|
|N_steps        |int            |100            |Number of simulation steps to take at every resolution|
|burnin         |int            |0              |Number of simulation steps before starting collecting the simulation diagnostic data|
|MC_step        |int            |1              |Simulation diagnostic data is going to be collected every MC_step|
|sim_step       |int            |20             |The simulation step of Langevin integrator|
|ev_formula     |str            |harmonic       |Type of the Excluded Volume (EV) repulsion. Available types: harmonic|
|ev_min_dist    |float          |1.0            |Excluded Volume (EV) minimal distance|
|ev_coef        |float          |1000.0         |Excluded Volume (EV) force coefficient|
|ev_coef_evol   |bool           |False          |Enable or disable the changing EV coefficient value.|
|bb_formula     |str            |harmonic       |Type of the Backbone (BB) potential. Available types: harmonic, gaussian|
|bb_opt_dist    |float          |1.0            |Backbone (BB) optimal distance|
|bb_lin_thresh  |float          |2.0            |Backbone (BB) distance after which the potential grows linearly.|
|bb_coef        |float          |1000.0         |Backbone (BB) force coefficient|
|bb_coef_evol   |bool           |False          |Enable or disable the changing BB coefficient value.|
|sc_formula     |str            |harmonic       |Type of the Single cell contact (SC) potential. Available types: harmonic, gaussian|
|sc_opt_dist    |float          |1.0            |Single cell contact (SC) optimal distance|
|sc_lin_thresh  |float          |2.0            |Single cell contact (SC) distance after which the potential grows linearly.|
|sc_coef        |float          |1000.0         |Single cell contact (SC) force coefficient|
|sc_coef_evol   |bool           |False          |Enable or disable the changing SC coefficient value.|
|ff_formula     |str            |harmonic       |Type of the Frame force (FF) potential. Available types: harmonic, gaussian|
|ff_opt_dist    |float          |1.0            |Frame force (FF) optimal distance|
|ff_lin_thresh  |float          |2.0            |Frame force (FF) distance after which the potential grows linearly.|
|ff_coef        |float          |1000.0         |Frame force (FF) force coefficient|
|ff_coef_evol   |bool           |False          |Enable or disable the changing FF coefficient value.|

## Copyrights

The software is freely distributed under the GNU GENERAL PUBLIC LICENSE (GPL-3.0).
