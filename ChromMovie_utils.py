#########################################################################
################## Krzysztof Banecki, WARSAW 2025 #######################
#########################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
from openmm.app import PDBxFile
import pyBigWig
import os
import seaborn as sns


############# Creation of mmcif and psf files #############
mmcif_atomhead = """data_nucsim
# 
_entry.id nucsim
# 
_audit_conform.dict_name       mmcif_pdbx.dic 
_audit_conform.dict_version    5.296 
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic 
# ----------- ATOMS ----------------
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.pdbx_PDB_ins_code 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z
"""

mmcif_connecthead = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
"""


def write_mmcif(points, cif_file_name='struct.cif', breaks=[], chroms=['A']):
    atoms = ''
    n = len(points)
    chrom_id = 0
    for i in range(0,n):
        x = points[i][0]
        y = points[i][1]
        try:
            z = points[i][2]
        except IndexError:
            z = 0.0
        atoms += f"ATOM {i+1} D CA . CHR {chroms[chrom_id]} {i+1} {i+1} ? {x:.3f} {y:.3f} {z:.3f}\n"
        if i in breaks:
            chrom_id += 1
            chrom_id = chrom_id%len(chroms)

    connects = ''
    chrom_id = 0
    j = 0
    for i in range(0,n-1):
        if i not in breaks:
            connects += f'C{j} covale CHR {chroms[chrom_id]} {i+1} CA CHR {chroms[chrom_id]} {i+2} CA\n'
            j += 1
        else:
            chrom_id += 1
            chrom_id = chrom_id%len(chroms)

    # Save files
    ## .pdb
    cif_file_content = mmcif_atomhead+atoms+mmcif_connecthead+connects

    with open(cif_file_name, 'w') as f:
        f.write(cif_file_content)


def mmcif2npy(positions_path):
    """
    Extracts numpy array (n,3) from cif file. 
    Positions are in nanometers hence multiplication by 10 to convert to Angstroms.
    """
    positions = PDBxFile(positions_path).positions
    m = len(positions)
    array = np.zeros((m, 3))
    for i, p in enumerate(positions):
        array[i, :] = [p[j].real*10 for j in range(3)] 
    return array


def dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance in R^3"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5  # faster than np.linalg.norm

def random_versor() -> np.ndarray:
    """Random versor"""
    x = np.random.normal()
    y = np.random.normal()
    z = np.random.normal()
    d = (x**2 + y**2 + z**2)**0.5
    return np.array([x/d, y/d, z/d])

def self_avoiding_random_walk(n: int, step: float = 1.0, bead_radius: float = 0.5, epsilon: float = 0.001, two_dimensions=False) -> np.ndarray:
    """Self avoiding random walk algorithm with the distance between consecutive beads equal to step 
    and with a condition that no beads should be closer than bead_radius to each other."""
    potential_new_step = [0, 0, 0]
    while True:
        points = [np.array([0, 0, 0])]
        for _ in range(n - 1):
            step_is_ok = False
            trials = 0
            while not step_is_ok and trials < 1000:
                potential_new_step = points[-1] + step * random_versor()
                if two_dimensions:
                    potential_new_step[2] = 0
                for j in points:
                    d = dist(j, potential_new_step)
                    if d < 2 * bead_radius - epsilon:
                        trials += 1
                        break
                else:
                    step_is_ok = True
            points.append(potential_new_step)
        points = np.array(points)
        return points


def align_self_avoiding_structures(structure_list: list) -> list:
    """
    Aligns structures from structure_list as that they do not overlap.
    Returns same structures with altered positions.
    """
    new_structure_list = [structure - structure.min(axis=0) for structure in structure_list]
    max_size = np.max([structure.max() for structure in new_structure_list])
    n = len(structure_list)
    cube_n = int(np.ceil(n**(1/3)))
    random_order = list(np.random.choice(n, n, replace=False))
    final_structure_list = []
    for i, structure in enumerate(new_structure_list):
        j = random_order.index(i)
        structure = structure.astype(np.float64)
        structure += np.array([max_size*(j%cube_n),
                               max_size*((j//cube_n)%cube_n), 
                               max_size*(j//cube_n**2)], dtype=np.float64)
        final_structure_list.append(structure)
    return final_structure_list

    
def extrapolate_points(points: np.array, n: int) -> np.array:
    """
    Extrapolates the points (m, 3) to new size (n, 3) preserving equal distances along the path created by points.
    """
    n1 = len(points)
    total_len = n1 - 1
    points_new = [points[0]]
    for i in range(1, n):
        curr_len = total_len * i / (n - 1)
        p1 = points[int(curr_len // 1)]
        if curr_len // 1 == n1 - 1:
            points_new.append(p1)
        else:
            p2 = points[int(curr_len // 1) + 1]
            alpha = curr_len % 1
            p = (p1[0] * (1 - alpha) + p2[0] * alpha, p1[1] * (1 - alpha) + p2[1] * alpha,
                 p1[2] * (1 - alpha) + p2[2] * alpha)
            points_new.append(p)
    return np.array(points_new)


def get_custom_force_formula(f_type: str="attractive", f_formula: str="harmonic", linear: bool=False, l_bound: float=1, u_bound: float=1, u_linear:float=1) -> str:
    """
    Creates a formula for the CustomBondForce function. 
    f_type: can be either "attractive" or "repulsive". 
    f_formula: can be either "harmonic" or "gaussian".
    linear: if True the formula will become linear after the u_linear value.
    l_bound: Marks the end of the repulsive force and the beginning of the flat bottom of the potential.
    u_bound: Marks the end of the flat bottom of the potential and the beginning of the attractive part. If l_bound==u_bound no flat bottom is used.
    u_linear: Marks the end of the attractive part of the potential and the beginning of the linear function. If u_bound==l_linear no linearization is used.
    Returns string with the desired force formula.
    """
    formula = ""
    if f_type == "attractive":
        if f_formula == "harmonic":
            if l_bound == u_bound == u_linear:
                formula = '(r-rl)^2'
            elif u_bound >= u_linear:
                formula = '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u))'
            else:
                formula = '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u)*step(r_u+d-r) + d*(2*r-d-2*r_u)*step(r-r_u-d))'
        elif f_formula == "gaussian":
            if l_bound == u_bound:
                formula = '(1-exp(-(r-r_l)^2/2/r_l^2))'
            else:
                formula = '((1-exp(-(r-r_l)^2/2/r_l^2))*step(r_l-r) + (1-exp(-(r-r_u)^2/2/r_l^2))*step(r-r_u))'
    elif f_type == "repulsive":
        formula = '(r-r_l)^2*step(r_l-r)'

    if formula == "":
        raise(Exception("Could not find a correct formula for a force."))
    return formula


def formula2lambda(formula: str, l_bound: float=1, u_bound: float=1, u_linear:float=1):
    """Creates a lambda function based on the formula and parameters."""
    if formula == '(r-rl)^2':
        return lambda r: (r-l_bound)**2
    elif formula == '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u))':
        return lambda r: (r-l_bound)**2 if r<l_bound else (r-u_bound)**2 if r>u_bound else 0
    elif formula == '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u)*step(r_u+d-r) + d*(2*r-d-2*r_u)*step(r-r_u-d))':
        d = u_linear - u_bound
        return lambda r: (r-l_bound)**2 if r<l_bound else d*(2*r-d-2*u_bound) if r>u_linear else (r-u_bound)**2 if r>u_bound else 0
    elif formula == '(1-exp(-(r-r_l)^2/2/r_l^2))':
        return lambda r: 1-np.exp(-(r-l_bound)**2/2/l_bound**2)
    elif formula == '((1-exp(-(r-r_l)^2/2/r_l^2))*step(r_l-r) + (1-exp(-(r-r_u)^2/2/r_l^2))*step(r-r_u))':
        return lambda r: (1-np.exp(-(r-l_bound)**2/2/l_bound**2)) if r<l_bound else (1-np.exp(-(r-u_bound)**2/2/l_bound**2)) if r>u_bound else 0
    elif formula == '(r-r_l)^2*step(r_l-r)':
        return lambda r: (r-l_bound)**2 if r<l_bound else 0
    else:
        raise(Exception(f"Unrecognized formula: {formula}"))
    

def resolution2text(resolution: int):
    """
    Takes the resolution (in base pairs) as input and creates a human readable version of it.
    For example resolution2text(1_000_000) returns "1Mb"
    """
    suffixes = ['', 'kb', 'Mb', 'Gb'] 
    
    # Find the appropriate suffix
    for i, suffix in enumerate(suffixes):
        if resolution < 1000:
            break
        resolution /= 1000.0
    
    formatted_num = f"{resolution:.1f}".rstrip('0').rstrip('.')
    result = f"{formatted_num}{suffix}"
    return result


def get_unique_chroms(contact_dfs: list) -> list:
    """
    Returns a single list of all unique chromosomes taken from columns 'chrom1' and 'chrom2' 
    of each of the data frames in the list 'contact_dfs'.
    """
    unique_values = set()
    for df in contact_dfs:
        values = pd.concat([df['chrom1'], df['chrom2']])
        unique_values.update(values.unique())
    unique_values = list(unique_values)
    unique_values.sort()
    return unique_values


def parse_cif_line(line):
    if not line.startswith("ATOM"):
        return None

    parts = line.strip().split()
    atom_id = int(parts[1])
    chromosome = parts[6]
    x, y, z = map(float, parts[-3:])

    return {
        'atom_id': atom_id,
        'chromosome': chromosome,
        'position': (x, y, z)
    }


def color_chrom_territories(path_cif: str, path_out:str):
    """
    Creates a .cmd file specified by path_out, to use in UCSF Chimera 
    for coloring of the residues of the structure in path_cif
    based on the compartment information from .bw file from path_bw
    """
    # Read cif structure file:
    with open(path_cif, 'r') as file:
        atoms = [parse_cif_line(line) for line in file if line.startswith("ATOM")]
    
    # Define color palette:
    palette = sns.color_palette("tab10", 23)
    palette = [f"{r:.2f},{g:.2f},{b:.2f}" for r,g,b in palette]
    n = len(palette)

    # Create commands for Chimera:
    chroms = []
    for atom in atoms:
        chroms.append(atom["chromosome"])
    chroms = np.unique(chroms)

    # Writing file:
    if os.path.exists(path_out):
        os.remove(path_out)
    with open(path_out, "a") as f:
        for i, chrom in enumerate(chroms):
            f.write(f"col {palette[i%n]} :.{chrom}\n")


def color_compartments(path_cif: str, path_bw: str, resolution: int, path_out:str):
    """
    Creates a .cmd file specified by path_out, to use in UCSF Chimera 
    for coloring of the residues of the structure in path_cif
    based on the compartment information from .bw file from path_bw
    """
    # Read bw compartment file:
    bw = pyBigWig.open(path_bw)

    # Read cif structure file:
    with open(path_cif, 'r') as file:
        atoms = [parse_cif_line(line) for line in file if line.startswith("ATOM")]
    
    # Create commands for Chimera:
    colors = []
    curr_atom_id = -1
    curr_chrom = atoms[0]["chromosome"]
    for atom in atoms:
        if atom["chromosome"] != curr_chrom:
            curr_chrom = atom["chromosome"]
            curr_atom_id = 0
        else:
            curr_atom_id += 1

        coord = int(curr_atom_id*resolution)
        comp_score = bw.values(atom["chromosome"], coord, coord+1)[0]
        atom_id = atom['atom_id']
        colors.append(("blue" if comp_score < 0 else "red" if comp_score > 0 else "grey", atom_id, atom['chromosome']))

    chimera_commands = []
    curr_color = colors[0]
    curr_end = colors[0][1]
    for color in colors[1:]:
        if color[0] == curr_color[0] and color[2] == curr_color[2]:
            curr_end = color[1]
        else:
            chimera_commands.append(f"color {curr_color[0]} :{curr_color[1]}-{curr_end}.{curr_color[2]}\n")
            curr_color = color
            curr_end = curr_color[1]

    # Writing file:
    if os.path.exists(path_out):
        os.remove(path_out)
    with open(path_out, "a") as f:
        f.write("color red\n")

        blue_command = "color blue "
        for com in chimera_commands:
            if "blue" in com:
                blue_command += com.split("blue ")[-1][:-1]+","
        f.write(blue_command[:-1] + "\n")

        grey_command = "color grey "
        for com in chimera_commands:
            if "grey" in com:
                grey_command += com.split("grey ")[-1][:-1]+","
        f.write(grey_command[:-1] + "\n")


def color_rnaseq(path_cif: str, path_tsv: str, resolution: int, path_out:str):
    """
    Creates a .cmd file specified by path_out, to use in UCSF Chimera 
    for coloring of the residues of the structure in path_cif
    based on the RNA-seq information from .tsv file from path_bw
    """
    # Read bw compartment file:
    df = pd.read_csv(path_tsv, header=None, sep="\t")
    df.columns = ["chrom", "pos"]

    # Read cif structure file:
    with open(path_cif, 'r') as file:
        atoms = [parse_cif_line(line) for line in file if line.startswith("ATOM")]
    
    # Create commands for Chimera:
    colors = []
    curr_atom_id = -1
    curr_chrom = atoms[0]["chromosome"]
    curr_df = df[df["chrom"]==curr_chrom]
    curr_hist = np.histogram(curr_df["pos"], bins=np.arange(0, 10_000_000_000, resolution))[0]
    curr_hist = [v if v>0 else 1 for v in curr_hist]
    curr_thresh = np.histogram(np.log10(curr_hist), bins=4)[1]
    for atom in atoms:
        if atom["chromosome"] != curr_chrom:
            curr_chrom = atom["chromosome"]
            curr_df = df[df["chrom"]==curr_chrom]
            curr_hist = np.histogram(curr_df["pos"], bins=np.arange(0, 10_000_000_000, resolution))[0]
            curr_hist = [v if v>0 else 1 for v in curr_hist]
            curr_thresh = np.histogram(np.log10(curr_hist), bins=4)[1]
            curr_atom_id = 0
        else:
            curr_atom_id += 1

        comp_score = np.log10(curr_hist[curr_atom_id])
        atom_id = atom['atom_id']
        colors.append(("blue" if comp_score <= 0 else \
                       "green" if comp_score <= curr_thresh[1] else \
                        "yellow" if comp_score <= curr_thresh[2] else \
                        "orange" if comp_score <= curr_thresh[3] else "red", atom_id, atom['chromosome']))

    chimera_commands = []
    curr_color = colors[0]
    curr_end = colors[0][1]
    for color in colors[1:]:
        if color[0] == curr_color[0] and color[2] == curr_color[2]:
            curr_end = color[1]
        else:
            chimera_commands.append(f"color {curr_color[0]} :{curr_color[1]}-{curr_end}.{curr_color[2]}\n")
            curr_color = color
            curr_end = curr_color[1]

    # Writing file:
    if os.path.exists(path_out):
        os.remove(path_out)
    with open(path_out, "a") as f:
        f.write("color red\n")
        for color in ["orange", "yellow", "green", "blue"]:
            color_command = f"color {color} "
            for com in chimera_commands:
                if color in com:
                    color_command += com.split(f"{color} ")[-1][:-1]+","
            f.write(color_command[:-1] + "\n")

            