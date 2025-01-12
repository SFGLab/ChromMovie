#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

import numpy as np
from tqdm import tqdm

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


def write_mmcif(points,cif_file_name='LE_init_struct.cif'):
    atoms = ''
    n = len(points)
    for i in range(0,n):
        x = points[i][0]
        y = points[i][1]
        try:
            z = points[i][2]
        except IndexError:
            z = 0.0
        atoms += ('{0:} {1:} {2:} {3:} {4:} {5:} {6:}  {7:} {8:} '
                '{9:} {10:.3f} {11:.3f} {12:.3f}\n'.format('ATOM', i+1, 'D', 'CA',\
                                                            '.', 'ALA', 'A', 1, i+1, '?',\
                                                            x, y, z))

    connects = ''
    for i in range(0,n-1):
        connects += f'C{i+1} covale ALA A {i+1} CA ALA A {i+2} CA\n'

    # Save files
    ## .pdb
    cif_file_content = mmcif_atomhead+atoms+mmcif_connecthead+connects

    with open(cif_file_name, 'w') as f:
        f.write(cif_file_content)

def dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Mierzy dystans w przestrzeni R^3"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5  # faster than np.linalg.norm

def random_versor() -> np.ndarray:
    """Losuje wersor"""
    x = np.random.normal()
    y = np.random.normal()
    z = np.random.normal()
    d = (x**2 + y**2 + z**2)**0.5
    return np.array([x/d, y/d, z/d])

def self_avoiding_random_walk(n: int, step: float = 1.0, bead_radius: float = 0.5, epsilon: float = 0.001, two_dimensions=False) -> np.ndarray:
    potential_new_step = [0, 0, 0]
    while True:
        points = [np.array([0, 0, 0])]
        for _ in tqdm(range(n - 1)):
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
    return points_new


def get_custom_force_formula(f_type: str="attractive", f_formula: str="harmonic", linear: bool=False, l_bound: float=1, u_bound: float=1, u_linear:float=1) -> str:
    """
    Creates a formula for the CustomBondForce function. 
    f_type: can be either attractive or repulsive. 
    f_formula: can be either harmonic or gaussian.
    linear: if True the formula will become linear after the u_linear value.
    l_bound: Marks the end of the repulsive force and the beginning of the flat bottom of the potential.
    u_bound: Marks the end of the flat bottom of the potential and the beginning of the attractive part. If l_bound==u_bound no flat bottom is used.
    l_linear: Marks the end of the attractive part of the potential and the beginning of the linear function. If u_bound==l_linear no linearization is used.
    Returns string with the desired force formula.
    """
    formula = ""
    if f_type == "attractive":
        if f_formula == "harmonic":
            if l_bound == u_bound == u_linear:
                formula = '(r-r0)^2'
            elif u_bound == u_linear:
                formula = '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u)'
            else:
                formula = '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u)*step(r_u+d-r) + d*(2*r-d-2*r_u)*step(r-r_u-d))'
        elif f_formula == "gaussian":
            if l_bound == u_bound:
                formula = '(1-exp(-(r-r_0)^2/2/r_0^2))'
            else:
                formula = '((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u))'
    elif f_type == "repulsive":
        formula = '((r-r_l)^2*step(r_l-r)'

    if formula == "":
        raise(Exception("Could not find a correct formula for a force."))
    return formula

