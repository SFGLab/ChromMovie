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
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-1, 1)
    d = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return np.array([x / d, y / d, z / d])

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




