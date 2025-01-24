import numpy as np
import pandas as pd
from points_io import point_reader, save_points_as_pdb
from scipy.spatial import distance_matrix
from scipy.stats import wilcoxon, mannwhitneyu
import openmm as mm
from Bio.PDB import qcprot

from ChromMovie_core import *
from create_insilico import get_structure_01, get_hicmaps

from generate_yaml import generate_yaml_config
from run_ChromMovie_from_yaml import ChromMovie_from_yaml
from reporter_utils import get_local_sc_violation

def create_random_matrix(m: int, n_contact: int) -> np.array:
    if (m**2-m)/2 < n_contact:
        raise ValueError("Number of contacts must be smaller than (m^2-m)/2.")
    
    mat = np.zeros((m, m))
    contacts = np.sort(np.random.choice(int((m**2-m)/2), n_contact))
    k = 0
    for i in range(m-1):
        for j in range(i+1, m):
            if len(contacts)>=1 and k == contacts[0]:
                mat[i, j] = mat[j, i] = 1
                contacts = contacts[1:]
            k += 1
    return mat


def add_noise(mat: np.array, n_error: int) -> np.array:
    m = mat.shape[0]
    errors = np.sort(np.random.choice(int((m**2-m)/2), n_error))
    k = 0
    for i in range(m-1):
        for j in range(i+1, m):
            if len(errors)>=1 and k == errors[0]:
                mat[i, j] = mat[j, i] = 1 - mat[i, j]
                errors = errors[1:]
            k += 1
    return mat


def test_mannwhitney(matrix, structure):
    m = matrix.shape[0]
    dist_mat = distance_matrix(structure, structure)
    dist_cont = []
    dist_nocont = []
    for i in range(m-1):
        for j in range(i+1, m):
            if matrix[i, j] == 1:
                dist_cont.append(dist_mat[i, j])
            else:
                dist_nocont.append(dist_mat[i, j])
    test = mannwhitneyu(dist_cont, dist_nocont, alternative="less")
    return test.pvalue


def rmsd(points1:np.array, points2: np.array) -> float:
    m = points1.shape[0]
    return np.sqrt(np.sum((points1-points2)*(points1-points2))/m)


def extrapolate_points(points, n):
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


def get_qcp_fit_model(image_structure, gd_structure):
    """
    applying the QCP transformation to fit into the points of the image
    :param image_structure: list of points from the image to which other structure is to be transformed
    :param gd_structure: structure to transform
    :return: transformed structure
    """
    n = len(image_structure)
    gd_points = np.array(extrapolate_points(gd_structure, n))
    qcp_transformation = qcprot.QCPSuperimposer()
    qcp_transformation.set(reference_coords=np.array(image_structure), coords=gd_points)
    qcp_transformation.run()
    return qcp_transformation.get_transformed()


if __name__ == "__main__":
    test_no = "on_off3"

    n = 5
    m = 40
    n_contacts = 50
    
    out_path = "./results/tests/test{}/".format(str(test_no).zfill(2))
    N_steps = 50
    sim_step = 20
    
    # heatmaps = [create_random_matrix(m, 40) for i in range(n)]
    structure_set = get_structure_02(frames=n, m=m)
    heatmaps = get_hicmaps(structure_set, n_contacts=n_contacts)

    # Saving the ground truth structure:
    path_init = os.path.join(out_path, "ground_truth")
    if os.path.exists(path_init): shutil.rmtree(path_init)
    if not os.path.exists(path_init): os.makedirs(path_init)
    for frame in range(n):
        save_points_as_pdb(structure_set[frame], os.path.join(path_init, f"frame_{str(frame).zfill(3)}.pdb"))


    df_results = pd.DataFrame(columns = ["base_d", "k_ev", "k_bb", "k_sc", "k_ff", "k_ev_evol", "k_bb_evol", "k_sc_evol", "k_ff_evol"] + [f"RMSD_frame{i}" for i in range(n)] + [f"scviol_frame{i}" for i in range(n)])
    n_tests = 1_000
    for i in range(n_tests):
        print(f"------Running test no {i}------")
        # heatmaps = [add_noise(mat, 5) for mat in heatmaps_orig]

        base_d = 1 #10**np.random.uniform(-2, 0)
        k_ev = 1e2 #10**np.random.uniform(0, 5)
        k_bb = 1e3 #10**np.random.uniform(0, 5)
        k_sc = 1e2 #10**np.random.uniform(0, 5)
        k_ff = 1e2

        k_ev_evol = bool(np.random.randint(2))
        k_bb_evol = bool(np.random.randint(2))
        k_sc_evol = bool(np.random.randint(2))
        k_ff_evol = bool(np.random.randint(2))
        
        # std_config = {
        #     'general': {
        #         'input': 'examples/example1_cell_cycle',
        #         'output': out_path,
        #         'pdf_report': True
        #     }
        # }

        user_specified_config = {
            'general': {
                'input': None,
                'output': out_path,
                'n': n,
                'm': m,
                'n_contacts': n_contacts,
                'artificial_structure': 2,
                'pdf_report': False
            },
            'simulation': {
                'resolutions': "2,1",
                'N_steps': N_steps
            },
            'forcefield': {
                'ev_coef': k_ev,
                'bb_coef': k_bb,
                'sc_coef': k_sc,
                'ff_coef': k_ff,
                'ev_coef_evol': k_ev_evol,
                'bb_coef_evol': k_bb_evol,
                'sc_coef_evol': k_sc_evol,
                'ff_coef_evol': k_ff_evol,
            }
        }

        generate_yaml_config('config.yaml', user_specified_config)
        ChromMovie_from_yaml('config.yaml')

        # pvals = []
        rmsds = []
        for frame in range(n):
            # structure = point_reader(os.path.join(out_path, "frames_pdb", "step{}_frame{}.pdb".format(str(N_steps-1).zfill(3), str(frame).zfill(3))))
            # pvals.append(test_mannwhitney(heatmaps[frame], structure))

            structure = mmcif2npy(os.path.join(out_path, "frames_cif", "step{}_frame{}.cif".format(str(N_steps-1).zfill(3), str(frame).zfill(3))))
            structure_QCP = get_qcp_fit_model(structure_set[frame], structure)
            rmsds.append(rmsd(structure_QCP, structure_set[frame]))

            cif_folder = os.path.join(out_path, "frames_cif")
            npy_folder = os.path.join(out_path, "frames_npy")
            heatmaps = [np.load(os.path.join(npy_folder, f"res2Mb_frame{str(i).zfill(3)}.npy")) for i in range(n)]
            df_sc = get_local_sc_violation(cif_folder, base_d*1.2, n, heatmaps)
            df_sc = df_sc.groupby("frame").sum().reset_index()
            violations = list(df_sc["sum_viol"])
        
        df_results.loc[i] = [base_d, k_ev, k_bb, k_sc, k_ff, k_ev_evol, k_bb_evol, k_sc_evol, k_ff_evol] + rmsds + violations

        if (i+1)%20 == 0:
            df_results.to_csv(os.path.join(out_path, "results_test{}.csv".format(str(test_no).zfill(2))))
