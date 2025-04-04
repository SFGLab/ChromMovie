#################################################################
########### Krzysztof Banecki, Warsaw 2025 ######################
#################################################################

import numpy as np
import pandas as pd
import os
from scipy.spatial import distance_matrix
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.spatial import distance_matrix
from ChromMovie_utils import mmcif2npy

def get_energy(energy_file: str) -> pd.DataFrame:
    """Returns data frame for simulation step, potential energy, total energy and temperature."""
    df = pd.read_csv(energy_file)
    return df


def get_rg(a: np.array) -> float:
    return np.sqrt(np.sum((a - a.mean(axis=0))**2)/a.shape[0])


def get_mean_Rg(cif_folder: str) -> pd.DataFrame:
    """Returns the mean radius of gyration across all of the frames for each simulation step."""
    files = os.listdir(cif_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "Rg"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = mmcif2npy(os.path.join(cif_folder, file))
        rg = get_rg(structure)
        df.loc[df.shape[0]] = [step, frame, rg]
    return df


def get_ev_violation(cif_folder: str, min_dist: float) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads and their expected distance."""
    files = os.listdir(cif_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "violation"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = mmcif2npy(os.path.join(cif_folder, file))
        m = structure.shape[0]
        if m <= 60:
            diffs = [np.sqrt(np.sum((structure[i, :]-structure[j,:])**2))-min_dist for i in range(m-1) for j in range(i+1, m)]
        else:
            diffs = []
            for _ in range(1_000):
                i = np.random.randint(m)
                j = np.random.randint(m)
                if i<j:
                    diffs.append(np.sqrt(np.sum((structure[i, :]-structure[j,:])**2))-min_dist)
        df.loc[df.shape[0]] = [step, frame, np.mean(diffs)]
    return df


def get_bb_violation(cif_folder: str, expected_dist: float) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads and their expected distance."""
    files = os.listdir(cif_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "violation"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = mmcif2npy(os.path.join(cif_folder, file))
        diffs = [np.sqrt(np.sum((structure[i, :]-structure[i+1,:])**2))-expected_dist for i in range(structure.shape[0]-1)]
        df.loc[df.shape[0]] = [step, frame, np.mean(diffs)]
    return df
        

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


def get_sc_pvals(cif_folder: str, heatmaps: list) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads connected by sc contact and their expected distances."""
    files = os.listdir(cif_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "pval"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = mmcif2npy(os.path.join(cif_folder, file))
        pval = test_mannwhitney(heatmaps[frame], structure)
        df.loc[df.shape[0]] = [step, frame, np.log10(pval)]
    return df


def get_sc_violation(cif_folder: str, expected_dist: float, heatmaps: list) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads connected by sc contact and their expected distances."""
    files = os.listdir(cif_folder)
    m = heatmaps[0].shape[0]
    lists_of_contacts = [[(i,j) for i in range(1, m) for j in range(i) if heatmaps[k][j, i]>0] for k in range(len(heatmaps))]
    df = pd.DataFrame(columns=["step", "frame", "pval"])
    
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = mmcif2npy(os.path.join(cif_folder, file))
        diffs = [np.sqrt(np.sum((structure[i, :]-structure[j,:])**2))-expected_dist for (i, j) in lists_of_contacts[frame]]
        value = 0 if len(diffs) == 0 else np.mean(diffs)
        df.loc[df.shape[0]] = [step, frame, value]
    return df


def get_ff_violation(cif_folder: str, expected_dist: float) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads connected by sc contact and their expected distances."""
    files = os.listdir(cif_folder)
    steps = [int(file.split("_")[0].split("step")[-1]) for file in files]
    frames = [int(file.split("_frame")[1].split(".")[0]) for file in files]
    min_step = np.min([step for step in steps if step>0])
    max_step = np.max(steps)
    max_frame = np.max(frames)
    df = pd.DataFrame(columns=["step", "frame", "violation"])

    for step in range(min_step, max_step+1):
        for frame in range(max_frame):
            structure_prev = mmcif2npy(os.path.join(cif_folder, "step{}_frame{}.cif".format(str(step).zfill(3), str(frame).zfill(3))))
            structure_curr = mmcif2npy(os.path.join(cif_folder, "step{}_frame{}.cif".format(str(step).zfill(3), str(frame+1).zfill(3))))
            diffs = [np.sqrt(np.sum((structure_prev[i, :]-structure_curr[i,:])**2))-expected_dist for i in range(structure_curr.shape[0])]
            df.loc[df.shape[0]] = [step, frame, np.mean(diffs)]
    return df


def compute_contact_probability(positions: np.ndarray, d_c: float=1.0) -> float:
    """
    Compute contact probability P(s) and fit exponent alpha for a polymer represented 
    by an n x 3 position matrix.

    Parameters:
        positions (numpy.ndarray): An (n x 3) matrix of 3D coordinates.
        d_c (float): Contact distance threshold.

    Returns:
        s_values (numpy.ndarray): Genomic separation values.
        P_s (numpy.ndarray): Contact probability P(s).
        alpha (float): Power-law exponent of P(s) ~ s^(-alpha).
    """
    m = positions.shape[0]

    dist_matrix = distance_matrix(positions, positions)
    threshold = d_c if d_c is not None else np.percentile(dist_matrix, 25)
    contact_matrix = (dist_matrix < threshold).astype(int)

    # Compute contact probability P(s)
    max_s = m - 1
    s_values = np.arange(1, max_s)
    P_s = np.zeros_like(s_values, dtype=float)

    for i, s in enumerate(s_values):
        contacts = [contact_matrix[j, j+s] for j in range(m - s)]
        P_s[i] = np.mean(contacts)

    # Fit power-law exponent alpha using log-log linear regression
    valid_idx = P_s > 0 
    log_s = np.log(s_values[valid_idx])
    log_Ps = np.log(P_s[valid_idx])

    alpha, _ = np.polyfit(log_s, log_Ps, 1)
    return -alpha


def get_ps_curve_alpha(cif_folder: str, d_c: float=1.0) -> pd.DataFrame:
    """Compute power-law exponent in the P(s) curves of structures."""
    files = os.listdir(cif_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "alpha"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = mmcif2npy(os.path.join(cif_folder, file))
        alpha = compute_contact_probability(structure, d_c=d_c)
        df.loc[df.shape[0]] = [step, frame, alpha]
    return df


def get_local_variability(cif_folder: str, n: int) -> pd.DataFrame:
    """Compute Rg-like variability throughout all frames per locus at the final step of simulation."""
    files = os.listdir(cif_folder)
    steps = [int(file.split("_")[0].split("step")[-1]) for file in files]
    final_step = np.max(steps)

    df = pd.DataFrame(columns=["pos", "rg"])
    structures = [mmcif2npy(os.path.join(cif_folder, f"step{str(final_step).zfill(3)}_frame{str(frame).zfill(3)}.cif")) for frame in range(n)]
    for pos in range(structures[0].shape[0]):
        points = np.array([structures[j][pos] for j in range(len(structures))])
        df.loc[df.shape[0]] = [pos, get_rg(points)]

    return df


def get_local_sc_violation(cif_folder: str, expected_dist: float, n: int, heatmaps: list) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads connected by sc contact and their expected distances."""
    files = os.listdir(cif_folder)
    steps = [int(file.split("_")[0].split("step")[-1]) for file in files]
    final_step = np.max(steps)
    m = heatmaps[0].shape[0]
    lists_of_contacts = [[(i,j) for i in range(1, m) for j in range(i) if heatmaps[k][j, i]>0] for k in range(len(heatmaps))]
    df = pd.DataFrame(columns=["pos", "frame", "sum_viol"])
    
    last_step_files = [os.path.join(cif_folder, f"step{str(final_step).zfill(3)}_frame{str(frame).zfill(3)}.cif") for frame in range(n)]
    structures = [mmcif2npy(file) for file in last_step_files]
    for pos in range(structures[0].shape[0]):
        for frame in range(n):
            structure = structures[frame]
            contacts = [ (i,j) for (i, j) in lists_of_contacts[frame] if i==pos or j==pos]
            viols = [np.sqrt(np.sum((structure[i, :]-structure[j,:])**2))-expected_dist for (i, j) in contacts]
            viols = [v for v in viols if v>0]
            df.loc[df.shape[0]] = [pos, frame, np.sum(viols)]
    return df