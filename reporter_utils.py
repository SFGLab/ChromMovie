import numpy as np
import pandas as pd
from points_io import point_reader
import os
from scipy.spatial import distance_matrix
from scipy.stats import wilcoxon, mannwhitneyu


def get_energy(energy_file: str) -> pd.DataFrame:
    """Returns data frame for simulation step, potential energy, total energy and temperature."""
    df = pd.read_csv(energy_file)
    return df


def get_rg(a: np.array) -> float:
    return np.sqrt(np.sum((a - a.mean(axis=0))**2)/a.shape[0])


def get_mean_Rg(pdb_folder: str) -> pd.DataFrame:
    """Returns the mean radius of gyration across all of the frames for each simulation step."""
    files = os.listdir(pdb_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "Rg"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = point_reader(os.path.join(pdb_folder, file))
        rg = get_rg(structure)
        df.loc[df.shape[0]] = [step, frame, rg]
    return df


def get_bb_violation(pdb_folder: str, expected_dist: float) -> pd.DataFrame:
    """Computes the mean difference between real distances between beads and their expected distance."""
    files = os.listdir(pdb_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "violation"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = point_reader(os.path.join(pdb_folder, file))
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


def get_sc_pvals(pdb_folder: str, expected_dist: float, heatmaps: list) -> pd.DataFrame:
    """Computes the mean difference between real distnaces between beads connected by sc contact and their expected distances."""
    files = os.listdir(pdb_folder)
    
    df = pd.DataFrame(columns=["step", "frame", "pval"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = point_reader(os.path.join(pdb_folder, file))
        pval = test_mannwhitney(heatmaps[frame], structure)
        df.loc[df.shape[0]] = [step, frame, np.log10(pval)]
    return df


def get_sc_violation(pdb_folder: str, expected_dist: float, heatmaps: list) -> pd.DataFrame:
    """Computes the mean difference between real distnaces between beads connected by sc contact and their expected distances."""
    files = os.listdir(pdb_folder)
    m = heatmaps[0].shape[0]
    lists_of_contacts = [[(i,j) for i in range(1, m) for j in range(i) if heatmaps[k][i, j]==1] for k in range(len(heatmaps))]
    df = pd.DataFrame(columns=["step", "frame", "pval"])
    for file in files:
        step = int(file.split("_")[0].split("step")[-1])
        frame = int(file.split("_frame")[1].split(".")[0])
        structure = point_reader(os.path.join(pdb_folder, file))
        diffs = [np.sqrt(np.sum((structure[i, :]-structure[j,:])**2))-expected_dist for (i, j) in lists_of_contacts[frame]]
        df.loc[df.shape[0]] = [step, frame, np.mean(diffs)]
    return df


def get_ff_violation(pdb_folder: str, expected_dist: float) -> pd.DataFrame:
    """Computes the mean difference between real distnaces between beads connected by sc contact and their expected distances."""
    files = os.listdir(pdb_folder)
    steps = [int(file.split("_")[0].split("step")[-1]) for file in files]
    frames = [int(file.split("_frame")[1].split(".")[0]) for file in files]
    min_step = np.min([step for step in steps if step>0])
    max_step = np.max(steps)
    max_frame = np.max(frames)
    df = pd.DataFrame(columns=["step", "frame", "violation"])

    for step in range(min_step+1, max_step+1):
        for frame in range(max_frame):
            structure_prev = point_reader(os.path.join(pdb_folder, "step{}_frame{}.pdb".format(str(step-1).zfill(3), str(frame).zfill(3))))
            structure_curr = point_reader(os.path.join(pdb_folder, "step{}_frame{}.pdb".format(str(step).zfill(3), str(frame).zfill(3))))
            diffs = [np.sqrt(np.sum((structure_prev[i, :]-structure_curr[i,:])**2))-expected_dist for i in range(structure_curr.shape[0])]
            df.loc[df.shape[0]] = [step, frame, np.mean(diffs)]
    return df