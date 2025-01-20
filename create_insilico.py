#################################################################
########### Krzysztof Banecki, Warsaw 2025 ######################
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance_matrix
from hilbertcurve.hilbertcurve import HilbertCurve


def get_structure_01(frames=5, turns=4, m=200, radius=1, shift=1):
    """
    Structrue 01: Spiral
    The Archimedean spiral with the middle section passing through the external over time.
    """
    r_range = np.linspace(0, radius, m)
    phi_range = np.linspace(0, 2*np.pi*turns, m)
    
    structures = []
    for frame in range(frames):
        structure = np.zeros((m, 3))
        max_shift = shift*(1-2*frame/(frames-1))
        for i, r, phi in zip(range(m), r_range, phi_range):
            structure[i, 0] = r*np.cos(phi)
            structure[i, 1] = r*np.sin(phi)
            structure[i, 2] = (radius - r)*max_shift
        structures.append(structure)
    return structures


def get_structure_02(frames=5, m=200, base_d=1, h_start=0.1, h_end=0.9):
    """
    Structrue 02: Zigzag
    A zigzag starting from close to straight line and then contracting and forming a circle.
    """
    h_range = np.linspace(h_start, h_end, frames)
    phi_range = np.linspace(4*np.pi/m*0.1, 4*np.pi/m*0.9, frames)
    
    structures = []
    for frame in range(frames):
        structure = np.zeros((m, 3))
        h = h_range[frame]
        phi = phi_range[frame]
        phi_current = 0
        a = np.sqrt(base_d**2 - h**2)
        for i in range(1, m):
            structure[i, 0] = structure[i-1, 0] + a*np.cos(phi_current)
            structure[i, 1] = structure[i-1, 1] + a*np.sin(phi_current)
            if i%2==1:
                structure[i, 2] = h
            else:
                structure[i, 2] = 0
                phi_current += phi
        structures.append(structure)
    return structures


def generate_hilbert_curve(n_points, p=8, n=3, displacement_sigma=0.1, add_noise=False):
    hilbert_curve = HilbertCurve(p, n)

    distances = list(range(n_points))
    points = np.array(hilbert_curve.points_from_distances(distances))
    if add_noise:
        displacement = np.random.normal(loc=0.0, scale=displacement_sigma, size=n_points*3).reshape(n_points, 3)
        V_interpol = V_interpol + displacement
    
    return points


def visualize_structure(structure_set, output_path):
    """
    Visualizes the first, last and the middle frame of the structure.
    """
    n = len(structure_set)
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 7))
    frames_to_plot = [0, int(n/2), n-1]
    for i, frame in enumerate(frames_to_plot):
        structure = structure_set[frame]
        ax[i].plot(structure[:, 0], structure[:, 1], structure[:, 2])
    ax[0].set_title("Start frame")
    ax[1].set_title("Frame {}/{}".format(str(frames_to_plot[1]+1), str(n)))
    ax[2].set_title("End frame")
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def get_hicmaps(structure_set, n_contacts=100):
    """
    Creates artificial single cell Hi-C meatmaps from a set of structures.
    """
    heatmaps = []
    n = len(structure_set)
    m = structure_set[0].shape[0]
    for i in range(n):
        heatmap = np.zeros((m, m))
        dist_mat = distance_matrix(structure_set[i], structure_set[i])
        dist_mat = (dist_mat+0.001)**(-1) #(np.max(dist_mat)-dist_mat)
        
        dist_weights = []
        for i in range(m-1):
            dist_weights += list(dist_mat[i, i+1:])
        dist_sum = np.sum(dist_weights)
        dist_weights = [w/dist_sum for w in dist_weights]
        contacts = np.sort(np.random.choice(int((m**2-m)/2), n_contacts, p=dist_weights, replace=True))

        k = 0
        for i in range(m-1):
            for j in range(i+1, m):
                if len(contacts)>=1 and k == contacts[0]:
                    heatmap[i, j] += 1
                    heatmap[j, i] += 1
                    contacts = contacts[1:]
                k += 1
            
        heatmaps.append(heatmap)
    return heatmaps
        

def matrix_plot(matrix, output_path):
    """
    The function visualises the scHi-C matrix and saves the result to the file_name.
    """
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap='binary', interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # structure_set = get_structure_01(turns=6)
    structure_set = get_structure_02(frames=10, m=20)
    visualize_structure(structure_set, "./plots/plot_00.png")
    # heatmaps = get_hicmaps(structure_set, n_contacts=50000)
    # matrix_plot(heatmaps[0], "./plots/heatmap_00.png")
