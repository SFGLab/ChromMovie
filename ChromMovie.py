#################################################################
########### Krzysztof Banecki, Warsaw 2024 ######################
#################################################################

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import openmm as mm
import openmm.unit as u
from simtk.openmm import Vec3
from simtk.openmm.app import Topology, Element
from sys import stdout
from openmm.app import PDBFile, PDBxFile, ForceField, Simulation, PDBReporter, PDBxReporter, DCDReporter, StateDataReporter, CharmmPsfFile
from create_insilico import *
from ChromMovie_utils import *
from points_io import save_points_as_pdb
from reporter_utils import get_energy, get_mean_Rg, get_bb_violation, get_sc_violation, get_ff_violation
import shutil

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from fpdf import FPDF
from PIL import Image


class MD_simulation:
    def __init__(self, main_config, sim_config, heatmaps, contact_dfs, output_path, N_steps, burnin, MC_step, platform, force_params):
        '''
        Expects either heatmaps or contact_dfs to be not None.
        '''
        self.genome = main_config["genome"]
        self.chrom = main_config["chrom"]
        self.chrom_size = pd.read_csv(f"chrom_sizes/{self.genome}.txt", 
                                    header=None, index_col=0, sep="\t").loc[self.chrom, 1]
        self.contact_dfs = contact_dfs
        
        if contact_dfs is not None:
            self.n = len(contact_dfs)
            init_resolution = float(str(sim_config['resolutions']).split(",")[0].strip())*1_000_000
            self.m = int(np.ceil(self.chrom_size/init_resolution))
            self.heatmaps = self.get_heatmaps_from_dfs(init_resolution)
        else:
            n = len(heatmaps)
            if n > 1:
                self.n = n
            else:
                raise(Exception("At least two heatmaps must be provided. Got: {}".format(str(n))))
            if not all(h.shape == heatmaps[0].shape for h in heatmaps):
                raise ValueError("Not all heatmaps have the same shape.")
            if heatmaps[0].shape[0] != heatmaps[0].shape[1]:
                raise ValueError("The heatmaps must be rectangular.")
            self.m = heatmaps[0].shape[0]
            if not all(((h==0) | (h==1)).all() for h in heatmaps):
                raise ValueError("Not all heatmaps are binary.")
            self.heatmaps = heatmaps

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

        self.N_steps = N_steps
        self.step, self.burnin = MC_step, burnin//MC_step
        self.platform = platform
        self.force_params = force_params
        self.resolutions = [float(res.strip())*1_000_000 for res in str(sim_config['resolutions']).split(",")]

    
    def run_pipeline(self, run_MD=True, sim_step=5, write_files=False, plots=False):
        '''
        This is the basic function that runs the molecular simulation pipeline.

        Input parameters:
        run_MD (bool): True if user wants to run molecular simulation (not only energy minimization).
        sim_step (int): the simulation step of Langevin integrator.
        write_files (bool): True if the user wants to save the structures that determine the simulation ensemble.
        plots (bool): True if the user wants to see the output average heatmaps.
        '''
        # Define initial structure
        print('Building initial structure...')
        points_init_frame = self_avoiding_random_walk(n=self.m, step=self.force_params[1], bead_radius=self.force_params[1]/2)
        points = np.vstack(tuple([points_init_frame+i*0.001 for i in range(self.n)]))

        write_mmcif(points, self.output_path+'/init_struct.cif')
        path_init = os.path.join(self.output_path, "init")
        if os.path.exists(path_init): shutil.rmtree(path_init)
        if not os.path.exists(path_init): os.makedirs(path_init)
        for frame in range(self.n):
            save_points_as_pdb(points[(frame*self.m):((frame+1)*self.m), :], os.path.join(path_init, f"frame_{str(frame).zfill(3)}.pdb"))

        # Define System
        pdb = PDBxFile(self.output_path+'/init_struct.cif')
        forcefield = ForceField('forcefields/classic_sm_ff.xml')
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

        # Add forces
        print('Adding forces...')
        self.free_start = free_start = True
        if free_start:
            params = self.force_params.copy()
            params[1] = params[7] = 0
            self.add_forcefield(params)
        else:
            self.add_forcefield(self.force_params)

        # Minimize energy
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
        self.simulation.reporters.append(StateDataReporter(os.path.join(self.output_path, "energy.csv"), (self.N_steps*sim_step)//10, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
        # self.simulation.reporters.append(DCDReporter(self.output_path+'/stochastic_MD.dcd', 5))
        self.simulation.context.setPositions(pdb.positions)
        current_platform = self.simulation.context.getPlatform()
        print(f"Simulation will run on platform: {current_platform.getName()}")
        self.simulation.minimizeEnergy()
        print('Energy minimization done\n')

        frame_path_npy = os.path.join(self.output_path, "frames_npy")
        if write_files and os.path.exists(frame_path_npy): shutil.rmtree(frame_path_npy)
        if write_files and not os.path.exists(frame_path_npy): os.makedirs(frame_path_npy)
        frame_path_pdb = os.path.join(self.output_path, "frames_pdb")
        if write_files and os.path.exists(frame_path_pdb): shutil.rmtree(frame_path_pdb)
        if write_files and not os.path.exists(frame_path_pdb): os.makedirs(frame_path_pdb)
        
        # Run molecular dynamics simulation
        
        if run_MD:
            for i, res in enumerate(self.resolutions):
                print(f'Running molecular dynamics at {res/1_000_000}Mb resolution ({i+1}/{len(self.resolutions)})...')
                if i != 0:
                    self.resolution_change(new_res=res, sim_step=sim_step)
                self.save_state(frame_path_npy, frame_path_pdb, step=0)
                self.simulate_resolution(sim_step=sim_step, frame_path_npy=frame_path_npy, frame_path_pdb=frame_path_pdb, 
                                        params=params, free_start=free_start)
                


    def simulate_resolution(self, sim_step: int, frame_path_npy: str, frame_path_pdb: str, params: list, free_start: bool=True):
        start = time.time()
        for i in range(1, self.N_steps):
            self.simulation.step(sim_step)
            if i%self.step == 0 and i > self.burnin*self.step:
                self.save_state(frame_path_npy, frame_path_pdb, step=i)
            # updating the repulsive and frame force strength:
            if free_start:
                t = (i+1)/self.N_steps
                self.system.removeForce(self.ev_force_index)
                self.add_evforce(r=params[0], strength=self.force_params[1]*t)
                self.system.removeForce(self.ff_force_index-1)
                self.add_between_frame_forces(r=params[6], strength=self.force_params[7]*t)
                self.ev_force_index, self.bb_force_index, self.sc_force_index, self.ff_force_index = 3, 1, 2, 4
        self.plot_reporter()
        end = time.time()
        elapsed = end - start

        print(f'Simulation finished succesfully.\nMD finished in {elapsed/60:.2f} minutes.\n')


    def get_heatmaps_from_dfs(self, res: float) -> list:
        """
        Creates a list of heatmaps according to a given resolution based on contact information in self.contact_dfs.
        """
        new_m = int(np.ceil(self.chrom_size / res))
        bin_edges = [i*res for i in range(new_m + 1)]

        new_heatmaps = []
        for df in self.contact_dfs:
            x_bins = np.digitize(df['x'], bin_edges) - 1
            y_bins = np.digitize(df['y'], bin_edges) - 1
            bin_matrix = np.zeros((new_m, new_m), dtype=int)
            for x_bin, y_bin in zip(x_bins, y_bins):
                if 0 <= x_bin < new_m and 0 <= y_bin < new_m:
                    bin_matrix[x_bin, y_bin] += 1
            new_heatmaps.append(bin_matrix)

        return new_heatmaps


    def resolution_change(self, new_res: float, sim_step: int):
        """
        Prepares the system for the simulation in new resolution new_res.
        Updates self.heatmaps, positions of the beads (with added addtional ones) and self.m.
        The OpenMM system is reinitailized and ready for a new simulation.
        """
        # rescaling heatmaps:
        self.heatmaps = self.get_heatmaps_from_dfs(new_res)

        # interpolating structures:
        new_m = int(np.ceil(self.chrom_size / new_res))
        frames = self.get_frames_positions_npy()
        frames_new = [extrapolate_points(frame, new_m) for frame in frames]

        # Saving new starting point:
        points = np.vstack(tuple(frames_new))
        write_mmcif(points, self.output_path+f"/struct_res{str(int(new_res))}.cif")

        # updating parameters:
        self.m = new_m

        # Define System
        pdb = PDBxFile(self.output_path+f"/struct_res{str(int(new_res))}.cif")
        forcefield = ForceField('forcefields/classic_sm_ff.xml')
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

        # Add forces
        print('Adding forces...')
        self.free_start = free_start = True
        if free_start:
            params = self.force_params.copy()
            params[1] = params[7] = 0
            self.add_forcefield(params)
        else:
            self.add_forcefield(self.force_params)

        # Prepare simulation
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
        self.simulation.reporters.append(StateDataReporter(os.path.join(self.output_path, "energy.csv"), (self.N_steps*sim_step)//10, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
        self.simulation.context.setPositions(pdb.positions)
        

    def get_frames_positions_npy(self) -> list:
        """
        Creates a list of size n of (m, 3) numpy arrays with current positions of beads in each frame.
        """
        self.state = self.simulation.context.getState(getPositions=True)
        positions = self.state.getPositions()
        frames = []
        for frame in range(self.n):
            x = [positions[i].x for i in range(frame*self.m, (frame+1)*self.m)]
            y = [positions[i].y for i in range(frame*self.m, (frame+1)*self.m)]
            z = [positions[i].z for i in range(frame*self.m, (frame+1)*self.m)]
            frame_positions = np.hstack((np.array(x).reshape((len(x), 1)), 
                                            np.array(y).reshape((len(y), 1)), 
                                            np.array(z).reshape((len(z), 1))))
            frames.append(frame_positions)
        
        return frames


    def save_state(self, path_npy, path_pdb, step):
        "Saves the current state of the simulation in pdb files and npy arrays. Parameters step and frame specify the name of the file"
        frames = self.get_frames_positions_npy()
        for frame in range(self.n):
            frame_positions = frames[frame]
            np.save(os.path.join(path_npy, "step{}_frame{}.npy".format(str(step).zfill(3), str(frame).zfill(3))),
                    frame_positions)
            save_points_as_pdb(frame_positions, 
                                os.path.join(path_pdb, "step{}_frame{}.pdb".format(str(step).zfill(3), str(frame).zfill(3))),
                                verbose=False)


    def add_evforce(self, r=0.6, strength=4e1):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce('epsilon*(r-r_opt)^2*step(r_opt-r)*delta(frame1-frame2)')
        self.ev_force.addGlobalParameter('r_opt', defaultValue=r)

        self.ev_force.addGlobalParameter('epsilon', defaultValue=strength)
        self.ev_force.addPerParticleParameter('frame')
        
        for frame in range(self.n):
            for _ in range(self.m):
                self.ev_force.addParticle([frame])
        self.ev_force_index = self.system.addForce(self.ev_force)


    def add_backbone(self, r=0.6, strength=1e4):
        'Harmonic bond force between succesive beads'
        self.bond_force = mm.HarmonicBondForce()
        for frame in range(self.n):
            for i in range(self.m-1):
                self.bond_force.addBond(frame*self.m + i, frame*self.m + i + 1, length=r, k=strength)
        self.bb_force_index = self.system.addForce(self.bond_force)


    def add_schic_contacts(self, r=0.3, strength=1e5):
        'Harmonic bond force between loci connected by a scHi-C contact'
        self.sc_force = mm.HarmonicBondForce()
        for frame in range(self.n):
            for i in range(self.m):
                for j in range(i+1, self.m):
                    m_ij = self.heatmaps[frame][i, j]
                    if m_ij > 0:
                        self.sc_force.addBond(frame*self.m + i, frame*self.m + j, length=r/m_ij**(1/3), k=strength)
        self.sc_force_index = self.system.addForce(self.sc_force)

    def add_between_frame_forces(self, r=0.4, strength=1e3):
        'Harmonic bond force between same loci from different frames'
        self.frame_force = mm.CustomBondForce('epsilon2*(r-r0)^2*step(r-r0)') # harmonic with flat beginning (only attractive part)
        # self.frame_force = mm.CustomBondForce('epsilon2*(1-exp(-(r-r_thr)^2/2/r_thr^2))') # gaussian bond
        # self.frame_force = mm.CustomBondForce('epsilon2*((1-exp(-(r-r_l)^2/2/r_l^2))*step(r_l-r) + (1-exp(-(r-r_u)^2/2/r_l^2))*step(r-r_u))') # flat-bottom (r_l, r_u) gaussian bond
        # self.frame_force = mm.CustomBondForce('epsilon2*((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u))') # flat-bottom (r_l, r_u) bond
        # self.frame_force = mm.CustomBondForce('epsilon2*((r-r_l)^2*step(r_l-r) + (r-r_u)^2*step(r-r_u)*step(r_u+d-r) + d*(2*r-d-2*r_u)*step(r-r_u-d))') # flat-bottom (r_l, r_u) bond with linear end after r_u+d

        # self.frame_force.addGlobalParameter('r_u', defaultValue=r*0.5)
        # self.frame_force.addGlobalParameter('r_l', defaultValue=r*1.5)
        # self.frame_force.addGlobalParameter('d', defaultValue=r*0.2)

        self.frame_force.addGlobalParameter('epsilon2', defaultValue=strength)
        self.frame_force.addGlobalParameter('r_thr', defaultValue=r)
        self.frame_force.addPerBondParameter("r0")
        for frame in range(self.n-1):
            for locus in range(self.m):
                self.frame_force.addBond(frame*self.m + locus, (frame+1)*self.m + locus, [1000])
        self.ff_force_index = self.system.addForce(self.frame_force)
        
    def add_forcefield(self, params):
        '''
        Here is the definition of the forcefield.

        There are the following energies:
        - ev force: repelling LJ-like forcefield
        - harmonic bond force: to connect adjacent beads.
        '''
        # self.add_evforce(r=0.4, strength=1e4)
        # self.add_backbone(r=0.2, strength=1e5)
        # self.add_schic_contacts(r=0.1, strength=1e5)
        # self.add_between_frame_forces(r=0.1, strength=1e4)

        self.add_evforce(r=params[0], strength=params[1])
        self.add_backbone(r=params[2], strength=params[3])
        self.add_schic_contacts(r=params[4], strength=params[5])
        self.add_between_frame_forces(r=params[6], strength=params[7])

    def plot_reporter(self):
        print("Creating a simulation report...")
        "Creates a pdf file with reporter plots."
        pdf = FPDF()
        # Page 0
        pdf.add_page()
        pdf.set_font('helvetica', size=20)
        pdf.cell(0, 12, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=12)
        pdf.cell(0, 12, text="Simulation parameters:", new_x="LMARGIN", new_y="NEXT")

        with pdf.table(col_widths=(40, 30, 120)) as table:
            row = table.row()
            row.cell("Parameter")
            row.cell("value")
            row.cell("Parameter description")
            row = table.row()
            row.cell("force_params[0]")
            row.cell(str(self.force_params[0]))
            row.cell("Excluded Volume (EV) minimal distance")
            row = table.row()
            row.cell("force_params[1]")
            row.cell(str(self.force_params[1]))
            row.cell("Excluded Volume (EV) force coefficient")
            row = table.row()
            row.cell("force_params[2]")
            row.cell(str(self.force_params[2]))
            row.cell("Backbone (BB) optimal distance")
            row = table.row()
            row.cell("force_params[3]")
            row.cell(str(self.force_params[3]))
            row.cell("Backbone (BB) force coefficient")
            row = table.row()
            row.cell("force_params[4]")
            row.cell(str(self.force_params[4]))
            row.cell("Single cell contact (SC) optimal distance")
            row = table.row()
            row.cell("force_params[5]")
            row.cell(str(self.force_params[5]))
            row.cell("Single cell contact (SC) force coefficient")
            row = table.row()
            row.cell("force_params[6]")
            row.cell(str(self.force_params[6]))
            row.cell("Frame force (FF) optimal distance")
            row = table.row()
            row.cell("force_params[7]")
            row.cell(str(self.force_params[7]))
            row.cell("Frame force (FF) coefficient")

        # # Page 1
        pdf.add_page()
        pdf.set_font('helvetica', size=20)
        pdf.cell(0, 12, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=12)
        pdf.cell(0, 12, text="Forces functional forms", new_x="LMARGIN", new_y="NEXT")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
        ax[0].plot(np.random.rand(100))
        ax[1].plot(np.linspace(1, 10, 100))
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        pdf.image(img, w=pdf.epw)

        pdf.set_font('helvetica', size=12)
        pdf.cell(0, 12, text="Forces coefficients", new_x="LMARGIN", new_y="NEXT")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
        ax[0].plot(np.random.rand(100))
        ax[1].plot(np.linspace(1, 10, 100))
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        pdf.image(img, w=pdf.epw)

        # # Page 2
        pdf.add_page()
        pdf.set_font('helvetica', size=20)
        pdf.cell(0, 12, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=12)
        pdf.cell(0, 12, text="Simulation metrics", new_x="LMARGIN", new_y="NEXT")

        cmap = mpl.colormaps['coolwarm']

        df_energy = get_energy(os.path.join(self.output_path, "energy.csv"))
        fig, ax = plt.subplots(3, 2, figsize=(10, 12), dpi=300)
        ax[0][0].set_title("Energy")
        ax[0][0].set_ylabel("kJ/mole")
        ax[0][0].set_xlabel("simulation step")
        ax[0][0].plot(df_energy["#\"Step\""], df_energy["Potential Energy (kJ/mole)"], label="Potential E")
        ax[0][0].plot(df_energy["#\"Step\""], df_energy["Total Energy (kJ/mole)"], label="Total E")
        ax[0][0].legend()

        ax[0][1].set_title("Temperature")
        ax[0][1].set_ylabel("K")
        ax[0][1].set_xlabel("simulation step")
        ax[0][1].plot(df_energy["#\"Step\""], df_energy["Temperature (K)"])

        df_rg = get_mean_Rg(os.path.join(self.output_path, "frames_pdb"))
        ax[1][0].set_title("Mean radius of gyration")
        ax[1][0].set_ylabel("Rg")
        ax[1][0].set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_rg[df_rg["frame"]==frame].sort_values("step")
            ax[1][0].plot(df_temp["step"], df_temp["Rg"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax[1][0].legend()

        df_bb = get_bb_violation(os.path.join(self.output_path, "frames_pdb"), self.force_params[2])
        ax[1][1].set_title("Mean backbone distance violation")
        ax[1][1].set_ylabel("")
        ax[1][1].set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_bb[df_bb["frame"]==frame].sort_values("step")
            ax[1][1].plot(df_temp["step"], df_temp["violation"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax[1][1].axhline(0, linestyle='--', c="black")
        ax[1][1].legend()

        df_sc = get_sc_violation(os.path.join(self.output_path, "frames_pdb"), self.force_params[4], self.heatmaps)
        ax[2][0].set_title("Mean sc contact violation")
        ax[2][0].set_ylabel("")
        ax[2][0].set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_sc[df_sc["frame"]==frame].sort_values("step")
            ax[2][0].plot(df_temp["step"], df_temp["pval"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax[2][0].axhline(0, linestyle='--', c="black")
        ax[2][0].legend()

        df_ff = get_ff_violation(os.path.join(self.output_path, "frames_pdb"), self.force_params[6])
        ax[2][1].set_title("Mean frame force violation")
        ax[2][1].set_ylabel("")
        ax[2][1].set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_ff[df_ff["frame"]==frame].sort_values("step")
            ax[2][1].plot(df_temp["step"], df_temp["violation"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        # ax[2][1].axhline(0, linestyle='--', c="black")
        ax[2][1].legend()

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        pdf.image(img, w=pdf.epw)

        pdf.output(os.path.join(self.output_path, "simulation_report.pdf"))
        
if __name__ == "__main__":
    # m = 40
    # structure_set = get_structure_01(frames=20, m=m, turns=5)
    # visualize_structure(structure_set, "./plots/plot_00.png")
    # heatmaps = get_hicmaps(structure_set, n_contacts=int(m*m*2))
    # matrix_plot(heatmaps[0], "./plots/heatmap_00.png")

    # One-sided LE example
    # n = 20
    # m = 20
    # heatmaps = [np.zeros((m, m)) for i in range(n)]
    # for i in range(n):
    #     heatmaps[i][0, i*int(m/n)] = 1

    # Two-sided LE example:
    n = 10
    m = 10
    heatmaps = [np.zeros((m, m)) for i in range(n)]
    for i in range(n):
        heatmaps[i][int(m/2)-int(i*m/2/n), int(m/2)+int(i*m/2/n)] = 1
        # print(int(m/2)-int(i*m/2/n), int(m/2)+int(i*m/2/n))

    # triangle example
    # m = 5int(m/2)-i*int(m/2/n), int(m/2)+i*int(m/2/n)
    # n = 20
    # heatmaps = [np.zeros((m, m)) for i in range(n)]
    # for i in range(n):
    #     heatmaps[i][0, 2] = heatmaps[i][0,3] = heatmaps[i][1, 3] = heatmaps[i][i%3, 4] = heatmaps[i][(i+1)%3, 4] = 1

    # square example
    # m = 3
    # n = 10
    # heatmaps = [np.zeros((m, m)) for i in range(n)]
    # for i in range(n):
    #     heatmaps[i][0, 2] = 1

    # cube example
    # m = 8
    # n = 10
    # heatmaps = [np.zeros((m, m)) for i in range(n)]
    # for i in range(n):
    #     heatmaps[i][0, 3] = heatmaps[i][0, 7] = heatmaps[i][1, 6] = heatmaps[i][2, 5] = heatmaps[i][4, 7] = 1

    out_path = "./results/"
    md = MD_simulation(heatmaps, out_path, N_steps=100, burnin=5, MC_step=1, platform='CPU')
    md.run_pipeline(write_files=True, plots=True, sim_step=100)