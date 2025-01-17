#################################################################
########### Krzysztof Banecki, Warsaw 2025 ######################
#################################################################

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import openmm as mm
import openmm.unit as u
from sys import stdout
from openmm.app import PDBxFile, ForceField, Simulation, StateDataReporter
from create_insilico import *
from ChromMovie_utils import *
from reporter_utils import get_energy, get_mean_Rg, get_ev_violation, get_bb_violation, get_sc_violation, get_ff_violation
import shutil

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from fpdf import FPDF
from PIL import Image


class MD_simulation:
    def __init__(self, main_config: dict, sim_config: dict, heatmaps: list, contact_dfs: list, output_path: str, 
                 N_steps: int, burnin: int, MC_step: int, platform: str, force_params: dict):
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
        self.resolutions = [int(float(res.strip())*1_000_000) for res in str(sim_config['resolutions']).split(",")]
        self.resolutions.sort(reverse=True)
        self.user_force_params = force_params
        self.adjust_force_params(self.resolutions[0])


    def adjust_force_params(self, resolution: int) -> list:
        "Adjusts the force parameters addordingly to the current resolution of simulation."
        dist_transformation = lambda d: d*pow(resolution/self.resolutions[-1], 1/3)

        force_params_new = self.user_force_params.copy()
        force_params_new["ev_min_dist"] = dist_transformation(force_params_new["ev_min_dist"])
        force_params_new["bb_opt_dist"] = dist_transformation(force_params_new["bb_opt_dist"])
        force_params_new["bb_lin_thresh"] = dist_transformation(force_params_new["bb_lin_thresh"])
        force_params_new["sc_opt_dist"] = dist_transformation(force_params_new["sc_opt_dist"])
        force_params_new["sc_lin_thresh"] = dist_transformation(force_params_new["sc_lin_thresh"])
        force_params_new["ff_opt_dist"] = dist_transformation(force_params_new["ff_opt_dist"])
        force_params_new["ff_lin_thresh"] = dist_transformation(force_params_new["ff_lin_thresh"])
        
        self.force_params = force_params_new
        
        
    
    def run_pipeline(self, run_MD: bool=True, sim_step: int=5, write_files: bool=True) -> None:
        '''
        This is the basic function that runs the molecular simulation pipeline.

        Input parameters:
        run_MD (bool): True if user wants to run molecular simulation (not only energy minimization).
        sim_step (int): the simulation step of Langevin integrator.
        write_files (bool): True if the user wants to save the structures that determine the simulation ensemble.
        '''
        # Define initial structure
        print('Building initial structure...')
        points_init_frame = self_avoiding_random_walk(n=self.m, step=self.force_params["bb_opt_dist"], bead_radius=self.force_params["bb_opt_dist"]/2)
        points = np.vstack(tuple([points_init_frame+i*0.001*self.force_params["bb_opt_dist"] for i in range(self.n)]))

        write_mmcif(points, self.output_path+'/struct_00_init.cif')
        path_init = os.path.join(self.output_path, "init")
        if os.path.exists(path_init): shutil.rmtree(path_init)
        if not os.path.exists(path_init): os.makedirs(path_init)
        for frame in range(self.n):
            write_mmcif(points[(frame*self.m):((frame+1)*self.m), :], os.path.join(path_init, f"frame_{str(frame).zfill(3)}.cif"))

        # Define System
        cif = PDBxFile(self.output_path+'/struct_00_init.cif')
        forcefield = ForceField('forcefields/classic_sm_ff.xml')
        self.system = forcefield.createSystem(cif.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

        # Add forces
        params = self.force_params.copy()
        if self.user_force_params["ev_coef_evol"]:
            params["ev_coef"] = 0
        if self.user_force_params["bb_coef_evol"]:
            params["bb_coef"] = 0
        if self.user_force_params["sc_coef_evol"]:
            params["sc_coef"] = 0
        if self.user_force_params["ff_coef_evol"]:
            params["ff_coef"] = 0
        self.add_forcefield(params)

        # Minimize energy
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(cif.topology, self.system, integrator, platform)
        self.simulation.reporters.append(StateDataReporter(os.path.join(self.output_path, "energy.csv"), (self.N_steps*sim_step)//10, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
        # self.simulation.reporters.append(DCDReporter(self.output_path+'/stochastic_MD.dcd', 5))
        self.simulation.context.setPositions(cif.positions)
        current_platform = self.simulation.context.getPlatform()
        print(f"Simulation will run on platform: {current_platform.getName()}")
        self.simulation.minimizeEnergy()
        print('Energy minimization done\n')

        frame_path_npy = os.path.join(self.output_path, "frames_npy")
        if write_files and os.path.exists(frame_path_npy): shutil.rmtree(frame_path_npy)
        if write_files and not os.path.exists(frame_path_npy): os.makedirs(frame_path_npy)
        frame_path_cif = os.path.join(self.output_path, "frames_cif")
        if write_files and os.path.exists(frame_path_cif): shutil.rmtree(frame_path_cif)
        if write_files and not os.path.exists(frame_path_cif): os.makedirs(frame_path_cif)
        
        # Run molecular dynamics simulation
        if run_MD:
            for i, res in enumerate(self.resolutions):
                print(f'Running molecular dynamics at {resolution2text(res)} resolution ({i+1}/{len(self.resolutions)})...')
                # Changing the system for a new resolution:
                self.resolution_change(new_res=res, sim_step=sim_step, index=i+1, setup=(i!=0))

                # MD simulation at a given resolution
                self.save_state(frame_path_npy, frame_path_cif, step=0)
                self.simulate_resolution(resolution=res, sim_step=sim_step, frame_path_npy=frame_path_npy, frame_path_cif=frame_path_cif, 
                                        params=params)
                
                # Saving structure after resolution simulation:
                frames = self.get_frames_positions_npy()
                points = np.vstack(tuple(frames))
                cif_path = self.output_path+f"/struct_{str(i+1).zfill(2)}_res{resolution2text(res)}_ready.cif"
                write_mmcif(points, cif_path)
                

    def simulate_resolution(self, resolution: int, sim_step: int, frame_path_npy: str, frame_path_cif: str, params: dict) -> None:
        """Runs a simulation for a given resolution and saves the pdf report."""
        start = time.time()
        for i in range(1, self.N_steps):
            self.simulation.step(sim_step)
            if i%self.step == 0 and i > self.burnin*self.step:
                self.save_state(frame_path_npy, frame_path_cif, step=i)
            # updating the repulsive and frame force strength:
            if self.user_force_params["ev_coef_evol"] or self.user_force_params["bb_coef_evol"] or \
                self.user_force_params["sc_coef_evol"] or self.user_force_params["ff_coef_evol"]:
                t = np.arctan(10*(2*(i+1)/self.N_steps-1))/np.pi + 0.5
                self.system.removeForce(self.ff_force_index)
                self.system.removeForce(self.sc_force_index)
                self.system.removeForce(self.bb_force_index)
                self.system.removeForce(self.ev_force_index)

                if self.user_force_params["ev_coef_evol"]:
                    params["ev_coef"] *= t
                if self.user_force_params["bb_coef_evol"]:
                    params["bb_coef"] *= t
                if self.user_force_params["sc_coef_evol"]:
                    params["sc_coef"] *= t
                if self.user_force_params["ff_coef_evol"]:
                    params["ff_coef"] *= t
                self.add_forcefield(params)

        self.plot_reporter(resolution=resolution)
        end = time.time()
        elapsed = end - start

        print(f'Simulation finished succesfully.\nMD finished in {elapsed/60:.2f} minutes.\n')


    def get_heatmaps_from_dfs(self, res: int) -> list:
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


    def resolution_change(self, new_res: int, sim_step: int, index: int, setup: bool=False) -> None:
        """
        Prepares the system for the simulation in new resolution new_res.
        Updates self.heatmaps, positions of the beads (with added addtional ones) and self.m.
        The OpenMM system is reinitailized and ready for a new simulation.
        """
        # Rescaling heatmaps:
        self.heatmaps = self.get_heatmaps_from_dfs(new_res)

        # Interpolating structures:
        new_m = int(np.ceil(self.chrom_size / new_res))
        frames = self.get_frames_positions_npy()
        frames_new = [extrapolate_points(frame, new_m) for frame in frames]

        # Saving new starting point:
        points = np.vstack(tuple(frames_new))
        cif_path = self.output_path+f"/struct_{str(index).zfill(2)}_res{resolution2text(new_res)}_init.cif"
        write_mmcif(points, cif_path)

        # Updating parameters:
        self.m = new_m

        if setup:
            # Define System
            cif = PDBxFile(cif_path)
            forcefield = ForceField('forcefields/classic_sm_ff.xml')
            self.system = forcefield.createSystem(cif.topology, nonbondedCutoff=1*u.nanometer)
            integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

            # Add forces
            self.adjust_force_params(new_res)
            params = self.force_params.copy()
            if self.user_force_params["ev_coef_evol"]:
                params["ev_coef"] = 0
            if self.user_force_params["bb_coef_evol"]:
                params["bb_coef"] = 0
            if self.user_force_params["sc_coef_evol"]:
                params["sc_coef"] = 0
            if self.user_force_params["ff_coef_evol"]:
                params["ff_coef"] = 0
            self.add_forcefield(params)

            # Prepare simulation
            platform = mm.Platform.getPlatformByName(self.platform)
            self.simulation = Simulation(cif.topology, self.system, integrator, platform)
            self.simulation.reporters.append(StateDataReporter(os.path.join(self.output_path, "energy.csv"), (self.N_steps*sim_step)//10, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
            self.simulation.context.setPositions(cif.positions)
        

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


    def save_state(self, path_npy: str, path_cif: str, step: int) -> None:
        "Saves the current state of the simulation in cif files and npy arrays. Parameters step and frame specify the name of the file"
        frames = self.get_frames_positions_npy()
        for frame in range(self.n):
            frame_positions = frames[frame]
            np.save(os.path.join(path_npy, "step{}_frame{}.npy".format(str(step).zfill(3), str(frame).zfill(3))),
                    frame_positions)
            write_mmcif(frame_positions, 
                                os.path.join(path_cif, "step{}_frame{}.cif".format(str(step).zfill(3), str(frame).zfill(3))))


    def add_evforce(self, formula_type: str="harmonic", r_min: float=1, coef: float=1e3) -> None:
        'Leonard-Jones potential for excluded volume'
        force_formula = get_custom_force_formula(f_type="repulsive", f_formula=formula_type,
                                                 l_bound=r_min, u_bound=r_min)
        self.ev_formula = force_formula
        force_formula = force_formula.replace("r_l", "r_min")

        self.ev_force = mm.CustomNonbondedForce(f'epsilon_ev*{force_formula}*delta(frame1-frame2)')
        self.ev_force.addGlobalParameter('r_min', defaultValue=r_min)
        self.ev_force.addGlobalParameter('epsilon_ev', defaultValue=coef)
        self.ev_force.addPerParticleParameter('frame')
        
        for frame in range(self.n):
            for _ in range(self.m):
                self.ev_force.addParticle([frame])
        self.ev_force_index = self.system.addForce(self.ev_force)


    def add_backbone(self, formula_type: str="harmonic", r_opt: float=1, r_linear: float=0, coef: float=1e4) -> None:
        'Harmonic bond force between succesive beads'
        force_formula = get_custom_force_formula(f_type="attractive", f_formula=formula_type,
                                                 l_bound=r_opt*0.8, u_bound=r_opt*1.2, u_linear=r_linear)
        self.bb_formula = force_formula
        force_formula = force_formula.replace("r_l", "r_l_bb")
        force_formula = force_formula.replace("r_u", "r_u_bb")
        force_formula = force_formula.replace("d", "d_bb")

        self.bond_force = mm.CustomBondForce(f'epsilon_bb*{force_formula}')
        self.bond_force.addGlobalParameter('r_l_bb', defaultValue=r_opt*0.8)
        self.bond_force.addGlobalParameter('r_u_bb', defaultValue=r_opt*1.2)
        self.bond_force.addGlobalParameter('d_bb', defaultValue=r_linear-r_opt*1.2)
        self.bond_force.addGlobalParameter('epsilon_bb', defaultValue=coef)

        for frame in range(self.n):
            for i in range(self.m-1):
                self.bond_force.addBond(frame*self.m + i, frame*self.m + i + 1)
        self.bb_force_index = self.system.addForce(self.bond_force)


    def add_schic_contacts(self, formula_type: str="harmonic", r_opt: float=1, r_linear: float=0, coef: float=1e4) -> None:
        'Harmonic bond force between loci connected by a scHi-C contact'
        force_formula = get_custom_force_formula(f_type="attractive", f_formula=formula_type,
                                                 l_bound=r_opt*0.8, u_bound=r_opt*1.2, u_linear=r_linear)
        self.sc_formula = force_formula
        force_formula = force_formula.replace("r_l", "r_l_sc")
        force_formula = force_formula.replace("r_u", "r_u_sc")
        force_formula = force_formula.replace("d", "d_sc")

        self.sc_force = mm.CustomBondForce(f'epsilon_sc*{force_formula}')
        self.sc_force.addPerBondParameter('r_l_sc')
        self.sc_force.addPerBondParameter('r_u_sc')
        self.sc_force.addGlobalParameter('d_sc', defaultValue=r_linear-r_opt*1.2)
        self.sc_force.addGlobalParameter('epsilon_sc', defaultValue=coef)

        for frame in range(self.n):
            for i in range(self.m):
                for j in range(i+1, self.m):
                    m_ij = self.heatmaps[frame][i, j]
                    if m_ij > 0:
                        r_opt_contact = r_opt/m_ij**(1/3)
                        self.sc_force.addBond(frame*self.m + i, frame*self.m + j, [r_opt_contact*0.8, r_opt_contact*1.2])
        self.sc_force_index = self.system.addForce(self.sc_force)


    def add_between_frame_forces(self, formula_type: str="harmonic", r_opt: float=1, r_linear: float=0, coef: float=1e3) -> None:
        'Harmonic bond force between same loci from different frames'
        force_formula = get_custom_force_formula(f_type="attractive", f_formula=formula_type,
                                                 l_bound=r_opt*0.8, u_bound=r_opt*1.2, u_linear=r_linear)
        self.ff_formula = force_formula
        force_formula = force_formula.replace("r_l", "r_l_ff")
        force_formula = force_formula.replace("r_u", "r_u_ff")
        force_formula = force_formula.replace("d", "d_ff")

        self.frame_force = mm.CustomBondForce(f'epsilon_ff*{force_formula}*step(r-r_u_ff)')

        self.frame_force.addGlobalParameter('r_u_ff', defaultValue=r_opt*0.8)
        self.frame_force.addGlobalParameter('r_l_ff', defaultValue=r_opt*1.2)
        self.frame_force.addGlobalParameter('d_ff', defaultValue=r_linear-r_opt*1.2)
        self.frame_force.addGlobalParameter('epsilon_ff', defaultValue=coef)

        for frame in range(self.n-1):
            for locus in range(self.m):
                self.frame_force.addBond(frame*self.m + locus, (frame+1)*self.m + locus)
        self.ff_force_index = self.system.addForce(self.frame_force)


    def add_forcefield(self, params: dict) -> None:
        '''
        Here is the definition of the force field.

        Force field consists of the following components:
        - ev force: repelling forcefield.
        - backbone force: to connect adjacent beads.
        - sc contact: to attract beads connected by a sc contact.
        - frame force: to attract corresponding beads from adjacent frames.
        '''
        self.add_evforce(formula_type=params["ev_formula"], r_min=params["ev_min_dist"], coef=params["ev_coef"])
        self.add_backbone(formula_type=params["bb_formula"], r_opt=params["bb_opt_dist"], 
                          r_linear=params["bb_lin_thresh"], coef=params["bb_coef"])
        self.add_schic_contacts(formula_type=params["sc_formula"], r_opt=params["sc_opt_dist"], 
                          r_linear=params["sc_lin_thresh"], coef=params["sc_coef"])
        self.add_between_frame_forces(formula_type=params["ff_formula"], r_opt=params["ff_opt_dist"], 
                          r_linear=params["ff_lin_thresh"], coef=params["ff_coef"])


    def plot_reporter(self, resolution: int) -> None:
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
            row.cell("genome")
            row.cell(str(self.genome))
            row.cell("Genome assembly of the data")
            row = table.row()
            row.cell("chrom")
            row.cell(str(self.chrom))
            row.cell("Chromosome")
            row = table.row()
            row.cell("resolution")
            row.cell(resolution2text(resolution))
            row.cell("Simulation resolution")
            row = table.row()
            row.cell("n")
            row.cell(str(self.n))
            row.cell("Number of frames/scHi-C maps")
            row = table.row()
            row.cell("m")
            row.cell(str(self.m))
            row.cell("Number of beads in each structure/frame")

            row = table.row()
            row.cell("ev_min_dist")
            row.cell(str(round(self.force_params["ev_min_dist"],3)))
            row.cell("Excluded Volume (EV) minimal distance")
            row = table.row()
            row.cell("ev_coef")
            row.cell(str(round(self.force_params["ev_coef"],3)))
            row.cell("Excluded Volume (EV) force coefficient")
            row = table.row()
            row.cell("bb_opt_dist")
            row.cell(str(round(self.force_params["bb_opt_dist"],3)))
            row.cell("Backbone (BB) optimal distance")
            row = table.row()
            row.cell("bb_coef")
            row.cell(str(round(self.force_params["bb_coef"],3)))
            row.cell("Backbone (BB) force coefficient")
            row = table.row()
            row.cell("sc_opt_dist")
            row.cell(str(round(self.force_params["sc_opt_dist"],3)))
            row.cell("Single cell contact (SC) optimal distance")
            row = table.row()
            row.cell("sc_coef")
            row.cell(str(round(self.force_params["sc_coef"],3)))
            row.cell("Single cell contact (SC) force coefficient")
            row = table.row()
            row.cell("ff_opt_dist")
            row.cell(str(round(self.force_params["ff_opt_dist"],3)))
            row.cell("Frame force (FF) optimal distance")
            row = table.row()
            row.cell("ff_coef")
            row.cell(str(round(self.force_params["ff_coef"],3)))
            row.cell("Frame force (FF) coefficient")

        # # Page 1
        pdf.add_page()
        pdf.set_font('helvetica', size=20)
        pdf.cell(0, 12, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=12)
        pdf.cell(0, 12, text="Forces functional forms", new_x="LMARGIN", new_y="NEXT")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
        y_max = 0

        x = np.linspace(0, self.force_params["ev_min_dist"]*1.5, 100)
        f = formula2lambda(self.ev_formula, l_bound=self.force_params["ev_min_dist"], 
                        u_bound=self.force_params["ev_min_dist"], u_linear=self.force_params["ev_min_dist"])
        y = [f(xi) for xi in x]
        y_max = max(y_max, np.max(y))
        ax[0].plot(x, y)

        if self.force_params["bb_formula"] == "gaussian":
            x = np.linspace(0, self.force_params["bb_opt_dist"]*4, 100)
        else:
            x = np.linspace(0, self.force_params["bb_opt_dist"]*2.5, 100)
        f = formula2lambda(self.bb_formula, l_bound=self.force_params["bb_opt_dist"]*0.8, 
                        u_bound=self.force_params["bb_opt_dist"]*1.2, u_linear=self.force_params["bb_lin_thresh"])
        y = [f(xi) for xi in x]
        y_max = max(y_max, np.max(y))
        ax[1].plot(x, y)

        if self.force_params["sc_formula"] == "gaussian":
            x = np.linspace(0, self.force_params["sc_opt_dist"]*4, 100)
        else:
            x = np.linspace(0, self.force_params["sc_opt_dist"]*2.5, 100)
        f = formula2lambda(self.sc_formula, l_bound=self.force_params["sc_opt_dist"]*0.8, 
                        u_bound=self.force_params["sc_opt_dist"]*1.2, u_linear=self.force_params["sc_lin_thresh"])
        y = [f(xi) for xi in x]
        y_max = max(y_max, np.max(y))
        ax[2].plot(x, y)
        
        if self.force_params["ff_formula"] == "gaussian":
            x = np.linspace(0, self.force_params["ff_opt_dist"]*4, 100)
        else:
            x = np.linspace(0, self.force_params["ff_opt_dist"]*2.5, 100)
        f = formula2lambda(self.ff_formula, l_bound=self.force_params["ff_opt_dist"]*0.8, 
                        u_bound=self.force_params["ff_opt_dist"]*1.2, u_linear=self.force_params["ff_lin_thresh"])
        y = [f(xi) if xi>self.force_params["ff_opt_dist"]*1.2 else 0 for xi in x ]
        y_max = max(y_max, np.max(y))
        ax[3].plot(x, y)
        
        for i, code in enumerate(["ev", "bb", "sc", "ff"]):
            ax[i].set_ylim((-y_max/30, y_max))
            if code=="ev" or (self.force_params[code+"_formula"]=="harmonic" and self.force_params[code+"_lin_thresh"] <= self.force_params[code+"_opt_dist"]*1.2):
                text = "harmonic"
            elif self.force_params[code+"_formula"]=="harmonic":
                text = "harmonic+linear"
            elif self.force_params[code+"_formula"]=="gaussian":
                text = "gaussian"
            else:
                raise(Exception("Unrecognized force formula."))
            ax[i].set_title(code.upper()+f" force ({text})")
            ax[i].set_xlabel("3D distance")

        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        pdf.image(img, w=pdf.epw)

        pdf.set_font('helvetica', size=12)
        pdf.cell(0, 12, text="Forces coefficients", new_x="LMARGIN", new_y="NEXT")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
        for i, code in enumerate(["ev", "bb", "sc", "ff"]):
            if self.force_params[f"{code}_coef_evol"]:
                x = np.linspace(0, 1, 100)
                y = [self.force_params[f"{code}_coef"]*(np.arctan(10*(2*xi-1))/np.pi + 0.5) for xi in x]
                ax[i].plot(x, y)
            else:
                ax[i].plot([self.force_params[f"{code}_coef"]]*2)
            ax[i].set_title(code.upper()+" coefficient")
            ax[i].set_xlabel("Simulation time (start to end)")
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

        # df_rg = get_mean_Rg(os.path.join(self.output_path, "frames_cif"))
        # ax[1][0].set_title("Mean radius of gyration")
        # ax[1][0].set_ylabel("Rg")
        # ax[1][0].set_xlabel("simulation step")
        # for frame in range(self.n):
        #     df_temp = df_rg[df_rg["frame"]==frame].sort_values("step")
        #     ax[1][0].plot(df_temp["step"], df_temp["Rg"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        # ax[1][0].legend()

        ax1 = ax[1][0]
        df_ev = get_ev_violation(os.path.join(self.output_path, "frames_cif"), 0)
        ax1.set_title("Mean EV distance violation")
        ax1.set_ylabel("")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_ev[df_ev["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["violation"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax1.axhline(self.force_params["ev_min_dist"], linestyle='--', c="black")
        y_lim = ax1.get_ylim()
        ax1.axhspan(y_lim[0]-100, self.force_params["ev_min_dist"], color='red', alpha=0.2)
        ax1.set_ylim(y_lim)
        ax1.legend()

        ax1 = ax[1][1]
        df_bb = get_bb_violation(os.path.join(self.output_path, "frames_cif"), expected_dist=0)
        ax1.set_title("Mean backbone distance violation")
        ax1.set_ylabel("")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_bb[df_bb["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["violation"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax1.axhline(self.force_params["bb_opt_dist"]*0.8, linestyle='--', c="black")
        ax1.axhline(self.force_params["bb_opt_dist"], linestyle='--', c="black")
        ax1.axhline(self.force_params["bb_opt_dist"]*1.2, linestyle='--', c="black")
        y_lim = ax1.get_ylim()
        ax1.axhspan(self.force_params["bb_opt_dist"]*1.2, y_lim[1]+100, color='red', alpha=0.2)
        ax1.axhspan(y_lim[0]-100, self.force_params["bb_opt_dist"]*0.8, color='red', alpha=0.2)
        ax1.set_ylim(y_lim)
        ax1.legend()

        ax1 = ax[2][0]
        df_sc = get_sc_violation(os.path.join(self.output_path, "frames_cif"), 0, self.heatmaps)
        ax1.set_title("Mean sc contact violation")
        ax1.set_ylabel("")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_sc[df_sc["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["pval"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax1.axhline(self.force_params["sc_opt_dist"]*0.8, linestyle='--', c="black")
        ax1.axhline(self.force_params["sc_opt_dist"], linestyle='--', c="black")
        ax1.axhline(self.force_params["sc_opt_dist"]*1.2, linestyle='--', c="black")
        y_lim = ax1.get_ylim()
        ax1.axhspan(self.force_params["sc_opt_dist"]*1.2, y_lim[1]+100, color='red', alpha=0.2)
        ax1.axhspan(y_lim[0]-100, self.force_params["sc_opt_dist"]*0.8, color='red', alpha=0.2)
        ax1.set_ylim(y_lim)
        ax1.legend()

        ax1 = ax[2][1]
        df_ff = get_ff_violation(os.path.join(self.output_path, "frames_cif"), 0)
        ax1.set_title("Mean frame force violation")
        ax1.set_ylabel("")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_ff[df_ff["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["violation"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax1.axhline(self.force_params["ff_opt_dist"]*1.2, linestyle='--', c="black")
        y_lim = ax1.get_ylim()
        ax1.axhspan(self.force_params["ff_opt_dist"]*1.2, y_lim[1]+100, color='red', alpha=0.2)
        ax1.set_ylim(y_lim)
        ax1.legend()

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        pdf.image(img, w=pdf.epw)

        pdf.output(os.path.join(self.output_path, f"simulation_report{resolution2text(resolution)}.pdf"))
        

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
    md.run_pipeline(sim_step=100, write_files=True)

