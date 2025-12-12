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
from openmm.app import PDBxFile, ForceField, Simulation, StateDataReporter
from create_insilico import *
from ChromMovie_utils import *
from reporter_utils import *
import shutil

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from fpdf import FPDF
from PIL import Image


class MD_simulation:
    def __init__(self, main_config: dict, sim_config: dict, contact_dfs: list, output_path: str, 
                 N_steps: int, burnin: int, MC_step: int, platform: str, force_params: dict):
        '''
        Expects either heatmaps or contact_dfs to be not None.
        '''
        self.genome = main_config["genome"]
        self.chroms = get_unique_chroms(contact_dfs)
        print("Chromosomes detected in the input data: ", self.chroms)
        self.chrom_sizes = [int(pd.read_csv(f"chrom_sizes/{self.genome}.txt", 
                                    header=None, index_col=0, sep="\t").loc[chrom.split("-")[0], 1])
                                    for chrom in self.chroms]
        self.contact_dfs = contact_dfs
        
        self.n = len(contact_dfs)
        self.resolutions = [int(float(res.strip())*1_000_000) for res in str(sim_config['resolutions']).split(",")]
        self.resolutions.sort(reverse=True)
        self.ms = [int(np.ceil(chrom_size/self.resolutions[0])) for chrom_size in self.chrom_sizes]
        self.m = int(np.sum(self.ms))
        m_cumsum = np.cumsum(self.ms)
        self.chrom_breaks = [m_cumsum[-1]*level + m_cumsum[i] - 1 for level in range(self.n) for i in range(len(self.ms))]
        
        self.heatmaps = self.get_heatmaps_from_dfs(self.resolutions[0])
        self.contact_dicts = self.get_dicts_from_dfs(self.resolutions[0])

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

        self.N_steps = N_steps
        self.step, self.burnin = MC_step, burnin
        self.platform = platform
        
        self.user_force_params = force_params
        self.adjust_force_params(self.resolutions[0])
        self.pdf_report = main_config["pdf_report"]
        self.remove_problematic = main_config["remove_problematic"]


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
        points_init_frames = [self_avoiding_random_walk(n=self.ms[i], 
                                                      step=self.force_params["bb_opt_dist"], 
                                                      bead_radius=self.force_params["bb_opt_dist"]/2)
                                for i in range(len(self.ms))]
        points_init_frames = align_self_avoiding_structures(points_init_frames)
        points_init_frame =  np.vstack(tuple(points_init_frames))
        points = np.vstack(tuple([points_init_frame+i*0.001*self.force_params["bb_opt_dist"] for i in range(self.n)]))

        write_mmcif(points, self.output_path+'/struct_00_init.cif', breaks=self.chrom_breaks, chroms=self.chroms)
        path_init = os.path.join(self.output_path, "init")
        if os.path.exists(path_init): shutil.rmtree(path_init)
        if not os.path.exists(path_init): os.makedirs(path_init)
        for frame in range(self.n):
            write_mmcif(points[(frame*self.m):((frame+1)*self.m), :], os.path.join(path_init, f"frame_{str(frame).zfill(3)}.cif"), 
                        breaks=self.chrom_breaks, chroms=self.chroms)

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
        self.add_forcefield(self.force_params)

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
                if self.remove_problematic and i != 0: 
                    print("Removing problematic contacts...")
                    self.remove_problematic_contacts(res)
                print(f'Running molecular dynamics at {resolution2text(res)} resolution ({i+1}/{len(self.resolutions)})...')
                # Changing the system for a new resolution:
                self.resolution_change(new_res=res, sim_step=sim_step, index=i+1, setup=(i!=0))

                # MD simulation at a given resolution
                self.save_state(frame_path_npy, frame_path_cif, res, step=0, save_heatmaps=True)
                self.simulate_resolution(resolution=res, sim_step=sim_step, frame_path_npy=frame_path_npy, frame_path_cif=frame_path_cif)
                
                # Saving structure after resolution simulation:
                frames = self.get_frames_positions_npy()
                points = np.vstack(tuple(frames))
                cif_path = self.output_path+f"/struct_{str(i+1).zfill(2)}_res{resolution2text(res)}_ready.cif"
                write_mmcif(points*10, cif_path, breaks=self.chrom_breaks, chroms=self.chroms)
                

    def simulate_resolution(self, resolution: int, sim_step: int, frame_path_npy: str, frame_path_cif: str) -> None:
        """Runs a simulation for a given resolution and saves the pdf report."""
        start = time.time()
        for i in range(1, self.N_steps):
            self.simulation.step(sim_step)
            if i%self.step == 0 and i >= self.burnin:
                self.save_state(frame_path_npy, frame_path_cif, resolution, step=i, save_heatmaps=False)
            # updating the repulsive and frame force strength:
            if self.user_force_params["ev_coef_evol"] or self.user_force_params["bb_coef_evol"] or \
                self.user_force_params["sc_coef_evol"] or self.user_force_params["ff_coef_evol"]:
                t = np.arctan(10*(2*(i+1)/self.N_steps-1))/np.pi + 0.5

                if self.user_force_params["ev_coef_evol"]:
                    self.simulation.context.setParameter("epsilon_ev", self.force_params["ev_coef"]*t)
                if self.user_force_params["bb_coef_evol"]:
                    self.simulation.context.setParameter("epsilon_ev", self.force_params["bb_coef"]*t)
                if self.user_force_params["sc_coef_evol"]:
                    self.simulation.context.setParameter("epsilon_ev", self.force_params["sc_coef"]*t)
                if self.user_force_params["ff_coef_evol"]:
                    self.simulation.context.setParameter("epsilon_ev", self.force_params["ff_coef"]*t)
        
        print(f'Simulation finished succesfully.')
        if self.pdf_report:
            self.plot_reporter(resolution=resolution)
        end = time.time()
        elapsed = end - start
        print(f'MD round finished in {elapsed/60:.5f} minutes.\n')


    def remove_problematic_contacts(self, res: int, thresh: float=1.2) -> None:
        """
        Removes problematic contacts from self.contact_dfs.
        Problematic contacts are those that even though connected by a contact they 
        are further away from each other that thresh*self.force_params["sc_opt_dist"]
        """
        frames = self.get_frames_positions_npy()
        bin_edges = [i*res for i in range(self.ms[0] + 1)]
        all_contacts = []
        removed = []
        for frame in tqdm(range(self.n)):
            # determining which contacts to remove
            max_removed = int(0.2*self.contact_dfs[frame].shape[0])
            to_remove = []
            contact_indexes = list(self.contact_dfs[frame].index)
            np.random.shuffle(contact_indexes)
            for c in contact_indexes:
                bin1 = int(min(self.ms[0]-1, np.digitize(self.contact_dfs[frame].loc[c, 'pos1'], bin_edges) - 1))
                bin2 = int(min(self.ms[0]-1, np.digitize(self.contact_dfs[frame].loc[c, 'pos2'], bin_edges) - 1))
                p1 = frames[frame][bin1, :]
                p2 = frames[frame][bin2, :]
                dist = np.sqrt(np.sum((p1 - p2)**2))
                if dist > 2*thresh*self.force_params["sc_opt_dist"]:
                    to_remove.append(c)
                    # prevent from removing too many contacts
                    if len(to_remove) >= max_removed:
                        break

            # saving new contact data
            self.contact_dfs[frame] = self.contact_dfs[frame].drop(to_remove)
            removed.append(len(to_remove))
            all_contacts.append(len(contact_indexes))
        
        m_cont_removed = np.mean(removed)
        average_percent = m_cont_removed/np.mean(all_contacts)*100
        print(f"Removed an average of {m_cont_removed:.1f} (~{average_percent:.2f}%) contacts per frame.")
                

    def get_heatmaps_from_dfs(self, res: int) -> list:
        """
        Creates a list of heatmaps according to a given resolution based on contact information in self.contact_dfs.
        """
        new_ms = [int(np.ceil(chrom_size / res)) for chrom_size in self.chrom_sizes]
        bin_edges = [i*res for i in range(new_ms[0] + 1)]

        new_heatmaps = []
        for df in self.contact_dfs:
            x_bins = np.digitize(df['pos1'], bin_edges) - 1
            y_bins = np.digitize(df['pos2'], bin_edges) - 1
            bin_matrix = np.zeros((new_ms[0], new_ms[0]), dtype=int)
            for x_bin, y_bin in zip(x_bins, y_bins):
                if 0 <= x_bin < new_ms[0] and 0 <= y_bin < new_ms[0]:
                    bin_matrix[x_bin, y_bin] += 1
                    bin_matrix[y_bin, x_bin] += 1
            new_heatmaps.append(bin_matrix)

        return new_heatmaps


    def get_dicts_from_dfs(self, res: int) -> list:
        """
        Creates a list of dictionaries according to a given resolution based on contact information in self.contact_dfs.
        The keys of the dictionaries are in the form (chrom1, i, chrom2, j), where i and j and the indices of bins in the resolution res.
        The values of the dictionaries are the number of contacts in the specified bin.
        """
        new_ms = [int(np.ceil(chrom_size / res)) for chrom_size in self.chrom_sizes]
        bin_edges = [i*res for i in range(new_ms[0] + 1)]

        new_dicts = []
        for df in self.contact_dfs:
            df_temp = df.copy()
            df_temp["x_bin"] = np.digitize(df_temp['pos1'], bin_edges) - 1
            df_temp["y_bin"] = np.digitize(df_temp['pos2'], bin_edges) - 1
            df_temp = df_temp[['chrom1', 'x_bin', 'chrom2', 'y_bin']].groupby(['chrom1', 'x_bin', 'chrom2', 'y_bin']).size().reset_index(name='count')
            count_dict = {
                (row.chrom1, row.x_bin, row.chrom2, row.y_bin): row.count
                for row in df_temp.itertuples(index=False)
            }
            new_dicts.append(count_dict)

        return new_dicts
    

    def resolution_change(self, new_res: int, sim_step: int, index: int, setup: bool=False) -> None:
        """
        Prepares the system for the simulation in new resolution new_res.
        Updates self.heatmaps, positions of the beads (with added addtional ones) and self.ms.
        The OpenMM system is reinitailized and ready for a new simulation.
        """
        # Rescaling heatmaps:
        self.heatmaps = self.get_heatmaps_from_dfs(new_res)
        self.contact_dicts = self.get_dicts_from_dfs(new_res)

        # Interpolating structures:
        new_ms = [int(np.ceil(chrom_size / new_res)) for chrom_size in self.chrom_sizes]
        frames = self.get_frames_positions_npy()
        breaks = [-1] + self.chrom_breaks[:len(new_ms)]
        frames_new = [np.vstack(tuple([extrapolate_points(frame[(breaks[i]+1):(breaks[i+1]+1),:], new_ms[i]) for i in range(len(new_ms))])) 
                                      for frame in frames]

        # Updating parameters:
        self.ms = new_ms
        self.m = int(np.sum(self.ms))
        m_cumsum = np.cumsum(new_ms)
        self.chrom_breaks = [m_cumsum[-1]*level + m_cumsum[i] - 1 for level in range(self.n) for i in range(len(new_ms))]

        # Saving new starting point:
        points = np.vstack(tuple(frames_new))
        cif_path = self.output_path+f"/struct_{str(index).zfill(2)}_res{resolution2text(new_res)}_init.cif"
        write_mmcif(points*10, cif_path, breaks=self.chrom_breaks, chroms=self.chroms)

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


    def save_state(self, path_npy: str, path_cif: str, resolution: int, step: int, save_heatmaps: bool) -> None:
        "Saves the current state of the simulation in cif files and npy arrays. Parameters step and frame specify the name of the file"
        frames = self.get_frames_positions_npy()
        for frame in range(self.n):
            frame_positions = frames[frame]
            write_mmcif(frame_positions, 
                            os.path.join(path_cif, "step{}_frame{}.cif".format(str(step).zfill(3), str(frame).zfill(3))),
                            breaks=self.chrom_breaks, chroms=self.chroms)
        if save_heatmaps:
            for frame in range(self.n):
                np.save(os.path.join(path_npy, "res{}_frame{}.npy".format(resolution2text(resolution), str(frame).zfill(3))),
                    self.heatmaps[frame])


    def add_evforce(self, formula_type: str="harmonic", r_min: float=1, coef: float=1e3) -> None:
        'General potential for excluded volume'
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
        'Bond force between succesive beads'
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
                if i not in self.chrom_breaks:
                    self.bond_force.addBond(frame*self.m + i, frame*self.m + i + 1)
        self.bb_force_index = self.system.addForce(self.bond_force)


    def add_schic_contacts(self, formula_type: str="harmonic", r_opt: float=1, r_linear: float=0, coef: float=1e4) -> None:
        'Bond force between loci connected by a scHi-C contact'
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

        breaks = [-1] + self.chrom_breaks
        for frame in range(self.n):
            for key in self.contact_dicts[frame].keys():
                chrom1, i, chrom2, j = key
                m_ij = self.contact_dicts[frame][key]
                id1 = breaks[self.chroms.index(chrom1)] + 1 + i
                id2 = breaks[self.chroms.index(chrom2)] + 1 + j
                
                r_opt_contact = r_opt/m_ij**(1/3)
                self.sc_force.addBond(frame*self.m + id1, frame*self.m + id2, [r_opt_contact*0.8, r_opt_contact*1.2])
        self.sc_force_index = self.system.addForce(self.sc_force)


    def add_between_frame_forces(self, formula_type: str="harmonic", r_opt: float=1, r_linear: float=0, coef: float=1e3) -> None:
        'Bond force between same loci from different frames'
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
        print("Creating simulation report...")
        "Creates a pdf file with reporter plots."
        pdf = FPDF()
        font_main = 15
        font_section = 10
        font_table = 8
        # --------------------- Page 0
        pdf.add_page()
        pdf.set_font('helvetica', size=font_main)
        pdf.cell(0, font_section, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=font_section)
        pdf.cell(0, font_section, text="Simulation parameters:", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=font_table)
        with pdf.table(col_widths=(25, 55, 110)) as table:
            row = table.row()
            row.cell("Parameter")
            row.cell("value")
            row.cell("Parameter description")
            row = table.row()
            row.cell("genome")
            row.cell(str(self.genome))
            row.cell("Genome assembly of the data")
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
            row.cell(", ".join([chrom+": "+str(m) for chrom,m in zip(self.chroms, self.ms)]))
            row.cell("Number of beads in each structure/frame per chromosome (only chromosomes detected in the data are modeled)")

            row = table.row()
            row.cell("ev_min_dist")
            row.cell(str(round(self.force_params["ev_min_dist"],3)))
            row.cell("Excluded Volume (EV) minimal distance")
            row = table.row()
            row.cell("ev_coef")
            row.cell(str(round(self.force_params["ev_coef"],3)))
            row.cell("Excluded Volume (EV) force coefficient")
            row = table.row()
            row.cell("ev_coef_evol")
            row.cell(str(self.force_params["ev_coef_evol"]))
            row.cell("Flag that enables (`True`) or disables (`False`) the increasing EV coefficient value")

            row = table.row()
            row.cell("bb_opt_dist")
            row.cell(str(round(self.force_params["bb_opt_dist"],3)))
            row.cell("Backbone (BB) optimal distance")
            row = table.row()
            row.cell("bb_lin_thresh")
            row.cell(str(round(self.force_params["bb_lin_thresh"],3)))
            row.cell("Backbone (BB) distance after which the potential grows linearly")
            row = table.row()
            row.cell("bb_coef")
            row.cell(str(round(self.force_params["bb_coef"],3)))
            row.cell("Backbone (BB) force coefficient")
            row = table.row()
            row.cell("bb_coef_evol")
            row.cell(str(self.force_params["bb_coef_evol"]))
            row.cell("Flag that enables (`True`) or disables (`False`) the increasing BB coefficient value")

            row = table.row()
            row.cell("sc_opt_dist")
            row.cell(str(round(self.force_params["sc_opt_dist"],3)))
            row.cell("Single cell contact (SC) optimal distance")
            row = table.row()
            row.cell("sc_lin_thresh")
            row.cell(str(round(self.force_params["sc_lin_thresh"],3)))
            row.cell("Single cell contact (SC) distance after which the potential grows linearly")
            row = table.row()
            row.cell("sc_coef")
            row.cell(str(round(self.force_params["sc_coef"],3)))
            row.cell("Single cell contact (SC) force coefficient")
            row = table.row()
            row.cell("sc_coef_evol")
            row.cell(str(self.force_params["sc_coef_evol"]))
            row.cell("Flag that enables (`True`) or disables (`False`) the increasing SC coefficient value")

            row = table.row()
            row.cell("ff_opt_dist")
            row.cell(str(round(self.force_params["ff_opt_dist"],3)))
            row.cell("Frame force (FF) optimal distance")
            row = table.row()
            row.cell("ff_lin_thresh")
            row.cell(str(round(self.force_params["ff_lin_thresh"],3)))
            row.cell("Frame force (FF) distance after which the potential grows linearly")
            row = table.row()
            row.cell("ff_coef")
            row.cell(str(round(self.force_params["ff_coef"],3)))
            row.cell("Frame force (FF) coefficient")
            row = table.row()
            row.cell("ff_coef_evol")
            row.cell(str(self.force_params["ff_coef_evol"]))
            row.cell("Flag that enables (`True`) or disables (`False`) the increasing FF coefficient value")

        # --------------------- Page 1
        pdf.add_page()
        pdf.set_font('helvetica', size=font_main)
        pdf.cell(0, font_section, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=font_section)
        pdf.cell(0, int(font_section/2), text="Forces functional forms", new_x="LMARGIN", new_y="NEXT")

        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), dpi=300)
        y_max = 0

        x = np.linspace(0, self.force_params["ev_min_dist"]*1.5, 100)
        f = formula2lambda(self.ev_formula, l_bound=self.force_params["ev_min_dist"], 
                        u_bound=self.force_params["ev_min_dist"], u_linear=self.force_params["ev_min_dist"])
        y = [self.force_params["ev_coef"]*f(xi) for xi in x]
        y_max = max(y_max, np.max(y))
        ax[0].plot(x, y)
        ax[0].axvline(self.force_params["ev_min_dist"], linestyle='--', c="black")

        if self.force_params["bb_formula"] == "gaussian":
            x = np.linspace(0, self.force_params["bb_opt_dist"]*4, 100)
        else:
            x = np.linspace(0, self.force_params["bb_opt_dist"]*2.5, 100)
        f = formula2lambda(self.bb_formula, l_bound=self.force_params["bb_opt_dist"]*0.8, 
                        u_bound=self.force_params["bb_opt_dist"]*1.2, u_linear=self.force_params["bb_lin_thresh"])
        y = [self.force_params["bb_coef"]*f(xi) for xi in x]
        y_max = max(y_max, np.max(y))
        ax[1].plot(x, y)
        ax[1].axvline(self.force_params["bb_opt_dist"]*0.8, linestyle='--', c="black")
        ax[1].axvline(self.force_params["bb_opt_dist"], linestyle='--', c="black")
        ax[1].axvline(self.force_params["bb_opt_dist"]*1.2, linestyle='--', c="black")

        if self.force_params["sc_formula"] == "gaussian":
            x = np.linspace(0, self.force_params["sc_opt_dist"]*4, 100)
        else:
            x = np.linspace(0, self.force_params["sc_opt_dist"]*2.5, 100)
        f = formula2lambda(self.sc_formula, l_bound=self.force_params["sc_opt_dist"]*0.8, 
                        u_bound=self.force_params["sc_opt_dist"]*1.2, u_linear=self.force_params["sc_lin_thresh"])
        y = [self.force_params["sc_coef"]*f(xi) for xi in x]
        y_max = max(y_max, np.max(y))
        ax[2].plot(x, y)
        ax[2].axvline(self.force_params["sc_opt_dist"]*0.8, linestyle='--', c="black")
        ax[2].axvline(self.force_params["sc_opt_dist"], linestyle='--', c="black")
        ax[2].axvline(self.force_params["sc_opt_dist"]*1.2, linestyle='--', c="black")
        
        if self.force_params["ff_formula"] == "gaussian":
            x = np.linspace(0, self.force_params["ff_opt_dist"]*4, 100)
        else:
            x = np.linspace(0, self.force_params["ff_opt_dist"]*2.5, 100)
        f = formula2lambda(self.ff_formula, l_bound=self.force_params["ff_opt_dist"]*0.8, 
                        u_bound=self.force_params["ff_opt_dist"]*1.2, u_linear=self.force_params["ff_lin_thresh"])
        y = [self.force_params["ff_coef"]*f(xi) if xi>self.force_params["ff_opt_dist"]*1.2 else 0 for xi in x ]
        y_max = max(y_max, np.max(y))
        ax[3].plot(x, y)
        ax[3].axvline(0, linestyle='--', c="black")
        ax[3].axvline(self.force_params["ff_opt_dist"]*1.2, linestyle='--', c="black")
        
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
        plt.close()

        pdf.set_font('helvetica', size=font_section)
        pdf.cell(0, int(font_section/2), text="Forces coefficients", new_x="LMARGIN", new_y="NEXT")

        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), dpi=300)
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
        plt.close()

        pdf.set_font('helvetica', size=font_section)
        pdf.cell(0, font_section, text="Mean distances according to different forces", new_x="LMARGIN", new_y="NEXT")

        cmap = mpl.colormaps['coolwarm']
        fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
        ax1 = ax[0][0]
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

        ax1 = ax[0][1]
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

        ax1 = ax[1][0]
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

        ax1 = ax[1][1]
        df_ff = get_ff_violation(os.path.join(self.output_path, "frames_cif"), 0)
        ax1.set_title("Mean frame force violation")
        ax1.set_ylabel("")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n-1):
            df_temp = df_ff[df_ff["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["violation"], label=f"frames {frame}~{frame+1}" if frame==0 or frame==self.n-2 else "_nolegend_", c=cmap(frame/(self.n-1)))
        ax1.axhline(0, linestyle='--', c="black")
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
        plt.close()

        # --------------------- Page 2
        pdf.add_page()
        pdf.set_font('helvetica', size=font_main)
        pdf.cell(0, font_section, text="ChromMovie simulation reporter", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('helvetica', size=font_section)
        pdf.cell(0, font_section, text="Simulation metrics", new_x="LMARGIN", new_y="NEXT")

        df_energy = get_energy(os.path.join(self.output_path, "energy.csv"))
        fig, ax = plt.subplots(3, 2, figsize=(10, 12), dpi=300)
        ax1 = ax[0][0]
        ax1.set_title("Energy")
        ax1.set_ylabel("kJ/mole")
        ax1.set_xlabel("simulation step")
        ax1.plot(df_energy["#\"Step\""], df_energy["Potential Energy (kJ/mole)"], label="Potential E")
        ax1.plot(df_energy["#\"Step\""], df_energy["Total Energy (kJ/mole)"], label="Total E")
        ax1.legend()

        ax1 = ax[0][1]
        ax1.set_title("Temperature")
        ax1.set_ylabel("K")
        ax1.set_xlabel("simulation step")
        ax1.plot(df_energy["#\"Step\""], df_energy["Temperature (K)"])

        ax1 = ax[1][0]
        df_rg = get_mean_Rg(os.path.join(self.output_path, "frames_cif"))
        ax1.set_title("Mean radius of gyration")
        ax1.set_ylabel("Rg")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_rg[df_rg["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["Rg"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax1.legend()

        ax1 = ax[1][1]
        df_rg = get_ps_curve_alpha(os.path.join(self.output_path, "frames_cif"), None)
        ax1.set_title("P(s) curve \u03B1 coefficient")
        ax1.set_ylabel("\u03B1")
        ax1.set_xlabel("simulation step")
        for frame in range(self.n):
            df_temp = df_rg[df_rg["frame"]==frame].sort_values("step")
            ax1.plot(df_temp["step"], df_temp["alpha"], label=f"frame {frame}" if frame==0 or frame==self.n-1 else "_nolegend_", c=cmap(frame/self.n))
        ax1.legend()

        ax1 = ax[2][0]
        df_rg = get_local_variability(os.path.join(self.output_path, "frames_cif"), self.n)
        ax1.set_title("Local loci variability throughout frames (last step only)")
        ax1.set_ylabel("Loci variability")
        ax1.set_xlabel("Structure bead")
        ax1.plot(df_rg["pos"], df_rg["rg"])

        ax1 = ax[2][1]
        df_loc_v = get_local_sc_violation(os.path.join(self.output_path, "frames_cif"), self.force_params["sc_opt_dist"]*1.2, self.n, self.heatmaps)
        df_loc_v = df_loc_v.groupby("pos").sum().reset_index()
        ax1.set_title("Local violation of sc contacts (last step only)")
        ax1.set_ylabel("Sum of sc contact violation")
        ax1.set_xlabel("Structure bead")
        ax1.plot(df_loc_v["pos"], df_loc_v["sum_viol"])

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        pdf.image(img, w=pdf.epw)
        plt.close()

        pdf.output(os.path.join(self.output_path, f"simulation_report{resolution2text(resolution)}.pdf"))
        


