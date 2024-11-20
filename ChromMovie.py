#################################################################
########### Krzysztof Banecki, Warsaw 2024 ######################
#################################################################

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.unit as u
from sys import stdout
from openmm.app import PDBFile, PDBxFile, ForceField, Simulation, PDBReporter, PDBxReporter, DCDReporter, StateDataReporter, CharmmPsfFile
from create_insilico import *
from ChromMovie_utils import *
from points_io import save_points_as_pdb
import shutil


class MD_simulation:
    def __init__(self, heatmaps, output_path, N_steps, burnin, MC_step, platform, force_params):
        '''
        
        '''
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

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        self.output_path = output_path

        self.N_steps = N_steps
        self.step, self.burnin = MC_step, burnin//MC_step
        self.platform = platform
        self.force_params = force_params
    
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
        points_init_frame = self_avoiding_random_walk(self.m, 0.2, 0.001)
        # points_init_frame = np.random.normal(points_init_frame, 0.01)
        points = np.vstack(tuple([points_init_frame+i*0.001 for i in range(self.n)]))

        # points = self_avoiding_random_walk(self.m*self.n, 2, 0.001)
        write_mmcif(points, self.output_path+'/init_struct.cif')

        # Define System
        pdb = PDBxFile(self.output_path+'/init_struct.cif')
        forcefield = ForceField('forcefields/classic_sm_ff.xml')
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

        # Add forces
        print('Adding forces...')
        self.add_forcefield()
        print('Forces added\n')

        # Minimize energy
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
        self.simulation.reporters.append(StateDataReporter(stdout, (self.N_steps*sim_step)//10, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
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
        
        self.state = self.simulation.context.getState(getPositions=True)
        positions = self.state.getPositions()
        for frame in range(self.n):
            x = [positions[i].x for i in range(frame*self.m, (frame+1)*self.m)]
            y = [positions[i].y for i in range(frame*self.m, (frame+1)*self.m)]
            z = [positions[i].z for i in range(frame*self.m, (frame+1)*self.m)]
            frame_positions = np.hstack((np.array(x).reshape((len(x), 1)), 
                                            np.array(y).reshape((len(y), 1)), 
                                            np.array(z).reshape((len(z), 1))))
            np.save(os.path.join(frame_path_npy, "step{}_frame{}.npy".format(str(0).zfill(3), str(frame).zfill(3))),
                    frame_positions)
            save_points_as_pdb(frame_positions, 
                                os.path.join(frame_path_pdb, "step{}_frame{}.pdb".format(str(0).zfill(3), str(frame).zfill(3))),
                                verbose=False)

        # Run molecular dynamics simulation
        if run_MD:
            print('Running molecular dynamics (wait for 10 steps)...')
            start = time.time()
            for i in range(1, self.N_steps):
                self.simulation.step(sim_step)
                if i%self.step==0 and i>self.burnin*self.step:
                    self.state = self.simulation.context.getState(getPositions=True)
                    if write_files:
                        positions = self.state.getPositions()
                        for frame in range(self.n):
                            x = [positions[i].x for i in range(frame*self.m, (frame+1)*self.m)]
                            y = [positions[i].y for i in range(frame*self.m, (frame+1)*self.m)]
                            z = [positions[i].z for i in range(frame*self.m, (frame+1)*self.m)]
                            frame_positions = np.hstack((np.array(x).reshape((len(x), 1)), 
                                                         np.array(y).reshape((len(y), 1)), 
                                                         np.array(z).reshape((len(z), 1))))
                            np.save(os.path.join(frame_path_npy, "step{}_frame{}.npy".format(str(i).zfill(3), str(frame).zfill(3))),
                                    frame_positions)
                            save_points_as_pdb(frame_positions, 
                                               os.path.join(frame_path_pdb, "step{}_frame{}.pdb".format(str(i).zfill(3), str(frame).zfill(3))),
                                               verbose=False)
                        # time.sleep(1)
            end = time.time()
            elapsed = end - start

            print(f'Everything is done! Simulation finished succesfully!\nMD finished in {elapsed/60:.2f} minutes.\n')


    def add_evforce(self, r=0.6, strength=4e1):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce('epsilon*(r-r_opt)^2*step(r_opt-r)*delta(frame1-frame2)')
        self.ev_force.addGlobalParameter('r_opt', defaultValue=r)

        self.ev_force.addGlobalParameter('epsilon', defaultValue=strength)
        self.ev_force.addPerParticleParameter('frame')
        
        for frame in range(self.n):
            for _ in range(self.m):
                self.ev_force.addParticle([frame])
        self.system.addForce(self.ev_force)

    def add_backbone(self, r=0.6, strength=1e4):
        'Harmonic bond force between succesive beads'
        self.bond_force = mm.HarmonicBondForce()
        for frame in range(self.n):
            for i in range(self.m-1):
                self.bond_force.addBond(frame*self.m + i, frame*self.m + i + 1, length=r, k=strength)
        self.system.addForce(self.bond_force)

    def add_schic_contacts(self, r=0.3, strength=1e5):
        'Harmonic bond force between loci connected by a scHi-C contact'
        self.sc_force = mm.HarmonicBondForce()
        for frame in range(self.n):
            for i in range(self.m):
                for j in range(i+1, self.m):
                    if self.heatmaps[frame][i, j] == 1:
                        self.sc_force.addBond(frame*self.m + i, frame*self.m + j, length=r, k=strength)
            # self.sc_force.addBond(frame*self.m, frame*self.m+2, length=0.2*np.sqrt(2), k=param)
            # self.sc_force.addBond(frame*self.m+1, frame*self.m + 3, length=0.2*np.sqrt(2), k=param)
            # self.sc_force.addBond(frame*self.m+4, frame*self.m+6, length=0.2*np.sqrt(2), k=param)
            # self.sc_force.addBond(frame*self.m+5, frame*self.m+7, length=0.2*np.sqrt(2), k=param)

            # self.sc_force.addBond(frame*self.m, frame*self.m+5, length=0.2*np.sqrt(3), k=param)
            # self.sc_force.addBond(frame*self.m+2, frame*self.m+7, length=0.2*np.sqrt(3), k=param)
            # self.sc_force.addBond(frame*self.m+3, frame*self.m+6, length=0.2*np.sqrt(3), k=param)
            # self.sc_force.addBond(frame*self.m+1, frame*self.m+4, length=0.2*np.sqrt(3), k=param)
            
        self.system.addForce(self.sc_force)

    def add_between_frame_forces(self, r=0.4, strength=1e3):
        'Harmonic bond force between same loci from different frames'
        self.frame_force = mm.CustomBondForce('epsilon2*(r-r_thr)^2*step(r-r_thr)')
        self.frame_force.addGlobalParameter('epsilon2', defaultValue=strength)
        self.frame_force.addGlobalParameter('r_thr', defaultValue=r)
        for frame in range(self.n-1):
            for locus in range(self.m):
                self.frame_force.addBond(frame*self.m + locus, (frame+1)*self.m + locus)
        self.system.addForce(self.frame_force)
        

    def add_forcefield(self):
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

        self.add_evforce(r=self.force_params[0], strength=self.force_params[1])
        self.add_backbone(r=self.force_params[2], strength=self.force_params[3])
        self.add_schic_contacts(r=self.force_params[4], strength=self.force_params[5])
        self.add_between_frame_forces(r=self.force_params[6], strength=self.force_params[7])

        
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