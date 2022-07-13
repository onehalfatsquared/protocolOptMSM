'''
This script will import a time dependent protocol for the potential strengths and 
perform a sequence of runs according to this protocol to generate a simulation trajectory.
Otherwise, it is the same as the other cone simulation scripts. 
'''

import os
#the following stops multi-threading on hpcc. It exceeds core limit quickly without this.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import math
import string
import random
import numpy as np

import hoomd
import hoomd.md as md
import gsd
import gsd.hoomd

import pandas as pd

import pickle

from config_dict import ConfigDict
from init import init_sim


'''
Pseudoatom types

A1-A6: Beads making up the Cone
R: Center of the rigid cones
S: Scaffold Sphere Particle (attractive to A1)

'''




def create_potential(rh, rp, Rs, E_hh, E_ph, rho_cc, E_nc, rho_nc):
    #create the potential corresponding to each interaction in the system
    #repulsion between all non-corresponding types for excluded volume
    #attraction between like attractor beads
    #attraction between bottom bead and scaffold

    #extract the radii from r
    rp1 = rp[0]
    rp2 = rp[1]
    rp3 = rp[2]
    rp4 = rp[3]
    rp5 = rp[4]
    rp6 = rp[5]

    rh1 = rh[0]
    rh2 = rh[1]
    rh3 = rh[2]
    rh4 = rh[3]
    rh5 = rh[4]
    rh6 = rh[5]

    #construct neighbor list with body exclusions
    nl = md.nlist.cell(check_period=1)  # Cell-based neighbor list
    nl.reset_exclusions(exclusions=['body'])

    #define a common repulsion strength for excluded volume ixns
    rep = 0.5

    #create the lennard jones potential - shifted to 0 at cutoff
    lj = md.pair.lj(r_cut = 2.0 * rh6, nlist=nl)
    lj.set_params(mode="shift")

    #init all pair interactions to zero
    lj.pair_coeff.set(['H','P','S','PA1','PA2','PA3','PA4','PA5','PA6','HA1','HA2','HA3','HA4','HA5','HA6'],
                      ['H','P','S','PA1','PA2','PA3','PA4','PA5','PA6','HA1','HA2','HA3','HA4','HA5','HA6'],
                      alpha=0, epsilon=0, r_cut=0, sigma=0)

    #cones - excluded volume ixns
    gamma = 1.25  #factor to increase cutoff for pentamers long range repulsion

    #Note: the A1 <-> A6 interaction is weaker for stability reasons. Particles leave
    #the domain much more often if this is set equal to the other repulsion strengths

    #pentamer-pentamer ixns
    lj.pair_coeff.set('PA1', 'PA1', alpha=0.0, epsilon=rep, r_cut=gamma*(rp1+rp1), sigma=rp1+rp1)
    lj.pair_coeff.set('PA1', 'PA2', alpha=0.0, epsilon=rep, r_cut=rp1+rp2, sigma=rp1+rp2)
    lj.pair_coeff.set('PA1', 'PA3', alpha=0.0, epsilon=rep, r_cut=rp1+rp3, sigma=rp1+rp3)
    lj.pair_coeff.set('PA1', 'PA4', alpha=0.0, epsilon=rep, r_cut=rp1+rp4, sigma=rp1+rp4)
    lj.pair_coeff.set('PA1', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp1+rp5, sigma=rp1+rp5)
    lj.pair_coeff.set('PA1', 'PA6', alpha=0.0, epsilon=0.05, r_cut=rp1+rp6, sigma=rp1+rp6)
    lj.pair_coeff.set('PA2', 'PA2', alpha=0.0, epsilon=rep, r_cut=gamma*(rp2+rp2), sigma=rp2+rp2)
    lj.pair_coeff.set('PA2', 'PA3', alpha=0.0, epsilon=rep, r_cut=rp2+rp3, sigma=rp2+rp3)
    lj.pair_coeff.set('PA2', 'PA4', alpha=0.0, epsilon=rep, r_cut=rp2+rp4, sigma=rp2+rp4)
    lj.pair_coeff.set('PA2', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp2+rp5, sigma=rp2+rp5)
    lj.pair_coeff.set('PA2', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp2+rp6, sigma=rp2+rp6)
    lj.pair_coeff.set('PA3', 'PA3', alpha=0.0, epsilon=rep, r_cut=gamma*(rp3+rp3), sigma=rp3+rp3)
    lj.pair_coeff.set('PA3', 'PA4', alpha=0.0, epsilon=rep, r_cut=rp3+rp4, sigma=rp3+rp4)
    lj.pair_coeff.set('PA3', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp3+rp5, sigma=rp3+rp5)
    lj.pair_coeff.set('PA3', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp3+rp6, sigma=rp3+rp6)
    lj.pair_coeff.set('PA4', 'PA4', alpha=0.0, epsilon=rep, r_cut=gamma*(rp4+rp4), sigma=rp4+rp4)
    lj.pair_coeff.set('PA4', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp4+rp5, sigma=rp4+rp5)
    lj.pair_coeff.set('PA4', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp4+rp6, sigma=rp4+rp6)
    lj.pair_coeff.set('PA5', 'PA5', alpha=0.0, epsilon=rep, r_cut=gamma*(rp5+rp5), sigma=rp5+rp5)
    lj.pair_coeff.set('PA5', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp6+rp5, sigma=rp6+rp5)
    lj.pair_coeff.set('PA6', 'PA6', alpha=0.0, epsilon=rep, r_cut=gamma*(rp6+rp6), sigma=rp6+rp6)

    #hexamer-hexamer ixns
    lj.pair_coeff.set('HA1', 'HA1', alpha=0.0, epsilon=rep, r_cut=rh1+rh1, sigma=rh1+rh1)
    lj.pair_coeff.set('HA1', 'HA2', alpha=0.0, epsilon=rep, r_cut=rh1+rh2, sigma=rh1+rh2)
    lj.pair_coeff.set('HA1', 'HA3', alpha=0.0, epsilon=rep, r_cut=rh1+rh3, sigma=rh1+rh3)
    lj.pair_coeff.set('HA1', 'HA4', alpha=0.0, epsilon=rep, r_cut=rh1+rh4, sigma=rh1+rh4)
    lj.pair_coeff.set('HA1', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh1+rh5, sigma=rh1+rh5)
    lj.pair_coeff.set('HA1', 'HA6', alpha=0.0, epsilon=0.05, r_cut=rh1+rh6, sigma=rh1+rh6)
    lj.pair_coeff.set('HA2', 'HA3', alpha=0.0, epsilon=rep, r_cut=rh2+rh3, sigma=rh2+rh3)
    lj.pair_coeff.set('HA2', 'HA4', alpha=0.0, epsilon=rep, r_cut=rh2+rh4, sigma=rh2+rh4)
    lj.pair_coeff.set('HA2', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh2+rh5, sigma=rh2+rh5)
    lj.pair_coeff.set('HA2', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh2+rh6, sigma=rh2+rh6)
    lj.pair_coeff.set('HA3', 'HA4', alpha=0.0, epsilon=rep, r_cut=rh3+rh4, sigma=rh3+rh4)
    lj.pair_coeff.set('HA3', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh3+rh5, sigma=rh3+rh5)
    lj.pair_coeff.set('HA3', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh3+rh6, sigma=rh3+rh6)
    lj.pair_coeff.set('HA4', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh4+rh5, sigma=rh4+rh5)
    lj.pair_coeff.set('HA4', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh4+rh6, sigma=rh4+rh6)
    lj.pair_coeff.set('HA5', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh6+rh5, sigma=rh6+rh5)
    lj.pair_coeff.set('HA6', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh6+rh6, sigma=rh6+rh6)

    #hexamer-pentamer ixns
    lj.pair_coeff.set('PA1', 'HA1', alpha=0.0, epsilon=rep, r_cut=rp1+rh1, sigma=rp1+rh1)
    lj.pair_coeff.set('PA1', 'HA2', alpha=0.0, epsilon=rep, r_cut=rp1+rh2, sigma=rp1+rh2)
    lj.pair_coeff.set('PA1', 'HA3', alpha=0.0, epsilon=rep, r_cut=rp1+rh3, sigma=rp1+rh3)
    lj.pair_coeff.set('PA1', 'HA4', alpha=0.0, epsilon=rep, r_cut=rp1+rh4, sigma=rp1+rh4)
    lj.pair_coeff.set('PA1', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp1+rh5, sigma=rp1+rh5)
    lj.pair_coeff.set('PA1', 'HA6', alpha=0.0, epsilon=0.05, r_cut=rp1+rh6, sigma=rp1+rh6)
    lj.pair_coeff.set('PA2', 'HA3', alpha=0.0, epsilon=rep, r_cut=rp2+rh3, sigma=rp2+rh3)
    lj.pair_coeff.set('PA2', 'HA4', alpha=0.0, epsilon=rep, r_cut=rp2+rh4, sigma=rp2+rh4)
    lj.pair_coeff.set('PA2', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp2+rh5, sigma=rp2+rh5)
    lj.pair_coeff.set('PA2', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp2+rh6, sigma=rp2+rh6)
    lj.pair_coeff.set('PA3', 'HA4', alpha=0.0, epsilon=rep, r_cut=rp3+rh4, sigma=rp3+rh4)
    lj.pair_coeff.set('PA3', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp3+rh5, sigma=rp3+rh5)
    lj.pair_coeff.set('PA3', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp3+rh6, sigma=rp3+rh6)
    lj.pair_coeff.set('PA4', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp4+rh5, sigma=rp4+rh5)
    lj.pair_coeff.set('PA4', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp4+rh6, sigma=rp4+rh6)
    lj.pair_coeff.set('PA5', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp6+rh5, sigma=rp6+rh5)
    lj.pair_coeff.set('PA6', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp6+rh6, sigma=rp6+rh6)

    #nanoparticle - excluded volume ixns
    lj.pair_coeff.set(['S'], ['PA4'], alpha=0, epsilon=rep, r_cut=Rs+rp4, sigma=Rs+rp4)
    lj.pair_coeff.set(['S'], ['PA5'], alpha=0, epsilon=rep, r_cut=Rs+rp5, sigma=Rs+rp5)
    lj.pair_coeff.set(['S'], ['PA6'], alpha=0, epsilon=rep, r_cut=Rs+rp6, sigma=Rs+rp6)

    lj.pair_coeff.set(['S'], ['HA4'], alpha=0, epsilon=rep, r_cut=Rs+rh4, sigma=Rs+rh4)
    lj.pair_coeff.set(['S'], ['HA5'], alpha=0, epsilon=rep, r_cut=Rs+rh5, sigma=Rs+rh5)
    lj.pair_coeff.set(['S'], ['HA6'], alpha=0, epsilon=rep, r_cut=Rs+rh6, sigma=Rs+rh6)

    #lj.disable() #for debugging potentials

    #set up the morse potential

    #create morse potential
    morse = md.pair.morse(r_cut=3.5, nlist=nl)
    morse.set_params(mode="shift")

    #init all interactions to zero
    morse.pair_coeff.set(['H','P','S','PA1','PA2','PA3','PA4','PA5','PA6','HA1','HA2','HA3','HA4','HA5','HA6'],
                         ['H','P','S','PA1','PA2','PA3','PA4','PA5','PA6','HA1','HA2','HA3','HA4','HA5','HA6'],
                         D0=0, alpha=0.0, r0=0.0, r_cut=0.0)

    #cone-cone attractive interactions
    #hexamer-hexamer
    #set eq distances for each bead -> equal to twice radius of hexamers
    dh2 = 2.0 * rh2
    dh3 = 2.0 * rh3
    dh4 = 2.0 * rh4
    dh5 = 2.0 * rh5
    morse.pair_coeff.set(['HA2'],['HA2'], D0=E_hh, alpha=rho_cc/dh2, r0=dh2, r_cut=2+dh2)
    morse.pair_coeff.set(['HA3'],['HA3'], D0=E_hh, alpha=rho_cc/dh3, r0=dh3, r_cut=2+dh3)
    morse.pair_coeff.set(['HA4'],['HA4'], D0=E_hh, alpha=rho_cc/dh4, r0=dh4, r_cut=2+dh4)
    morse.pair_coeff.set(['HA5'],['HA5'], D0=E_hh, alpha=rho_cc/dh5, r0=dh5, r_cut=2+dh5)

    #hexamer-pentamer
    #set eq distances for each bead -> equal to sum of radii of pentamer and hexamer
    eq2 = rp2 + rh2
    eq3 = rp3 + rh3
    eq4 = rp4 + rh4
    eq5 = rp5 + rh5
    morse.pair_coeff.set(['HA2'],['PA2'], D0=E_ph, alpha=rho_cc/eq2, r0=eq2, r_cut=2+eq2)
    morse.pair_coeff.set(['HA3'],['PA3'], D0=E_ph, alpha=rho_cc/eq3, r0=eq3, r_cut=2+eq3)
    morse.pair_coeff.set(['HA4'],['PA4'], D0=E_ph, alpha=rho_cc/eq4, r0=eq4, r_cut=2+eq4)
    morse.pair_coeff.set(['HA5'],['PA5'], D0=E_ph, alpha=rho_cc/eq5, r0=eq5, r_cut=2+eq5)

    #nanoparticle-cone attractions
    Ehex = E_nc         #set this via an input file parameter
    Epent = 0.9 * Ehex  #set so the ratio of pentamers to hexamers attached is ~ 12/30
    morse.pair_coeff.set(['S'],['HA1'], D0=Ehex,  alpha=rho_nc, r0=Rs+0.3, r_cut=Rs+3)
    morse.pair_coeff.set(['S'],['PA1'], D0=Epent,  alpha=rho_nc, r0=Rs+0.3, r_cut=Rs+3)

    #morse.disable() #for debugging potentials

    return nl, morse, lj


def create_integrator(ts, animation_time, kT, seed, file_num):
    #construct the time integrator for the simulation
    #makes rigid bodies integration group, sets an output file
    #return the path to the output file

    #test of running on expanse
    work_env   = os.getenv("WORK", "None")
    if work_env == "None":
        on_expanse = False
    elif work_env.split("/")[1] == "expanse":
        on_expanse = True
        print("Running on expanse")
    else:
        on_expanse = False

    #set the animation period
    animation_period = int(animation_time / ts)

    # integrate rigid and non rigid structures together (non rigid empty?)
    group_rigid = hoomd.group.rigid_center()
    group_nonrigid = hoomd.group.nonrigid()
    group_integrate = hoomd.group.union('integrate', group_rigid, group_nonrigid)

    #handle the output directory
    if (on_expanse):
        user     = os.environ.get("USER")
        slurm_id = os.environ.get("SLURM_JOB_ID")
        path = "/scratch/" + user + "/job_" + slurm_id + "/" 
        parameter_folder = ""
    else:
        path = "../trajectories/"
        parameter_folder = "opt" + file_num + "/"

        #check that this folder exists. if not, make it
        if (not os.path.exists(path+parameter_folder)):
            os.makedirs(path+parameter_folder)

    #finish naming the output trajectory file
    traj_num = "traj{}".format(seed)
    out_gsd = path + parameter_folder + traj_num + ".gsd"
    out_log = path + parameter_folder + traj_num + ".log"

    # dump the configuration every animation_period
    hoomd.dump.gsd(filename=out_gsd, period=animation_period, group=hoomd.group.all(), 
                    overwrite=True, phase=0)

    # output the energy
    hoomd.analyze.log(filename=out_log, quantities=['temperature', 'potential_energy'],
                      period=animation_period, overwrite=True, phase=0)

    #set the integration mode - langevin
    integrator_mode = md.integrate.mode_standard(dt=ts)
    md.integrate.langevin(group=group_integrate, kT=kT, seed=seed, dscale=True)
    #md.integrate.brownian(group=group_integrate, kT=1.0, seed=seed, dscale=True)

    #return the output file path
    return out_gsd

def modify_potential(potential, E_hh, E_ph, rho_cc, rh, rp):
    #modify the attractive strength of the potential according to the given values

    #cone-cone attractive interactions
    #hexamer-hexamer
    #set eq distances for each bead -> equal to twice radius of hexamers
    dh2 = 2.0 * rh[1]
    dh3 = 2.0 * rh[2]
    dh4 = 2.0 * rh[3]
    dh5 = 2.0 * rh[4]
    potential.pair_coeff.set(['HA2'],['HA2'], D0=E_hh, alpha=rho_cc/dh2, r0=dh2, r_cut=2+dh2)
    potential.pair_coeff.set(['HA3'],['HA3'], D0=E_hh, alpha=rho_cc/dh3, r0=dh3, r_cut=2+dh3)
    potential.pair_coeff.set(['HA4'],['HA4'], D0=E_hh, alpha=rho_cc/dh4, r0=dh4, r_cut=2+dh4)
    potential.pair_coeff.set(['HA5'],['HA5'], D0=E_hh, alpha=rho_cc/dh5, r0=dh5, r_cut=2+dh5)

    #hexamer-pentamer
    #set eq distances for each bead -> equal to sum of radii of pentamer and hexamer
    eq2 = rp[1] + rh[1]
    eq3 = rp[2] + rh[2]
    eq4 = rp[3] + rh[3]
    eq5 = rp[4] + rh[4]
    potential.pair_coeff.set(['HA2'],['PA2'], D0=E_ph, alpha=rho_cc/eq2, r0=eq2, r_cut=2+eq2)
    potential.pair_coeff.set(['HA3'],['PA3'], D0=E_ph, alpha=rho_cc/eq3, r0=eq3, r_cut=2+eq3)
    potential.pair_coeff.set(['HA4'],['PA4'], D0=E_ph, alpha=rho_cc/eq4, r0=eq4, r_cut=2+eq4)
    potential.pair_coeff.set(['HA5'],['PA5'], D0=E_ph, alpha=rho_cc/eq5, r0=eq5, r_cut=2+eq5)

    #print a message with the changed potential values
    print("HH strength changed to {}".format(E_hh))
    print("HP strength changed to {}".format(E_ph))
    return

def protocolRun(protocol, potential, rho_cc, rh, rp, lag, animation_time, dt):
    #run a sequence of simulations according to the protocol supplied

    #get the sequence of protocol values for each paramter
    HP = protocol[0]
    HH = protocol[1]

    #get the number of time steps to simulate for each value
    interval_length = lag * animation_time
    num_timesteps   = int(interval_length / dt)

    #quiet hoomd so it doesnt print out every potential change
    hoomd.util.quiet_status()

    #loop over each interval, set parameters, and run
    for i in range(len(HP)):

        #get the parameters 
        HP_attraction = HP[i]
        HH_attraction = HH[i]

        #modify the potentials to reflect this 
        modify_potential(potential, HH_attraction, HP_attraction, rho_cc, rh, rp)

        #print progress update
        print("Beginning simulation on interval {} of {}".format(i+1, len(HP)))

        #run for the given number of time steps
        hoomd.run(num_timesteps)

    #unquiet hooms notifications
    hoomd.util.unquiet_status()

    return


def simulate():
    #set up and perform the simulation

    #read input files
    try:
        input_file = sys.argv[1]          # Get input file name
        config = ConfigDict(input_file)   # Read input parameter file
        seed = int(sys.argv[2])           # Get unique random seed
    except:
        print("Usage: %s <input_file> <seed>" % sys.argv[0])
        raise

    #load in the pickled protocol
    try:
        #get the protocol according to the input file number
        file_num = input_file.split("input")[-1].split(".txt")[0]
        prot_file = "input_opt/protocol" + file_num
        with open(prot_file, 'rb') as f:
            protocol = pickle.load(f)
            print("Protocol loaded.")
    except:
        print("Protocol could not be loaded. Exiting.")
        raise()

    # Read in parameters from input file
    num_capsomers       = int(  config['Capsomers'])
    sphere_attraction   = float(config['Particle Bond Strength'])
    Rs                  = float(config['Sphere Radius'])
    sigma_cone          = float(config['Cone Diameter'])
    box_size            = float(config['Box Size'])
    ts                  = float(config['Time Step'])
    animation_time      = float(config['Animation Time'])
    lag                 = float(config['MSM Lag'])
    kT                  = float(config['kT'])

    h                   = float(config['Height'])
    angle               = float(config['Alpha'])
    rho_cc              = float(config['Alpha Morse CC'])
    rho_nc              = float(config['Alpha Morse NC'])
    size_ratio          = float(config['Size Ratio'])

    #init the ixn strengths to 0. will be set by protocol
    HH_attraction       = 0.1
    HP_attraction       = 0.1

    #get the radii of each bead in the hexamer cones
    angle_eff = angle * np.pi / 180.0
    r5 = sigma_cone / 2.0
    R_cone = r5 / np.sin(angle_eff) - h
    heights = [i*h/5.0 for i in range(6)]
    rh = (heights + R_cone) * np.sin(angle_eff)

    #get the radii of beads in the pentamers cones by scaling by 0.77
    rp = rh * size_ratio

    #init hoomd
    hoomd.context.initialize("--notice-level=1")

    #initialize the system
    global system
    system = init_sim(num_capsomers, box_size, rh, rp, h, R_cone, Rs, seed)

    #construct initial potential
    nl, morse, lj = create_potential(rh, rp, Rs, HH_attraction, HP_attraction, rho_cc, 
                                     sphere_attraction,  rho_nc)

    #create the temporal integrator
    outfile = create_integrator(ts, animation_time, kT, seed, file_num)
    print(outfile)

    #run the simulation according to the protocol
    protocolRun(protocol, morse, rho_cc, rh, rp, lag, animation_time, ts)

    #return the output gsd file for pathway analysis
    return outfile


if __name__ == "__main__":

   #do a simulation
   simulate()
