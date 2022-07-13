'''
This script simulates the assembly of conical subunits around a spherical nanoparticle,
similar to the 2018 Lazaro soft matter paper. These simulations have two subunit types,
representing hexamers and pentamers. The strength of the attractive potential between 
types is set as an input parameter.

Supports performing single long trajectories, initialized via placing particles 
randomly in the domain without overlap, or by performing an equilibration phase. 

Also supports initialization from a snapshot, which is used to perform ratcheting
to get simulation data from poorly sampled states. NOTE: ratcheting gave weird results 
when constructing MSMs, so be wary of using it. I believe it was just because the chosen
lag time was too small, but I have not verified this. 
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

from config_dict import ConfigDict
from init import init_sim
from init import init_sim_from_snap


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


def create_integrator(ts, animation_time, E_hh, E_ph, kT, seed, short = False, longSim = False):
    #construct the time integrator for the simulation
    #makes rigid bodies the integration group, 
    #sets an output file based on where its being run and returns the path to it

    #test if running on expanse
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

    # integrate rigid and non rigid structures together
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
        parameter_folder = "P{}H{}/".format(E_ph, E_hh)
        if (short):
            parameter_folder += 'short/'
        elif (longSim):
            parameter_folder += 'long/'
        #check that this folder exists
        if (not os.path.exists(path+parameter_folder)):
            os.makedirs(path+parameter_folder)

    #finish naming the output trajectory file
    if (short):
        #if ratcheting in parallel, set a temporary random string as the filename
        #to avoid race conditions in naming and overwriting.
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for i in range(16))
        out_gsd = path + parameter_folder + random_string + ".gsd"
        out_log = path + parameter_folder + random_string + ".log"
    else:
        #simply append the seed number as the trajectory name
        traj_num = "traj{}".format(seed)
        out_gsd = path + parameter_folder + traj_num + ".gsd"
        out_log = path + parameter_folder + traj_num + ".log"

    # dump the configuration every animation_period
    hoomd.dump.gsd(filename=out_gsd, period=animation_period, group=hoomd.group.all(), 
                    overwrite=True, phase=0)

    # output the energy and temp in a log file
    hoomd.analyze.log(filename=out_log, quantities=['temperature', 'potential_energy'],
                      period=animation_period, overwrite=True, phase=0)

    #set the integration mode - langevin
    integrator_mode = md.integrate.mode_standard(dt=ts)
    md.integrate.langevin(group=group_integrate, kT=kT, seed=seed, dscale=True)
    #md.integrate.brownian(group=group_integrate, kT=kT, seed=seed, dscale=True)

    #return the output file path
    return out_gsd

def simulate():
    #set up and perform the simulation

    #read input files
    try:
        config = ConfigDict(sys.argv[1])  # Read input parameter file
        seed = int(sys.argv[2])           # Get unique random seed
    except:
        print("Usage: %s <input_file> <seed>" % sys.argv[0])
        raise

    # Read in parameters from input file
    num_capsomers       = int(config['Capsomers'])
    HH_attraction       = float(config['HH Bond Strength'])
    HP_attraction       = float(config['HP Bond Strength'])
    sphere_attraction   = float(config['Particle Bond Strength'])
    Rs                  = float(config['Sphere Radius'])
    sigma_cone          = float(config['Cone Diameter'])
    box_size            = float(config['Box Size'])
    ts                  = float(config['Time Step'])
    Tf                  = float(config['Final Time'])
    animation_time      = float(config['Animation Time'])
    kT                  = float(config['kT'])

    h                   = float(config['Height'])
    angle               = float(config['Alpha'])
    rho_cc              = float(config['Alpha Morse CC'])
    rho_nc              = float(config['Alpha Morse NC'])
    size_ratio          = float(config['Size Ratio'])

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

    #construct potential
    nl, morse, lj = create_potential(rh, rp, Rs, HH_attraction, HP_attraction, rho_cc, 
                                     sphere_attraction,  rho_nc)

    #create the temporal integrator
    outfile = create_integrator(ts, animation_time, HH_attraction, HP_attraction, kT, seed)
    print(outfile)

    #set the final time and run
    num_steps = int(Tf / ts)
    hoomd.run(num_steps)

    #return the path to the output gsd file for analysis
    return outfile


def simulateFromSnap(snap, seed, longSim=False, LAG=100, num_lags=30):
    #set up and perform the simulation

    #read input files
    try:
        config = ConfigDict(sys.argv[1])  # Read input parameter file
    except:
        print("Usage: %s <input_file> <seed>" % sys.argv[0])
        raise

    # Read in parameters from input file
    num_capsomers       = int(config['Capsomers'])
    HH_attraction       = float(config['HH Bond Strength'])
    HP_attraction       = float(config['HP Bond Strength'])
    sphere_attraction   = float(config['Particle Bond Strength'])
    Rs                  = float(config['Sphere Radius'])
    sigma_cone          = float(config['Cone Diameter'])
    box_size            = float(config['Box Size'])
    ts                  = float(config['Time Step'])
    Tf                  = float(config['Final Time'])
    animation_time      = float(config['Animation Time'])
    kT                  = float(config['kT'])

    h                   = float(config['Height'])
    angle               = float(config['Alpha'])
    rho_cc              = float(config['Alpha Morse CC'])
    rho_nc              = float(config['Alpha Morse NC'])
    size_ratio          = float(config['Size Ratio'])

    #get the radii of each bead in the hexamer cones
    angle_eff = angle * np.pi / 180.0
    r5 = sigma_cone / 2.0
    R_cone = r5 / np.sin(angle_eff) - h
    heights = [i*h/5.0 for i in range(6)]
    rh = (heights + R_cone) * np.sin(angle_eff)

    #get the radii of beads in the pentamers cones by scaling by 0.77 (or other size ratio)
    rp = rh * size_ratio

    #set final time in accordance with a long sim (input file) or short sim (32 lags)
    if (longSim):
        short = False
    else:
        short    = True
        Tf = LAG * num_lags * animation_time

    #do the initialization
    global system
    system = init_sim_from_snap(snap, box_size, rp, rh, R_cone, h, Rs) 

    #construct potential
    nl, morse, lj = create_potential(rh, rp, Rs, HH_attraction, HP_attraction, rho_cc, 
                                     sphere_attraction,  rho_nc)

    #create the temporal integrator
    outfile = create_integrator(ts, animation_time, HH_attraction, HP_attraction, kT, seed, 
                                short=short, longSim = longSim)
    print(outfile)

    #set the final time and run
    num_steps = int(Tf / ts)
    hoomd.run(num_steps, profile=False)

    #return the path to the output gsd file for analysis
    return outfile



if __name__ == "__main__":

   #do a simulation
   simulate()
