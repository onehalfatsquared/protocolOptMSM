'''
This file contains functions to perform initialization for the cones simulation. 

First, we try to place the required number of particles in the domain by generating spheres
of the given radius, and using rejection sampling to place them without overlap. We then run
short BD sims with repulsive interactions to ensure there is no overlap. 

If this method fails, we instead place the particles on an equally spaced lattice, 
and randomize the idenities so the two subunit types are relatively uniformly distributed.
We then perform a longer BD equilibration run with repulsive interactions to init the system.

In the case of ratcheting, starting from the same initial bathcondition can bias the sampling. 
When starting from a snapshot, we do a BD equilibration, but first identify the cluster in 
order to hold those particles in place during the initialization. This equilibrates the bath 
while leaving the cluster intact. 
'''


import os
import sys
import math
import numpy as np
import random

import hoomd
import hoomd.md as md
import gsd
import gsd.hoomd

import pandas as pd


def genSpheres(N, box_dim, R):
    #generate N spheres of radius R in a given box - periodic BC

    #init a list to store particles
    positions = []

    #subtract R/2 from box dimensions to satisfy no overlap for particles close to edges
    effective_box = np.array(box_dim) - R / 2.0

    #get the first particle
    pos = np.multiply(effective_box, [np.random.random() for i in range(3)]) + R / 4.0
    positions.append(pos)
    distributed = 1

    #get new particles, check for overlap
    loop_count = 0
    loop_tol   = 50000
    while (distributed < N and loop_count < loop_tol):
        pos = np.multiply(effective_box, [np.random.random() for i in range(3)]) + R/4.0
        overlap = False
        for particle in positions:
            d = np.linalg.norm(pos-particle)
            if (d < 2*R):
                overlap=True
                break
        if (not overlap):
            positions.append(pos)
            distributed += 1

        loop_count += 1

    if (loop_count < loop_tol):
        print("Generated configuration in {} iterations".format(loop_count))
        for i in range(N):
            positions[i] = positions[i].flatten().tolist()
        success = True
    else:
        print("Warning: could not generate configuration in allotted time.\n" + 
                 "Please reduce N or use a bigger box.\n" + 
                 "Defaulting to lattice construction + equilibration")
        success = False

    return positions, success


def equally_spaced_positions(N, box_size):
    #generate positions of particles such that they are 'equally' spaced in the given box

    #define a tolerance from the edge of the box
    edge_tol = 0.5

    #get an effective box by subtracting twice the edge_tol
    box_eff = box_size - 2 * edge_tol

    #take cube root of N to get how many in each dimension - round up to overcount
    n = int(np.ceil(np.cbrt(N)))
    tot = n*n*n

    #create a 1d mesh, make 3d with meshgrid
    x1 = np.linspace(-box_eff / 2.0, box_eff / 2, n)
    x, y, z = np.meshgrid(x1, x1, x1)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    #generate a random permutation of first tot integers, pick positions in those indices
    perm = np.random.permutation(tot)
    positions = []
    for i in range(N):
        p = perm[i] 
        pos = [x[p], y[p], z[p]]
        positions.append(pos)

    return positions


def specify_particles(system, Rc, h, Rs):
    #defines all the particle types in the system and creates them
    #returns the group of rigid bodies

    #define the constituent particle types for the capsomer
    system.particles.types.add('HA1')
    system.particles.types.add('HA2')
    system.particles.types.add('HA3')
    system.particles.types.add('HA4')
    system.particles.types.add('HA5')
    system.particles.types.add('HA6')

    system.particles.types.add('PA1')
    system.particles.types.add('PA2')
    system.particles.types.add('PA3')
    system.particles.types.add('PA4')
    system.particles.types.add('PA5')
    system.particles.types.add('PA6')

    #compute the heights of each capsomer bead. the positions of the CoM of bead i is 
    #(0, 0, h_i) but it needs to be set in relation to the CoM of the cone, 
    #which is (0, 0, h/4) from the bead with the largest radius
    z = [-0.25 * Rc + (0.75 - (i-1.0)/5.0)*h for i in range(6)]
    pos_c = []
    for i in range(6):
        pos_c.append([0.0, 0.0, z[i]])

    #add in the particle types for the spheres
    system.particles.types.add('S')

    #declare rigid body storage
    rigid = hoomd.md.constrain.rigid()

    #Define capsomer rigid bodies
    rigid.set_param('H',
                    types=['HA1', 'HA2', 'HA3', 'HA4', 'HA5', 'HA6'],
                    positions=pos_c)

    rigid.set_param('P',
                    types=['PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6'],
                    positions=pos_c)

    #create the bodies and return them
    rigid.create_bodies()
    return rigid

def center_list(system):
    #return a list of all tags for the particle centers

    N = len(system.particles)
    ids = []
    for i in range(N):
        if (system.particles[i].type == 'P' or system.particles[i].type == 'S' or 
            system.particles[i].type == 'H'):
            ids.append(system.particles[i].tag)

    return ids

def get_eq_force(rp, rh, Rs):
    #define a repuslive force to be used for equilibration

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
    rep = 1.0

    #create the lennard jones potential - shifted to 0 at cutoff
    lj_eq = md.pair.lj(r_cut = 2.0 * rh6, nlist=nl)
    lj_eq.set_params(mode="shift")

    #init all pair interactions to zero
    lj_eq.pair_coeff.set(['H','P','S','PA1','PA2','PA3','PA4','PA5','PA6','HA1','HA2','HA3','HA4','HA5','HA6'],
                      ['H','P','S','PA1','PA2','PA3','PA4','PA5','PA6','HA1','HA2','HA3','HA4','HA5','HA6'],
                      alpha=0, epsilon=0, r_cut=0, sigma=0)

    #cones - excluded volume ixns
    #pentamer-pentamer
    lj_eq.pair_coeff.set('PA1', 'PA1', alpha=0.0, epsilon=rep, r_cut=rp1+rp1, sigma=rp1+rp1)
    lj_eq.pair_coeff.set('PA1', 'PA2', alpha=0.0, epsilon=rep, r_cut=rp1+rp2, sigma=rp1+rp2)
    lj_eq.pair_coeff.set('PA1', 'PA3', alpha=0.0, epsilon=rep, r_cut=rp1+rp3, sigma=rp1+rp3)
    lj_eq.pair_coeff.set('PA1', 'PA4', alpha=0.0, epsilon=rep, r_cut=rp1+rp4, sigma=rp1+rp4)
    lj_eq.pair_coeff.set('PA1', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp1+rp5, sigma=rp1+rp5)
    lj_eq.pair_coeff.set('PA1', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp1+rp6, sigma=rp1+rp6)
    lj_eq.pair_coeff.set('PA2', 'PA3', alpha=0.0, epsilon=rep, r_cut=rp2+rp3, sigma=rp2+rp3)
    lj_eq.pair_coeff.set('PA2', 'PA4', alpha=0.0, epsilon=rep, r_cut=rp2+rp4, sigma=rp2+rp4)
    lj_eq.pair_coeff.set('PA2', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp2+rp5, sigma=rp2+rp5)
    lj_eq.pair_coeff.set('PA2', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp2+rp6, sigma=rp2+rp6)
    lj_eq.pair_coeff.set('PA3', 'PA4', alpha=0.0, epsilon=rep, r_cut=rp3+rp4, sigma=rp3+rp4)
    lj_eq.pair_coeff.set('PA3', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp3+rp5, sigma=rp3+rp5)
    lj_eq.pair_coeff.set('PA3', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp3+rp6, sigma=rp3+rp6)
    lj_eq.pair_coeff.set('PA4', 'PA5', alpha=0.0, epsilon=rep, r_cut=rp4+rp5, sigma=rp4+rp5)
    lj_eq.pair_coeff.set('PA4', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp4+rp6, sigma=rp4+rp6)
    lj_eq.pair_coeff.set('PA5', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp6+rp5, sigma=rp6+rp5)
    lj_eq.pair_coeff.set('PA6', 'PA6', alpha=0.0, epsilon=rep, r_cut=rp6+rp6, sigma=rp6+rp6)

    #hexamer-hexamer
    lj_eq.pair_coeff.set('HA1', 'HA1', alpha=0.0, epsilon=rep, r_cut=rh1+rh1, sigma=rh1+rh1)
    lj_eq.pair_coeff.set('HA1', 'HA2', alpha=0.0, epsilon=rep, r_cut=rh1+rh2, sigma=rh1+rh2)
    lj_eq.pair_coeff.set('HA1', 'HA3', alpha=0.0, epsilon=rep, r_cut=rh1+rh3, sigma=rh1+rh3)
    lj_eq.pair_coeff.set('HA1', 'HA4', alpha=0.0, epsilon=rep, r_cut=rh1+rh4, sigma=rh1+rh4)
    lj_eq.pair_coeff.set('HA1', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh1+rh5, sigma=rh1+rh5)
    lj_eq.pair_coeff.set('HA1', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh1+rh6, sigma=rh1+rh6)
    lj_eq.pair_coeff.set('HA2', 'HA3', alpha=0.0, epsilon=rep, r_cut=rh2+rh3, sigma=rh2+rh3)
    lj_eq.pair_coeff.set('HA2', 'HA4', alpha=0.0, epsilon=rep, r_cut=rh2+rh4, sigma=rh2+rh4)
    lj_eq.pair_coeff.set('HA2', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh2+rh5, sigma=rh2+rh5)
    lj_eq.pair_coeff.set('HA2', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh2+rh6, sigma=rh2+rh6)
    lj_eq.pair_coeff.set('HA3', 'HA4', alpha=0.0, epsilon=rep, r_cut=rh3+rh4, sigma=rh3+rh4)
    lj_eq.pair_coeff.set('HA3', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh3+rh5, sigma=rh3+rh5)
    lj_eq.pair_coeff.set('HA3', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh3+rh6, sigma=rh3+rh6)
    lj_eq.pair_coeff.set('HA4', 'HA5', alpha=0.0, epsilon=rep, r_cut=rh4+rh5, sigma=rh4+rh5)
    lj_eq.pair_coeff.set('HA4', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh4+rh6, sigma=rh4+rh6)
    lj_eq.pair_coeff.set('HA5', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh6+rh5, sigma=rh6+rh5)
    lj_eq.pair_coeff.set('HA6', 'HA6', alpha=0.0, epsilon=rep, r_cut=rh6+rh6, sigma=rh6+rh6)

    #hexamer-pentamer
    lj_eq.pair_coeff.set('PA1', 'HA1', alpha=0.0, epsilon=rep, r_cut=rp1+rh1, sigma=rp1+rh1)
    lj_eq.pair_coeff.set('PA1', 'HA2', alpha=0.0, epsilon=rep, r_cut=rp1+rh2, sigma=rp1+rh2)
    lj_eq.pair_coeff.set('PA1', 'HA3', alpha=0.0, epsilon=rep, r_cut=rp1+rh3, sigma=rp1+rh3)
    lj_eq.pair_coeff.set('PA1', 'HA4', alpha=0.0, epsilon=rep, r_cut=rp1+rh4, sigma=rp1+rh4)
    lj_eq.pair_coeff.set('PA1', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp1+rh5, sigma=rp1+rh5)
    lj_eq.pair_coeff.set('PA1', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp1+rh6, sigma=rp1+rh6)
    lj_eq.pair_coeff.set('PA2', 'HA3', alpha=0.0, epsilon=rep, r_cut=rp2+rh3, sigma=rp2+rh3)
    lj_eq.pair_coeff.set('PA2', 'HA4', alpha=0.0, epsilon=rep, r_cut=rp2+rh4, sigma=rp2+rh4)
    lj_eq.pair_coeff.set('PA2', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp2+rh5, sigma=rp2+rh5)
    lj_eq.pair_coeff.set('PA2', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp2+rh6, sigma=rp2+rh6)
    lj_eq.pair_coeff.set('PA3', 'HA4', alpha=0.0, epsilon=rep, r_cut=rp3+rh4, sigma=rp3+rh4)
    lj_eq.pair_coeff.set('PA3', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp3+rh5, sigma=rp3+rh5)
    lj_eq.pair_coeff.set('PA3', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp3+rh6, sigma=rp3+rh6)
    lj_eq.pair_coeff.set('PA4', 'HA5', alpha=0.0, epsilon=rep, r_cut=rp4+rh5, sigma=rp4+rh5)
    lj_eq.pair_coeff.set('PA4', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp4+rh6, sigma=rp4+rh6)
    lj_eq.pair_coeff.set('PA5', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp6+rh5, sigma=rp6+rh5)
    lj_eq.pair_coeff.set('PA6', 'HA6', alpha=0.0, epsilon=rep, r_cut=rp6+rh6, sigma=rp6+rh6)

    #nanoparticle - excluded volume ixns
    lj_eq.pair_coeff.set(['S'], ['PA1'], alpha=0, epsilon=rep, r_cut=Rs+rp1, sigma=Rs+rp1)
    lj_eq.pair_coeff.set(['S'], ['PA2'], alpha=0, epsilon=rep, r_cut=Rs+rp2, sigma=Rs+rp2)
    lj_eq.pair_coeff.set(['S'], ['PA3'], alpha=0, epsilon=rep, r_cut=Rs+rp3, sigma=Rs+rp3)
    lj_eq.pair_coeff.set(['S'], ['PA4'], alpha=0, epsilon=rep, r_cut=Rs+rp4, sigma=Rs+rp4)
    lj_eq.pair_coeff.set(['S'], ['PA5'], alpha=0, epsilon=rep, r_cut=Rs+rp5, sigma=Rs+rp5)
    lj_eq.pair_coeff.set(['S'], ['PA6'], alpha=0, epsilon=rep, r_cut=Rs+rp6, sigma=Rs+rp6)

    lj_eq.pair_coeff.set(['S'], ['HA1'], alpha=0, epsilon=rep, r_cut=Rs+rh1, sigma=Rs+rh1)
    lj_eq.pair_coeff.set(['S'], ['HA2'], alpha=0, epsilon=rep, r_cut=Rs+rh2, sigma=Rs+rh2)
    lj_eq.pair_coeff.set(['S'], ['HA3'], alpha=0, epsilon=rep, r_cut=Rs+rh3, sigma=Rs+rh3)
    lj_eq.pair_coeff.set(['S'], ['HA4'], alpha=0, epsilon=rep, r_cut=Rs+rh4, sigma=Rs+rh4)
    lj_eq.pair_coeff.set(['S'], ['HA5'], alpha=0, epsilon=rep, r_cut=Rs+rh5, sigma=Rs+rh5)
    lj_eq.pair_coeff.set(['S'], ['HA6'], alpha=0, epsilon=rep, r_cut=Rs+rh6, sigma=Rs+rh6)
    

    return lj_eq

def reduce_overlap(lj_eq, ids, equilibrate=False):
    #equilibrate the system, reducing overlap - stability via force scaling
    #returns true for success, false for a failure

    # integrate rigid and non rigid structures together
    group_rigid = hoomd.group.rigid_center()
    group_nonrigid = hoomd.group.nonrigid()
    group_integrate = hoomd.group.union('integrate', group_rigid, group_nonrigid)

    #set base time step - very small in case of bad overlap
    ts = 1e-6

    #set the maximum allowed force before stopping
    F_tol = 10

    #set the integration mode - brownian with low temp
    integrator_mode = md.integrate.mode_standard(dt=ts)
    bd_int = md.integrate.brownian(group=group_integrate, kT=0.01, seed=0, dscale=True)

    #set the number of steps for each trial and number of trials
    num_steps  = 501
    num_trials = 100
    hoomd.util.quiet_status()    #suppress output for these trials
    for i in range(num_trials):
        #run the simulation
        try:
            if (i == 0):
                hoomd.run(10000)
            else:
                hoomd.run(num_steps)
        except:
            print("Error: Particle overlap is too severe to simulate.\n"+
                 "Consider less particles or a bigger box.\n" + 
                 "Defaulting to lattice construction + equilibration. ")
            return False
            

        #get the maximum force acting on any body
        F = 0
        for particle in ids:
            f = lj_eq.get_net_force(hoomd.group.tags(particle, particle))
            f = np.max(np.abs(f))
            F = np.maximum(f, F)
            print(f)

        #check if the force is small enough to use in simulation with dt = 1e-3
        if (F < F_tol):
            break

        #scale the timestep for the next trial - big as possible but maintain stability
        ts = 1.0 / F
        ts /= 1000
        integrator_mode = md.integrate.mode_standard(dt=ts)

    #equilibrate if the option is set
    if (equilibrate):
        print("\n\nPerforming initial equilibration\n")
        eq_steps = 100000
        ts       = 1e-4

        integrator_mode = md.integrate.mode_standard(dt=ts)
        hoomd.run(eq_steps)

    #resume hoomd output
    hoomd.util.unquiet_status()

    #disable the bd integator
    bd_int.disable()


    #print progress message
    if (i < num_trials):
        print("Equilibrated in {} BD trials, Max force is {}".format(i, F))
    else:
        print("Error: Particle overlap is too severe to simulate.\n"+
                 "Consider less particles or a bigger box.\n" + 
                 "Defaulting to lattice construction + equilibration. ")
        return False

    return True


def init_sim(num_capsomers, box_size, rp, rh,  h, R_cone, Rs, seed=None):
    '''
    Initialize the simulation of pentagonal capsomer self assembly
    Attempts a uniformly random assignment, checks for overlaps
    If previous method fails, assigns a lattice and performs equilibration
    '''

    #construct the rng
    if (seed is not None):
        np.random.seed(seed)
    else:
        np.random.seed()

    #construct moment of inertia in the CoM frame of the cone - assume uniform
    #ixx = iyy = 3/20R^2 + 3/80h^2, izz = 3/10R^2
    cone_base_h   = rh[5]
    cone_height = R_cone + h
    hxx = 0.15 * cone_base_h * cone_base_h + 0.0375 * cone_height * cone_height
    hyy = hxx
    hzz = 0.3 * cone_base_h * cone_base_h

    cone_base_p   = rp[5]
    cone_height = R_cone + h
    pxx = 0.15 * cone_base_p * cone_base_p + 0.0375 * cone_height * cone_height
    pyy = pxx
    pzz = 0.3 * cone_base_p * cone_base_p

    #set the stoichiometry of the two capsomer types
    #42 unit capsid has 12 pentagons, 30 hexagons. 12/42 -> about 0.3 pentamers
    num_p = int(0.3 * num_capsomers)
    num_h = num_capsomers - num_p

    #define the moment of inertia and type of the capsomers
    moi                  = [[pxx, pyy, pzz]] * num_p + [[hxx, hyy, hzz]] * num_h
    types                = ['P'] * num_p + ['H'] * num_h

    #append the sacaffold particle to moi and types
    moi.append([1,1,1])
    types.append('S')

    #generate spheres randomly in the box, trying to avoid overlap
    position, success    = genSpheres(num_capsomers+1, [box_size]*3, 3)

    if (success):
        #continue with the init

        #get a unitcell
        uc = hoomd.lattice.unitcell(N=num_capsomers+1,
                                    a1=[box_size,0,0], a2=[0,box_size,0], a3=[0,0,box_size],
                                    position=position, type_name=types, moment_inertia=moi)

        #make a lattice - 1 copy
        system = hoomd.init.create_lattice(unitcell=uc, n=1)

        #add in the particle types for the spheres
        system.particles.types.add('S')

        #create the rigid bodies and get the group of them
        rigid = specify_particles(system, R_cone, h, Rs)

        #get a repulsive force
        lj_eq = get_eq_force(rp, rh, Rs)

        #get particle center ids
        ids = center_list(system)

        #try to reduce the overlap by simulating with BD
        success = reduce_overlap(lj_eq, ids)

        #remove the repulsive force from the system
        lj_eq.disable()
        

    if (not success):
        #perform an equilibration scheme

        #create a new context
        hoomd.context.initialize("")

        #generate positions for equally spaced particles
        positionE = equally_spaced_positions(num_capsomers+1, box_size)

        #create a unit cell with particles
        uc = hoomd.lattice.unitcell(N=num_capsomers+1,
                                    a1=[box_size,0,0], a2=[0,box_size,0], a3=[0,0,box_size],
                                    position=positionE, type_name=types, moment_inertia=moi)

        #make a lattice - 1 copy
        system = hoomd.init.create_lattice(unitcell=uc, n=1)

        #add in the particle types for the spheres
        system.particles.types.add('S')

        #create the rigid bodies and get the group of them
        rigid = specify_particles(system, Rc, h, Rs)

        #get a repulsive force
        lj_eq = get_eq_force(rp, rh, Rs)

        #get particle center ids
        ids = center_list(system)

        #try to reduce the overlap by simulating with BD
        success = reduce_overlap(lj_eq, ids, equilibrate=True)

        #remove the repulsive force from the system
        lj_eq.disable()


    if (not success):
        #if both methods fail, give up
        print("Both initialization methods have failed. Exiting...")
        sys.exit()


    '''
    At this point, a set of coordinates has been generated with non-overlapping points
    and orientations, but the equilibrating simulation has stored accelerations that may
    cause a new integrator to blow up. Thus we perform another initialization and create a 
    system with the same positions, but empty velocity/acceleration for the true simulation.
    '''

    #initialize a new context and set the IC with equilibrated positions/properties
    hoomd.context.initialize("")

    #get the spatial properties from the old sim data
    orientation = np.zeros((num_capsomers+1,4))
    for i in range(num_capsomers+1):
        if (system.particles[i].type == 'H' or system.particles[i].type == 'P' or 
            system.particles[i].type == 'S'):
            position[i]    = system.particles[i].position
            orientation[i] = system.particles[i].orientation

    #get a unitcell
    uc = hoomd.lattice.unitcell(N=num_capsomers+1,
                                a1=[box_size,0,0], a2=[0,box_size,0], a3=[0,0,box_size],
                                position=position, type_name=types, moment_inertia=moi,
                                orientation=orientation)

    #make a lattice - 1 copy
    new_system = hoomd.init.create_lattice(unitcell=uc, n=1)

    #add in the particle types for the spheres - may be redundant
    new_system.particles.types.add('S')

    #create the rigid bodies at body centers
    specify_particles(new_system, R_cone, h, Rs)
            
    #return the new system
    return new_system


'''
The rest of this file contains functions to init a simulation from a snapshot
of the system. It performs an equilibration by resetting the position and orientation 
of the scaffold and nearby particles each time step.
'''

def distance(x0, x1, dimensions):
    #get the distance between the points x0 and x1
    #assumes periodic BC with box dimensions given in dimensions

    #get distance between particles in each dimension
    delta = np.abs(x0 - x1)

    #if distance is further than half the box, use the closer image
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)

    #compute and return the distance between the correct set of images
    return np.sqrt((delta ** 2).sum(axis=-1))

def getCoordinates(particles):
    #get coordinates of particles from a pandas particle array

    coords = np.array([particles['position_x'].values,
                       particles['position_y'].values,
                       particles['position_z'].values]).T

    return coords

def hold_cluster(system):
    #identify cluster around the nanparticle and keep the positions constant

    #suppress output for modification step
    hoomd.util.quiet_status()    

    #take a snapshot of the system
    data = system.take_snapshot()

    #get the dimensions fo the box
    box_dim = data.box.Lx

    #positions need to be seperated in each coordinate
    particle_info = {
        'type': [data.particles.types[typeid] 
                 for typeid in data.particles.typeid],
        'body': data.particles.body,
        'position_x': data.particles.position[:, 0],
        'position_y': data.particles.position[:, 1],
        'position_z': data.particles.position[:, 2],
    }
    pdata = pd.DataFrame(particle_info)
    #print('all\n', pdata)

    #get the position of the scaffold particle
    S_particle = pdata.loc[pdata['type'] == 'S']

    #grab the coordinates from this array location
    S_coords   = getCoordinates(S_particle)
    S_index    = S_particle.index.values[0]
    #print('S\n', S_coords, S_index) 

    #get the positions of the HA5 particles
    HA5_particles = pdata.loc[pdata['type'] == 'HA5']

    #grab the coordinates from this array location
    HA5_coords    = getCoordinates(HA5_particles)
    #print('Ha5\n', HA5_coords)

    #get the positions of the PA5 particles
    PA5_particles = pdata.loc[pdata['type'] == 'PA5']

    #grab the coordinates from this array location
    PA5_coords    = getCoordinates(PA5_particles)
    #print('pa5\n', PA5_particles, PA5_coords)

    #set search radius between S center and A5 particles
    search_tol    = 0.5 #add a tolerance due to repulsion force
    search_radius = 14 + search_tol  #warning: this is hard coded for 42 unit capsid

    #get collection of all capsomers that are atached to the scaffold and their body id
    Pdistances = distance(PA5_coords, S_coords, box_dim)
    attached_Pbodies_idx = np.where(Pdistances < search_radius)[0]
    attached_Pbodies     = PA5_particles.iloc[attached_Pbodies_idx]['body']
    #print(attached_Pbodies)

    #get all particles in the P bodies 
    in_Pbodies = pdata.loc[pdata['body'].isin(attached_Pbodies)]
    #print('Pbodies\n', in_Pbodies)

    #do the same for H bodies
    Hdistances = distance(HA5_coords, S_coords, box_dim)
    attached_Hbodies_idx = np.where(Hdistances < search_radius)[0]
    attached_Hbodies     = HA5_particles.iloc[attached_Hbodies_idx]['body']
    in_Hbodies = pdata.loc[pdata['body'].isin(attached_Hbodies)]
    #print('hbodies\n', in_Hbodies)

    #for each of the particles in these bodies, set positions to data0
    for particle in in_Hbodies.index.values:
        system.particles[particle].position = data0.particles.position[particle]
        system.particles[particle].orientation = data0.particles.orientation[particle]

    for particle in in_Pbodies.index.values:
        system.particles[particle].position = data0.particles.position[particle]
        system.particles[particle].orientation = data0.particles.orientation[particle]

    #set the S particle position 
    system.particles[S_index].position = data0.particles.position[S_index]
    system.particles[S_index].orientation = data0.particles.orientation[S_index]

    hoomd.util.unquiet_status()    #unsuppress output 
    return

def hold_cluster_callback(step):
    #this is a wrapper to the hold_cluster function, since it takes an input
    #that is not "step"

    hold_cluster(system)
    return

def init_sim_from_snap(snap, box_size, rp, rh, R_cone, h, Rs):
    #initialize the simulation from particle data in the snapshot

    #get the type id for 'H', 'P', and 'S'
    rigid_ids = []
    rigid_ids.append(np.where(np.array(snap.particles.types) == 'H')[0][0])
    rigid_ids.append(np.where(np.array(snap.particles.types) == 'P')[0][0])
    rigid_ids.append(np.where(np.array(snap.particles.types) == 'S')[0][0])

    #get particle ids for above types
    p_ids = []
    for p_type in rigid_ids:
        p_ids.extend(list(np.where(np.array(snap.particles.typeid) == p_type)[0]))

    #initialize the system from the snapshot - must be done manually using data
    #only grab the H, P, and S type particles
    positions   = [snap.particles.position[particle] for particle in p_ids]
    types       = [snap.particles.types[snap.particles.typeid[particle]] 
                   for particle in p_ids]
    moi         = [snap.particles.moment_inertia[particle] for particle in p_ids]
    orientation = [snap.particles.orientation[particle] for particle in p_ids]
    Np          = len(positions)

    #init hoomd
    hoomd.context.initialize("--notice-level=3")

    #create a unit cell with particles
    uc = hoomd.lattice.unitcell(N=Np,
                                a1=[box_size,0,0], a2=[0,box_size,0], a3=[0,0,box_size],
                                position=positions, type_name=types, moment_inertia=moi,
                                orientation=orientation)
    #make a lattice - 1 copy
    global system
    system = hoomd.init.create_lattice(unitcell=uc, n=1)

    #create the rigid bodies and get the group of them
    rigid = specify_particles(system, R_cone, h, Rs)

    #get the initial configuration into a global variable
    global data0
    data0 = system.take_snapshot()

    #get the equilibration potential
    #get a repulsive force
    lj_eq = get_eq_force(rp, rh, Rs)

    #set up integrator
    group_rigid = hoomd.group.rigid_center()
    group_nonrigid = hoomd.group.nonrigid()
    group_integrate = hoomd.group.union('integrate', group_rigid, group_nonrigid)

    integrator_mode = md.integrate.mode_standard(dt=0.0025)
    seed = random.randint(0,10000000)
    md.integrate.langevin(group=group_integrate, kT=1.0, seed=seed, dscale=True)

    #integrate 
    num_steps = 50 * 100 * 6
    hoomd.analyze.callback(callback=hold_cluster_callback, period=1)
    print("Performing the equilibration")
    hoomd.run(num_steps)

    #remove eq force
    lj_eq.disable()

    #re-init with new positions
    hoomd.context.initialize()

    for i in range(Np):
        if (system.particles[i].type == 'H' or system.particles[i].type == 'P' or 
            system.particles[i].type == 'S'):
            positions[i]   = system.particles[i].position
            orientation[i] = system.particles[i].orientation
            types[i]       = system.particles[i].type
            moi[i]         = system.particles[i].moment_inertia

    #get a unitcell
    uc = hoomd.lattice.unitcell(N=Np,
                                a1=[box_size,0,0], a2=[0,box_size,0], a3=[0,0,box_size],
                                position=positions, type_name=types, moment_inertia=moi,
                                orientation=orientation)

    #make a lattice - 1 copy
    new_system = hoomd.init.create_lattice(unitcell=uc, n=1)
    hold_cluster(new_system)

    #add in the particle types for the spheres
    new_system.particles.types.add('S')

    #create the rigid bodies at body centers
    specify_particles(new_system, R_cone, h, Rs)
    
    #return the system to begin the true simulation
    return new_system



if __name__ == '__main__':

    #test the non-overlapping sphere
    genSpheres(32,[8,8,8], 1.2) 