'''
This script looks at gsd files that are known to contain a specific final state
and creates a snapshot of this state to use as an initial frame for a new simulation. 
Ensure that the final frame does indeed contain the desired state before use. 
'''

import os
import sys
import math
import numpy as np

import hoomd
import hoomd.md as md
import gsd
import gsd.hoomd

import pandas as pd

from itertools import groupby
import scipy
from scipy import special

import pickle

##################################################################
############# Global Variable Declarations #######################
##################################################################

#define a bond disance cutoff for the A5 particles based on radius and potential
r5 = (3.85 + 4.0*0.8) * np.sin(19.9*np.pi/180) #this is wrong Rcone
alpha = 12
alpha_eff = 12.0/(2*r5)
search_cutoff = 2*r5 + 2.1/alpha_eff
print(search_cutoff, 2*r5)
search_cutoff = 10.3

#Define a search radius around the sphere center for A5 particles
sphere_radius = 8.3
height        = 4.0
morse_range   = 1.2
search_radius = sphere_radius + 0.3 + morse_range + height + 0.2

##################################################################
############# Pathway Analysis Functions  ########################
##################################################################


def distance(x0, x1, dimensions):
    #get the distance between the points x0 and x1
    #assumes periodic BC with box dimensions given in dimensions

    #get distance between particles in each dimension
    delta = np.abs(x0 - x1)

    #if distance is further than half the box, use the closer image
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)

    #compute and return the distance between the correct set of images
    return np.sqrt((delta ** 2).sum(axis=-1))


def get_particles(traj, frame):
    #return the needed info to track assembly from a trajectory frame

    #get the data from the current frame
    data = traj.read_frame(frame)

    #get the relevant data for each particle
    #positions need to be seperated in each coordinate
    particle_info = {
        'type': [data.particles.types[typeid] 
                 for typeid in data.particles.typeid],
        'body': data.particles.body,
        'position_x': data.particles.position[:, 0],
        'position_y': data.particles.position[:, 1],
        'position_z': data.particles.position[:, 2],
    }

    #return a dataframe with the relevant info for each particle
    return pd.DataFrame(particle_info)


def get_S_center(particle_info):
    #get center of the scaffold particle from particle_info

    #get the array location of the S type particle
    S_particles = particle_info.loc[particle_info['type'] == 'S']

    #grab the coordinates from this array location
    S_coords = np.array([S_particles['position_x'].values, 
                         S_particles['position_y'].values, 
                         S_particles['position_z'].values]).T

    return S_coords


def get_A5_contacts(particle_info, box_dim, size_ratio, center=None, radius=None):
    #determine how many A5 sites are in contact around the sphere
    #can only consider particles in a cutoff sphere for efficiency

    #set cutoff distance for bond formation
    HH_length = search_cutoff
    HP_length = HH_length * size_ratio

    #get the A5 particle coordinates
    PA5_particles = particle_info.loc[(particle_info['type'] == 'PA5')]
    PA5_coords = np.array([np.array(PA5_particles['position_x'].values), 
                          np.array(PA5_particles['position_y'].values), 
                          np.array(PA5_particles['position_z'].values)]).T

    HA5_particles = particle_info.loc[(particle_info['type'] == 'HA5')]
    HA5_coords = np.array([np.array(HA5_particles['position_x'].values), 
                          np.array(HA5_particles['position_y'].values), 
                          np.array(HA5_particles['position_z'].values)]).T

    #if a cutoff sphere is given, exclude particles outside this sphere
    if (radius and center.all()):
        PA5_coords = PA5_coords[np.where(distance(PA5_coords, center, box_dim) < radius)]
        HA5_coords = HA5_coords[np.where(distance(HA5_coords, center, box_dim) < radius)]

    #get the number of attached subunits
    attached_P = len(PA5_coords)
    attached_H = len(HA5_coords)

    #init an array for distances, particle pairs, and bond num
    distancesHH = []
    distancesHP = []
    pairsHH     = []
    pairsHP     = []

    #loop over each particles, compute distances to all others 
    for i in range(len(HA5_coords)):
        #distances to hexamers
        for j in range(i+1,len(HA5_coords)):
            distancesHH.append(distance(HA5_coords[i], HA5_coords[j], box_dim))
            pairsHH.append((i,j))

        #distances to pentamers
        for j in range(0,len(PA5_coords)):
            distancesHP.append(distance(HA5_coords[i], PA5_coords[j], box_dim))
            pairsHP.append((i,j))

    #check if each distance is less than the bond length. count how many bonds per type
    bondedParticlesPos = []
    num_HH = 0
    num_HP = 0
    for i in range(len(distancesHH)):
        if (distancesHH[i] < HH_length):
            num_HH += 1
            bondPair = pairsHH[i]
            bondedParticlesPos.append([HA5_coords[bondPair[0]], HA5_coords[bondPair[1]]])

    HPbonds = []
    for i in range(len(distancesHP)):
        if (distancesHP[i] < HP_length):
            num_HP += 1
            bondPair = pairsHP[i]
            bondedParticlesPos.append([HA5_coords[bondPair[0]], PA5_coords[bondPair[1]]])
            HPbonds.append(bondPair[0])

    #evaluate the spherical harmonic order parameters
    # print(center)
    # allPos = np.concatenate([PA5_coords, HA5_coords])
    # Q = evalBOOPsCenter(allPos, center[0])
    # print(Q)

    #get the number of hexamers that are adjacent to two pentamers (30 = perfect icosahedral)
    grouped_L = [sum(1 for _ in group) for _, group in groupby(HPbonds)]
    correct_hex = grouped_L.count(2)


    op = (attached_P, attached_H, num_HP, num_HH, correct_hex)
    return op


def getState(filename, frame):
    #construct a trajectory in (n_s, n_b, n_p, n_h) space from simulation results

    #check if the filename calls for a different size_ratio
    size_check = filename.split('s0')
    if (len(size_check) > 1):
        size_ratio = float(size_check[1].split('/')[0])
    else:
        size_ratio = 0.77

    #get the collection of snapshots and get number of frames
    snaps = gsd.hoomd.open(name=filename, mode="rb")
    frames = len(snaps)

    #get box configuration
    box = snaps[0].configuration.box
    box_dim = np.array([box[0], box[1], box[2]])

    #get the particle info for the current frame
    particle_info = get_particles(snaps, frame)

    #get the location of the scafold particle center
    S_c = get_S_center(particle_info)

    #analyze the capsid at this frame and store the order parameter tuple
    q = get_A5_contacts(particle_info, box_dim, size_ratio, center=S_c, radius=search_radius)

    #ask if a snapshot is desired
    while True:
        x = input("The state is {}. Would you like a snapshot of this frame? (Y/N)".format(q))
        if (x == "Y"):
            return True, q
        elif (x == "N"):
            return False, q



if __name__ == "__main__":

    #read in a filename from command line
    try:
        gsd_file = sys.argv[1]  #get filename
    except:
        print("Usage: %s <gsd_file> " % sys.argv[0])
        raise


    #try to read the file
    try:
        snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    except:
        print("The file could not be read by gsd. Make sure the correct file was given")

    #extract the final snapshot
    L = len(snaps)
    frame = L-1
    snap  = snaps.read_frame(frame)
    
    #determine if we want the snap
    proceed, q = getState(gsd_file, frame)

    if (proceed):
        #pickle the state
        with open("state{}_{}_{}".format(q[0],q[1],q[-1]), 'wb') as f:
            pickle.dump(snap, f)

        #test loading the pickle
        with open("state{}_{}_{}".format(q[0],q[1],q[-1]), 'rb') as f:
            snap_test = pickle.load(f)

        print("Testing: ", snap_test.particles.position == snap.particles.position)

