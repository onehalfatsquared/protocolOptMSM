'''
This script will extract the configuration information for structures that get mapped 
to the T4 target state, (12,30,30), and perform analysis to see if the structures are 
truly T4 or if they are D5. 

The difference to test this is that the pentamer-pentamer distance should be the same
for all nearest pentamer pairs in a T4 cluster, whereas for the D5, some will be shorter 
and some will be longer. By constructing a histogram, or average, of the distances we can 
see which case each cluster falls into, to get an idea of what structure is actually forming. 
'''

import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys
from itertools import groupby
import scipy
from scipy import special

##################################################################
############# Global Variable Declarations #######################
##################################################################

#define a bond disance cutoff for two PA5 particles based on radius
search_cutoff = 15

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
    PP_length = search_cutoff

    #get the A5 particle coordinates
    PA5_particles = particle_info.loc[(particle_info['type'] == 'PA5')]
    PA5_coords = np.array([np.array(PA5_particles['position_x'].values), 
                          np.array(PA5_particles['position_y'].values), 
                          np.array(PA5_particles['position_z'].values)]).T

    #if a cutoff sphere is given, exclude particles outside this sphere
    if (radius and center.all()):
        PA5_coords = PA5_coords[np.where(distance(PA5_coords, center, box_dim) < radius)]

    #get the number of attached subunits
    attached_P = len(PA5_coords)

    #init an array for distances, particle pairs, and bond num
    distancesPP = []
    pairsPP     = []

    #loop over each particles, compute distances to all others 
    for i in range(len(PA5_coords)):
        #distances to pentamers
        for j in range(i+1,len(PA5_coords)):
            distancesPP.append(distance(PA5_coords[i], PA5_coords[j], box_dim))
            pairsPP.append((i,j))

    #check if each distance is less than the bond length. add to list
    pentamer_bond_dists = []
    for i in range(len(distancesPP)):
        if (distancesPP[i] < PP_length):
            pentamer_bond_dists.append(distancesPP[i])

    #get the average
    S = 0
    for dist in pentamer_bond_dists:
    	S += dist

    S = S / len(pentamer_bond_dists)


    
    return S, pentamer_bond_dists
        

def getPentDists(filename):
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
    particle_info = get_particles(snaps, frames-1)
    #particle_info = get_particles(snaps, 500)

    #get the location of the scafold particle center
    S_c = get_S_center(particle_info)

    #analyze the capsid at this frame and store the order parameter tuple
    S, D = get_A5_contacts(particle_info, box_dim, size_ratio, center=S_c, radius=search_radius)
    print(filename, S)
    print(D)

    plt.hist(D, 10)
    plt.show()

    
if __name__ == "__main__":

	try:
		file = sys.argv[1]
	except:
		print("Please supply a gsd file to load")
		raise()

	getPentDists(file)
