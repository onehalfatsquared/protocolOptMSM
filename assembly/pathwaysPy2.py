'''
This file contains code for tracking the assembly progress of a capsid of conical subunits
around a nanoparticle scaffold. Progress is tracked as the tuple (n_p, n_h, b_p, b_h, c_h), 
where n_p is the number of attached pentamers, n_h is number of hexamers, b_p is the
number of h-p bonds, b_h is the number of h-h bonds, and c_h is the number of hexamers
in contact with exactly two pentamers.

For analysis, only (n_p, n_h, c_h) is used to construct discrete states, which cuts the 
number of states from around 80,000 to around 3000. 

A perfect capsid is denoted by the tuple (12,30,60,60,30). The (12,30,30) reduction
still uniquely identifies the desired state, but lumps other states. For example
(12,30,60,60,26) is a complete capsid with mixed T3/T4 structure, but (12,30,26) includes
this structure as well as transients on their way to forming a slighlt larger cluster.

Code is adapted from assembly_pathways_FM.py, written by Farri Mohajerani and ... .
Written November 22nd, 2021. 
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
############# Spherical Harmonic Computation #####################
##################################################################

def icosaQ6test():
    #test the boop functions on a perfect 12 unit icosahedron

    x = np.array([[0, 0.5000, 0.8090],
                 [0,-0.5000, 0.8090],
                 [0,-0.5000,-0.8090],
                 [0, 0.5000,-0.8090],
                 [0.8090, 0, 0.5000],
                 [0.8090, 0,-0.5000],
                 [-0.8090,0,-0.5000],
                 [-0.8090,0, 0.5000],
                 [0.5000,  0.8090, 0],
                 [-0.5000, 0.8090, 0],
                 [-0.5000,-0.8090, 0],
                 [0.5000, -0.8090, 0]])

    Q = evalBOOPsCenter(x, [0,0,0])
    print(Q)
    sys.exit()



def getPolarAngles(p0, p1):
    #get the polar angles between two points

    #get the difference in coordinates
    x = p1[0] - p0[0]
    y = p1[1] - p0[1]
    z = p1[2] - p0[2]

    #convert (x,y,z) to (r,theta,phi) (theta is azimuthal)
    theta = np.arctan(y / x)
    phi   = np.arctan(np.sqrt(x*x+y*y) / z)

    return theta, phi


def computeYlm(l, m, theta, phi):
    #compute the l,m spherical harmonic (theta is azimuthal in scipy)

    return scipy.special.sph_harm(m, l, theta, phi)


def evalBOOPs(bondPositions):
    #eval q4 and q6 and q10 for the bonded particles in the input

    Y4m = np.zeros(9, dtype='complex')
    Y6m = np.zeros(13, dtype='complex')
    numBonds = len(bondPositions)
    print(numBonds)
    if numBonds == 0:
        return 0

    for m in range(-4,5):
        SHaverage = 0
        for pair in bondPositions:
            theta, phi = getPolarAngles(pair[0],pair[1])
            SHaverage += computeYlm(4, m, theta, phi)
        Y4m[m] = SHaverage / float(numBonds)

    Q4 = 0
    for i in range(-4,5):
        Q4 += np.real(Y4m[i] * np.conjugate(Y4m[i]))
    Q4 *= ((4*np.pi) / (2.0 * 4.0 + 1.0))
    Q4 = np.sqrt(Q4)

    for m in range(-6,7):
        SHaverage = 0
        for pair in bondPositions:
            theta, phi = getPolarAngles(pair[0],pair[1])
            SHaverage += computeYlm(6, m, theta, phi)
        Y6m[m] = SHaverage / float(numBonds)

    Q6 = 0
    for i in range(-6,7):
        Q6 += np.real(Y6m[i] * np.conjugate(Y6m[i]))
    Q6 *= ((4*np.pi) / (2.0 * 6.0 + 1.0))
    Q6 = np.sqrt(Q6)

    return Q4, Q6

def evalBOOPsMid(bondPositions, center):
    #eval q4 and q6 and q10 for the bonded particles in the input

    Y4m = np.zeros(9, dtype='complex')
    Y6m = np.zeros(13, dtype='complex')
    numBonds = len(bondPositions)
    print(numBonds)
    if numBonds == 0:
        return 0

    for m in range(-4,5):
        SHaverage = 0
        for pair in bondPositions:
            mid = (pair[0] + pair[1]) / 2.0
            theta, phi = getPolarAngles(center, mid)
            SHaverage += computeYlm(4, m, theta, phi)
        Y4m[m] = SHaverage / float(numBonds)

    Q4 = 0
    for i in range(-4,5):
        Q4 += np.real(Y4m[i] * np.conjugate(Y4m[i]))
    Q4 *= ((4*np.pi) / (2.0 * 4.0 + 1.0))
    Q4 = np.sqrt(Q4)

    for m in range(-6,7):
        SHaverage = 0
        for pair in bondPositions:
            mid = (pair[0] + pair[1]) / 2.0
            theta, phi = getPolarAngles(center, mid)
            SHaverage += computeYlm(6, m, theta, phi)
        Y6m[m] = SHaverage / float(numBonds)

    Q6 = 0
    for i in range(-6,7):
        Q6 += np.real(Y6m[i] * np.conjugate(Y6m[i]))
    Q6 *= ((4*np.pi) / (2.0 * 6.0 + 1.0))
    Q6 = np.sqrt(Q6)

    return Q4, Q6

def evalBOOPsCenter(allPos, center):
    #eval q4 and q6 and q10 for the bonded particles in the input

    Y4m = np.zeros(9, dtype='complex')
    Y6m = np.zeros(13, dtype='complex')
    numBonds = len(allPos)
    print(numBonds)
    if numBonds == 0:
        return 0

    for m in range(-4,5):
        SHaverage = 0
        for pos in allPos:
            theta, phi = getPolarAngles(center, pos)
            SHaverage += computeYlm(4, m, theta, phi)
        Y4m[m] = SHaverage / float(numBonds)

    Q4 = 0
    for i in range(-4,5):
        Q4 += np.real(Y4m[i] * np.conjugate(Y4m[i]))
    Q4 *= ((4*np.pi) / (2.0 * 4.0 + 1.0))
    Q4 = np.sqrt(Q4)

    for m in range(-6,7):
        SHaverage = 0
        for pos in allPos:
            theta, phi = getPolarAngles(center, pos)
            SHaverage += computeYlm(6, m, theta, phi)
        Y6m[m] = SHaverage / float(numBonds)

    Q6 = 0
    for i in range(-6,7):
        Q6 += np.real(Y6m[i] * np.conjugate(Y6m[i]))
    Q6 *= ((4*np.pi) / (2.0 * 6.0 + 1.0))
    Q6 = np.sqrt(Q6)

    Y8m = np.zeros(17, dtype='complex')
    for m in range(-8,9):
        SHaverage = 0
        for pos in allPos:
            theta, phi = getPolarAngles(center, pos)
            SHaverage += computeYlm(8, m, theta, phi)
        Y8m[m] = SHaverage / float(numBonds)

    Q8 = 0
    for i in range(-8,9):
        Q8 += np.real(Y8m[i] * np.conjugate(Y8m[i]))
    Q8 *= ((4*np.pi) / (2.0 * 8.0 + 1.0))
    Q8 = np.sqrt(Q8)

    Y10m = np.zeros(21, dtype='complex')
    for m in range(-10,11):
        SHaverage = 0
        for pos in allPos:
            theta, phi = getPolarAngles(center, pos)
            SHaverage += computeYlm(10, m, theta, phi)
        Y10m[m] = SHaverage / float(numBonds)

    Q10 = 0
    for i in range(-10,11):
        Q10 += np.real(Y10m[i] * np.conjugate(Y10m[i]))
    Q10 *= ((4*np.pi) / (2.0 * 10.0 + 1.0))
    Q10 = np.sqrt(Q10)

    return Q4, Q6, Q8, Q10

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

def get_energy(particle_info, box_dim, size_ratio, center=None, radius=None):
    #get potential energy of a configuration

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
    Ehh = 0
    Ehp = 0
    for i in range(len(distancesHH)):
        if (distancesHH[i] < HH_length):
            r0 = 7.21
            alpha = 14 / r0
            d = distancesHH[i]
            Ehh += 1.5*(np.exp(-2*alpha*(d-r0)) - 2*np.exp(-alpha*(d-r0)))



    HPbonds = []
    for i in range(len(distancesHP)):
        if (distancesHP[i] < HP_length):
            r0 = 7.21/2.0 + 0.77*7.21/2.0
            alpha = 14 / r0
            d = distancesHP[i]
            Ehp += 1.3*(np.exp(-2*alpha*(d-r0)) - 2*np.exp(-alpha*(d-r0)))


    # print(Ehh)
    # print(Ehp)
    # print(Ehh+Ehp)
    return Ehh+Ehp

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
        

def getPathway(filename, verbose = False):
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

    #init storage for state (n_s, n_b) as fn of frame
    out = np.zeros(frames, dtype="i,i,i,i,i")
    #out = np.zeros(frames, dtype="i,i,i")

    #loop over each frame
    for frame in range(0,frames):

        #get the particle info for the current frame
        particle_info = get_particles(snaps, frame)

        #get the location of the scafold particle center
        S_c = get_S_center(particle_info)

        #analyze the capsid at this frame and store the order parameter tuple
        q = get_A5_contacts(particle_info, box_dim, size_ratio, center=S_c, radius=search_radius)
        out[frame] = q
        if (verbose):
            print("State = {}".format(out[frame]))

    return out

def convert2npy(filename, path):
    #convert the path in (n_s, n_b) space to npy file

    #strip the gsd off the filename
    filename = filename.split('.gsd')[0]

    #save to npy file
    np.save(filename, path)

    return 


def checkIcosa():
    #check how many structures have icosahedral symmetry

    samples = 0
    icosa = 0
    size_ratio = 0.77

    for i in range(200):
        filename = "../trajectories/T1.0/traj" + str(i) + ".gsd"

        #get the collection of snapshots and get number of frames
        try:
            snaps = gsd.hoomd.open(name=filename, mode="rb")
        except:
            continue
        frames = len(snaps)
        samples += 1

        #get box configuration
        box = snaps[0].configuration.box
        box_dim = np.array([box[0], box[1], box[2]])

        #init storage for state (n_s, n_b) as fn of frame
        out = np.zeros(frames, dtype="i,i,i,i")

        #loop over each frame
        for frame in range(frames-15,frames):

            #get the particle info for the current frame
            particle_info = get_particles(snaps, frame)

            #get the location of the scafold particle center
            S_c = get_S_center(particle_info)

            #analyze the capsids
            q = get_A5_contacts(particle_info, box_dim, size_ratio, center=S_c, radius=search_radius)

            if (q[0] == 12 and q[1] == 30 and q[-1] == 30):
                icosa += 1
                break

    frac = float(icosa) / float(samples)
    print("{} % of clusters have icosa symmetry. {} trajectories.".format(frac, samples))

    return

if __name__ == "__main__":

    #get the name of the input file
    try:
        filename = sys.argv[1]
    except:
        print("Usage: %s <gsd_file>" % sys.argv[0])
        raise

    #get the pathway from the input gsd file
    # path = getPathway(filename, verbose=True)

    #convert to npy file
    # convert2npy(filename, path)





    #get the collection of snapshots and get number of frames
    snaps = gsd.hoomd.open(name=filename, mode="rb")
    #get box configuration
    box = snaps[0].configuration.box
    box_dim = np.array([box[0], box[1], box[2]])
    energies = []
    in_target = False
    for i in range(len(snaps)-2000, len(snaps)):
        particle_info = get_particles(snaps, i)
        #get the location of the scafold particle center
        S_c = get_S_center(particle_info)

        if not in_target:
            q = get_A5_contacts(particle_info, box_dim, 0.77, center=S_c, radius=search_radius)
            if q[0] == 12 and q[1] == 30 and q[-1] == 30:

                in_target = True
        else:
            E = get_energy(particle_info, box_dim, 0.77, center=S_c, radius=search_radius)
            print(i,E)
            energies.append(E)

    print("Average energy in the final state is {}".format(np.mean(energies)))

    #icosaQ6test()
    #checkIcosa()
