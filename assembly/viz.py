#import fresnel
import gsd.hoomd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import matplotlib.animation as animation

import pandas as pd
import sys
import os
import inspect

import pickle

#set a path to the assembly folder to do relative importing
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir) 
assembly_dir = parent_dir + "/simulation/"
sys.path.insert(0, assembly_dir) 

from ratchet import Database
from ratchet import State



def makeMovie():

    #get the final snapshot
    filename = "../trajectories/P1.33H1.6/traj4.gsd"
    seed = filename.split('traj')[-1].split('.')[0]
    frameex = -1
    with gsd.hoomd.open(name=filename, mode="rb") as gsd_file:
        snap = gsd_file[frameex]

    #get the box data
    box = snap.configuration.box
    L = [box[0], box[1], box[2]]
    shift = [x/2 for x in L]

    #get snaps and number of frames
    snaps = gsd.hoomd.open(name=filename, mode="rb")
    frames = len(snaps)

    def animate(f):
        print("Generating frame {}".format(f))
        particle_info = get_particles(snaps, f)

        #get the location of the scafold particle center
        S_c = get_S_center(particle_info)[0]

        #restrict to box around the scaffold
        extend = 22
        M = S_c + extend
        m = S_c - extend

        #analyze the capsids
        box_dim = np.array(L)
        ploc, hloc, pbonds, hbonds = get_A5_contacts(particle_info, box_dim, center=S_c, radius=16)

        #plot all the capsomer particles
        fig = plt.figure()
        ax = a3.Axes3D(fig)
        pentamers = 0
        hexamers  = 0
        rHex = 7.7/2.0
        rPent= rHex * 0.77
        for i in range(len(ploc)):
            xc, yc, zc = ploc[i][0], ploc[i][1], ploc[i][2]
            xc, yc, zc = fixCoordinates(xc, yc, zc, S_c, L) 
            (x,y,z) = drawSphere(xc, yc, zc, rPent)
            c = 'purple'
            if pbonds[i] == 5:
                c = 'green'
                pentamers += 1
            elif pbonds[i] == 6:
                c = 'red'
                hexamers += 1
            ax.plot_wireframe(x,y,z,rPent,color=c)
        for i in range(len(hloc)):
            xc, yc, zc = hloc[i][0], hloc[i][1], hloc[i][2]
            xc, yc, zc = fixCoordinates(xc, yc, zc, S_c, L) 
            (x,y,z) = drawSphere(xc, yc, zc, rHex)
            c = 'yellow'
            if hbonds[i] == 5:
                c = 'orange'
                pentamers += 1
            elif hbonds[i] == 6:
                c = 'blue'
                hexamers += 1
            ax.plot_wireframe(x,y,z,rHex,color=c)

        #plot the scaffold sphere
        r_S = 9.0
        (xs,ys,zs) = drawSphere(S_c[0], S_c[1], S_c[2], r_S)
        ax.plot_wireframe(xs,ys,zs,r_S,color='grey')
 
        #plot result
        ax.set_xlim(m[0], M[0])
        ax.set_ylim(m[1], M[1])
        ax.set_zlim(m[2], M[2])
        set_axes_equal(ax)

        #save result 
        if not os.path.exists('movie{}'.format(seed)):
            os.makedirs('movie{}'.format(seed))
        plt.savefig("movie{}/frame{}.png".format(seed, f))
        plt.close()

    #get the frames
    for i in range(frames):
        if not os.path.exists('movie{}/frame{}.png'.format(seed,i)):
            animate(i)
        
    #load frames into array
    images = []
    for i in range(frames):
        img = Image.open("movie{}/frame{}.png".format(seed, i))
        images.append(img.copy())
        img.close()
    images[0].save('movie{}.gif'.format(seed), save_all=True, append_images=images[1:], duration=40, loop=0)
    
    return

def distance(x0, x1, dimensions):
    #get the distance between the points x0 and x1
    #assumes periodic BC with box dimensions given in dimensions

    #get distance between particles in each dimension
    delta = np.abs(x0 - x1)

    #if distance is further than half the box, use the closer image
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)

    #compute and return the distance between the correct set of images
    return np.sqrt((delta ** 2).sum(axis=-1))

def get_particle_info(data):
    #return particle info dataframe from snap data

    #positions need to be seperated in each coordinate
    particle_info = {
        'type': [data.particles.types[typeid] 
                 for typeid in data.particles.typeid],
        'body': data.particles.body,
        'position_x': data.particles.position[:, 0],
        'position_y': data.particles.position[:, 1],
        'position_z': data.particles.position[:, 2],
    }

    return pd.DataFrame(particle_info)

def get_particles(traj, frame):
    #return the needed info to track assembly from a trajectory frame

    #get the data from the current frame
    data = traj.read_frame(frame)

    #return a dataframe with the relevant info for each particle
    return get_particle_info(data)


def get_S_center(particle_info):
    #get center of the scaffold particle from particle_info

    #get the array location of the S type particle
    S_particles = particle_info.loc[particle_info['type'] == 'S']

    #grab the coordinates from this array location
    S_coords = np.array([S_particles['position_x'].values, 
                         S_particles['position_y'].values, 
                         S_particles['position_z'].values]).T

    return S_coords


def get_A5_contacts(particle_info, box_dim, center=None, radius=None):
    #determine how many A5 sites are in contact around the sphere
    #can only consider particles in a cutoff sphere for efficiency

    #set cutoff distance for bond formation
    HH_length = 10.3 #10.15 #9.25
    HP_length = HH_length * 0.77

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

        print(distance(PA5_coords, center, box_dim))

    #get the number of attached subunits
    attached_P = len(PA5_coords)
    attached_H = len(HA5_coords)
    print("Pentamers {}, Hexamers {}".format(attached_P, attached_H))

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

    #check if each distance is less than the bond length. count how many bonds per particle
    bbpH = np.zeros(attached_H, dtype='int')
    bbpP = np.zeros(attached_P, dtype='int')
    for i in range(len(distancesHH)):
        if (distancesHH[i] < HH_length):
            p1 = pairsHH[i][0]
            p2 = pairsHH[i][1]
            bbpH[p1] += 1
            bbpH[p2] += 1
            print(distancesHH[i])

    for i in range(len(distancesHP)):
        if (distancesHP[i] < HP_length):
            p1 = pairsHP[i][0]
            p2 = pairsHP[i][1]
            bbpH[p1] += 1
            bbpP[p2] += 1
            print(distancesHP[i])

    return PA5_coords, HA5_coords, bbpP, bbpH
                

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def fixCoordinates(x,y,z,C,L):
    #if some coordinates are on the other side of the box, shift them

    tol = 30
    xf, yf, zf = x, y, z

    if (np.abs(x-C[0]) > tol):
        diff = -np.sign(x - C[0])
        xf = x + diff * L[0]

    if (np.abs(y-C[1]) > tol):
        diff = -np.sign(y - C[1])
        yf = y + diff * L[1]

    if (np.abs(z-C[2]) > tol):
        diff = -np.sign(z - C[2])
        zf = z + diff * L[2]

    return xf, yf, zf

def plotParticleInfo(particle_info, box_dim):
    #given particle and box info, plot the structure

    #get the location of the scafold particle center
    S_c = get_S_center(particle_info)[0]
    print(S_c)

    #restrict to box around the scaffold
    extend = 40
    M = S_c + extend
    m = S_c - extend

    #analyze the capsids
    ploc, hloc, pbonds, hbonds = get_A5_contacts(particle_info, box_dim, center=S_c, radius=15)
    #ploc, hloc, pbonds, hbonds = get_A5_contacts(particle_info, box_dim, center=S_c, radius=80)

    #plot all the capsomer particles
    ax = a3.Axes3D(plt.figure())
    pentamers = 0
    hexamers  = 0
    rHex = 7.7/2.0
    rPent= rHex * 0.77
    for i in range(len(ploc)):
        xc, yc, zc = ploc[i][0], ploc[i][1], ploc[i][2]
        xc, yc, zc = fixCoordinates(xc, yc, zc, S_c, box_dim) 
        (x,y,z) = drawSphere(xc, yc, zc, rPent)
        c = 'purple'
        if pbonds[i] == 5:
            c = 'green'
            pentamers += 1
        elif pbonds[i] == 6:
            c = 'red'
            hexamers += 1
        print(c)
        ax.plot_wireframe(x,y,z,rPent,color=c)
    for i in range(len(hloc)):
        xc, yc, zc = hloc[i][0], hloc[i][1], hloc[i][2]
        xc, yc, zc = fixCoordinates(xc, yc, zc, S_c, box_dim) 
        (x,y,z) = drawSphere(xc, yc, zc, rHex)
        c = 'yellow'
        if hbonds[i] == 5:
            c = 'orange'
            pentamers += 1
        elif hbonds[i] == 6:
            c = 'blue'
            hexamers += 1
        print(c)
        ax.plot_wireframe(x,y,z,rHex,color=c)

    #plot the scaffold sphere
    r_S = 8.3
    (xs,ys,zs) = drawSphere(S_c[0], S_c[1], S_c[2], r_S)
    ax.plot_wireframe(xs,ys,zs,r_S,color='grey')

    #print num hex and pent
    print("Total {}, Pentamers: {}, Hexamers: {}".format(len(ploc)+len(hloc), pentamers, hexamers))
    #plot result
    ax.set_xlim(m[0], M[0])
    ax.set_ylim(m[1], M[1])
    ax.set_zlim(m[2], M[2])
    set_axes_equal(ax)
    plt.show()


def plotFromTraj(filename, frame):
    #use matplotlib to plot the structure

    #get all frames from the traj
    snaps = gsd.hoomd.open(name=filename, mode="rb")

    #determine the frame to look at (with support for negative indexing)
    true_frame = frame
    if (frame < 0):
        true_frame = len(snaps)+frame

    #get the particle info for that frame
    particle_info = get_particles(snaps, true_frame)

    #get the box info
    box = snaps[0].configuration.box
    L = [box[0], box[1], box[2]]
    box_dim = np.array(L)

    #plot the structure
    plotParticleInfo(particle_info, box_dim)

    


def plotFromSnap(db_file, state):
    #plot a structure given a database file and the state description

    #load the database 
    with open(db_file, 'rb') as f:
        db = pickle.load(f)

    # for i in range(db.num_states):
    #     print(db.states[i].pair)

    #search the database for the state
    ic_state = db.search_by_pair(state)
    print("Found state {} in Database".format(state))
    snap = db.states[ic_state].get_random_snap()

    #get the particle info
    particle_info = get_particle_info(snap)

    #get the box info
    print(dir(snap.configuration))
    # box = snap.configuration.box
    # L = [box[0], box[1], box[2]]
    L = [120,120,120]
    box_dim = np.array(L)

    #plot the structure
    plotParticleInfo(particle_info, box_dim)

    





if __name__ == "__main__":
    

    # traj_file = "../trajectories/P1.6H1.2/old50/traj13.gsd"
    # traj_file = "../trajectories/s0.9/traj47.gsd"
    # traj_file = "../trajectories/P1.2H1.4/short/traj1077.gsd"
    traj_file = "../trajectories/opt0/traj1.gsd"
    #plotFromTraj(traj_file, 0)
    plotFromTraj(traj_file, -1)

    db_file = "../trajectories/P1.5H1.4/stateDB"
    state = [12,25,10]
    #plotFromSnap(db_file, state)




    #makeMovie()




