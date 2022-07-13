import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import inspect

import pickle

#set a path to the assembly folder to do relative importing
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir) 
assemblydir = parentdir + "/assembly/"
sys.path.insert(0, assemblydir) 

#import the necessary functions
from pathwaysPy2 import getPathway
from pathwaysPy2 import convert2npy
from simulate import simulate as sim
from simulate import simulateFromSnap as simFromSnap
from simulateSize import simulate as simSize
from simulateTD import simulate as simTD


def set_run_types():
    #create a dictionary mapping run types to integers

    run_types = dict()
    run_types["input_hpcc"] = 0
    run_types["input_size"] = 1
    run_types["input_opt"]  = 2
    run_types["input_long"] = 3

    return run_types


def get_run_type(run_types):
    #determine the run type from command line argument

    #read the input filename and determine what folder it is in
    input_file = sys.argv[1]
    loc        = input_file.split("/")[0]

    #extract a run type from the folder, if possible
    try:

        run_type   = run_types[loc]
        print("Run type is {}".format(run_type))

    except:

        print("Run type cannot be determined form the input file name")
        raise()

    #return run type
    return run_type

def move_long_files(outfile, snap_loc):
    #rename trajectories saved to long folder according to its starting state

    #determine the folder storing the outfile
    outfolder = outfile.split('long')[0] + 'long/'

    #determine what folder this trajectory should go in, and create if it doesn't exist
    new_folder = outfile.split("long")[0] + snap_loc.split('/')[-1] + '/'
    if (not os.path.exists(new_folder)):
        os.makedirs(new_folder)

    #move the files (gsd and log) from current to new folder
    gsd_ext = outfile.split('/')[-1]
    log_ext = gsd_ext.split('.')[0] + ".log"
    os.rename(outfolder+gsd_ext, new_folder+gsd_ext)
    os.rename(outfolder+log_ext, new_folder+log_ext)

    #return the new outfile location
    return new_folder + gsd_ext



def run_sim(run_type):
    #run the simulation according to run_type

    if run_type == 0:

        outfile = sim()

    elif run_type == 1:

        outfile = simSize()

    elif run_type == 2:

        outfile = simTD()

    elif run_type == 3:

        #get the seed and snapshot from command line
        seed = int(sys.argv[2])
        snap_loc = sys.argv[3]
        with open(snap_loc, 'rb') as f:
            snap = pickle.load(f)

        #run the simulation and get the output file location
        outfile = simFromSnap(snap, seed, True)
        
        #move the trajectory from the long folder to stateX for X in snap_loc
        outfile = move_long_files(outfile, snap_loc)

    #return the output file
    return outfile




if __name__ == '__main__':
    #perform a simulation with parameters from the command line and log the outfile,
    #use that outfile to analyze the trajectory and convert to npy format

    #create a dictionary of possible run types
    run_types = set_run_types()

    #get the run type from the input file location
    run_type = get_run_type(run_types)

    #run the simulation
    outfile = run_sim(run_type)

    #get the pathway from the input gsd file
    path = getPathway(outfile, verbose=True)

    #convert to npy file
    convert2npy(outfile, path)

