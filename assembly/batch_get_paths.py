'''
This script will construct pathways for each trajectory in the /trajectories/ folder. 

Loops over each sub-directory (different ensembles), counts the number of .gsd files, 
and calls the getPathway function on each to generate a pathway. These are then output 
to .npy files in the same directory, to be opened by the MSM construction code. 

Also writes a text file containing the number of frames (which should be constant for a 
given ensemble) and the number of trajectories in the folder (to be used by the MSM code).
'''

import numpy as np
import fnmatch
import os
import sys
import inspect
import glob
import pickle

import string
import random

from pathwaysPy2 import getPathway

#set a path to the MSM folder to do relative importing
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir) 
msm_dir = parent_dir + "/MSM/"
sys.path.insert(0, msm_dir) 
from manualMSM import loadStateDict


def analyze_folder_gsd(folder):
    #loop over all trajectories in the specfied folder and analyze each pathway
    #save an npy file with the trajectory in (n_s, n_b) space

    #count the number of .gsd files in this directory
    trajs = fnmatch.filter(os.listdir(folder), '*.gsd')
    print(trajs)
    
    #loop over the gsd files
    for traj in trajs:

        #set the gsd_file location and output npy file
        gsd_file = folder + traj
        npy_file = folder + traj.split(".")[0]
        #print progress update
        print("Extracting pathway from {}".format(gsd_file))

        #call getPathways on the file
        try:
            path = getPathway(gsd_file)

            #save the result as an .npy file with the same index
            np.save(npy_file, path)
        except:
            #something went wrong, save a flag file to delete
            print("Could not extract file {}".format(gsd_file))
            np.save(npy_file + 'bad', ' ')
            continue

    return



def gsd2npy():
    #loop over all directories, construct npy tuples with pathways

    #set the relative path to the trajectories folder
    traj_dir = "../trajectories/"

    #loop over each directory in the /trajectories/ folder
    dir_list = glob.glob(traj_dir + '*')
    for folder in dir_list:

        #add the / to go into the directory
        folder += "/"
        
        #analyze everything in the folder
        analyze_folder_gsd(folder)

        #try to analyze the short folder - may not exist
        try:
            short_folder = folder+ "short/"
            analyze_folder_gsd(short_folder)
        except:
            continue

        

    #exit success
    return 0


def analyze_folder_npy(folder, stateDict, refineList = []):
    #loop over all npy trajectories in the specfied folder and construct state mapping

    #count the number of .npy files in this directory
    trajs = np.sort(fnmatch.filter(os.listdir(folder), '*.npy'))

    #check if performing refinement of state space discr.
    refine = False
    if len(refineList)>0:
        refine = True
    
    #loop over the gsd files
    for traj in trajs:

        #set the npy filename
        npy_file = folder + traj.split(".")[0] + ".npy"

        #print progress update
        print("Extracting pathway from {}".format(npy_file))

        #load the npy file
        try:
            path = np.load(npy_file)
        except:
            continue

        #loop over the npy file
        for state in path:

            #get the reduced representation
            reducedState = (state[0], state[1], state[-1])

            #check if performing refinement
            if refine:

                #check if this state is to be refined. if so use 5 coordinate rep
                if reducedState in refineList:
                    reducedState = state

                
            #if the tuple is not in the dictionary, add it
            if tuple(reducedState) not in stateDict:
                state_index = len(stateDict)
                stateDict[tuple(reducedState)] = state_index

                #print an update message
                print("State {} added to the dict as index {}".format(reducedState,state_index))
        

    return


def npy2dict(use_folder = "", refine = False):
    #loop over all directories, make dict

    #init a dictionary to hold the tuple to state mapping
    stateDict = {}

    #if refining, load the list
    if refine:
        set_name = "../MSM/data/refineStates"
        try:
            with open(set_name, 'rb') as f:
                refineList = pickle.load(f)
            print("Set of refinement candidates loaded from {}".format(set_name))
        except:
            print("A refinement file {} was not found.".format(set_name),\
            "Please create this before performing refinement")
            raise()
    else:
        refineList = []

    #set the relative path to the trajectories folder
    traj_dir = "../trajectories/"

    #get all subdirectories of trajectories
    dir_list = glob.glob(traj_dir + 'P*')


    #if a folder is specified, just use that one
    if not (use_folder == ""):
        dir_list = [use_folder]

    #loop over folder extracting trajectories
    for folder in dir_list:

        #add the / to go into the directory
        folder += "/"

        #analyze everything in the folder
        try:
            analyze_folder_npy(folder, stateDict, refineList)
        except:
            continue

        #try to analyze the short folder (may not exist yet)
        try:
            short_folder = folder+ "short/"
            analyze_folder_npy(short_folder, stateDict, refineList)
        except:
            continue

        #try to analyze disassembly traj folders, stateX
        try:
            dis_folders = glob.glob(folder+"state*")
            for dis_folder in dis_folders:
                analyze_folder_npy(dis_folder+"/", stateDict)
        except:
            continue


    #pickle the dictionary to the msm/data folder
    msm_folder = "../MSM/data/"

    #define alphabet and create temp random string name
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(16))
    random_temp_name = msm_folder + random_string

    #write the temp file
    with open(random_temp_name, 'wb') as f:
        pickle.dump(stateDict, f)
    print("Temporary dictionary {} created".format(random_temp_name))

    #rename to stateDict
    new_name = msm_folder + "stateDict"
    if refine:
        new_name += "Refined"

    os.rename(random_temp_name, new_name)
    print("Temp dictionary {} renamed to {}".format(random_temp_name, new_name))
        
    #exit success
    return 0


def update_dict(folder, refine = False):
    #update the pickled dict with the given folder

    #load the state dictionary
    stateDict, refineList = loadStateDict(refine=refine)

    #do the update
    try:
        analyze_folder_npy(folder, stateDict, refineList)
    except:
        print("An error occured processing the folder")
        raise()

    #pickle the dictionary to the msm/data folder
    msm_folder = "../MSM/data/"

    #define alphabet and create temp random string name
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(16))
    random_temp_name = msm_folder + random_string

    #write the temp file
    with open(random_temp_name, 'wb') as f:
        pickle.dump(stateDict, f)
    print("Temporary dictionary {} created".format(random_temp_name))

    #rename to stateDict
    new_name = msm_folder + "stateDict"
    if refine:
        new_name += "Refined"

    os.rename(random_temp_name, new_name)
    print("Temp dictionary {} renamed to {}".format(random_temp_name, new_name))
        
    #exit success
    return 0


def analyze_file():
    #analyze a test file

    file = "../trajectories/E6.5S6.0/traj4.gsd"
    getPathway(file, verbose=True)


def analyzeDict():
    #analyze states in the dictionary

    #get the stateDict dictionary mapping
    with open("../MSM/data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    keys = stateDict.keys()

    for key in keys:
        if (key[0] == 12 and key[1] == 30):
            print(key)





if __name__ == "__main__":

    #gsd2npy()

    #analyze_file()
    #analyzeDict()

    if len(sys.argv) == 1:
        print("Creating dict from scratch")
        npy2dict()

    elif len(sys.argv) == 2:
        refine = bool(int(sys.argv[1]))
        print("Creating dict from scratch with refine={}".format(refine))
        npy2dict(refine = refine)

    elif len(sys.argv) == 3:
        folder = sys.argv[2]
        refine = bool(int(sys.argv[1]))
        print("Updating dict using files in {} with refine={}".format(folder, refine))
        update_dict(folder, refine = refine)

