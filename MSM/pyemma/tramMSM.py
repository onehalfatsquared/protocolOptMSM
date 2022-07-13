import matplotlib.pyplot as plt
import numpy as np
import pyemma
import sys
import os
import fnmatch
import glob

import pickle










def exclude(pair):
    #define rules to exclude a trajectory

    if (pair[0] > 12 or pair[1] > 30):
        return True

    return False


def manualMapToState(pair2state, traj, frames):
    #map the trajectory in (n_s, n_b) space, to non-negative state index

    #init an array for the path in state space
    state_path = np.zeros(frames, dtype=int)

    #loop over tuples, get state index
    for i in range(frames):
        pair = tuple(traj[i])
        if (exclude(pair)):
            return np.zeros(0)
        state = pair2state[pair]
        state_path[i] = state

    return state_path



def extractTrajectoriesT(folder, dtrajs, ttrajs, traj_files, ensemble, pair2state):
    #extract trajectories stored in npy files, append to dtrajs

    for file in traj_files:

        file = folder + file
        #try to load the file (may fail due to blowup for given seed)
        try:
            data = np.load(file)
        except:
            continue
        frames = len(data)

        #convert to an integer state trajectory
        traj = manualMapToState(pair2state, data, frames)

        #debugging
        #print(file)
        #print(traj)

        if (len(traj)):
            #append to dtraj
            dtrajs.append(traj)

            #append to ttraj
            ttrajs.append(np.ones(frames, dtype=int) * ensemble)

    return

def analyze_folder_npy(folder, dtrajs, ttrajs, ensemble, pair2state):
    #loop over all npy files in the specified folder, append the trajectories

    #idetify the folders with long and short trajectories
    long_folder  = folder + "/"
    short_folder = long_folder + "short/"

    #get all trajectory files
    long_trajs  = fnmatch.filter(os.listdir(long_folder), '*.npy')
    short_trajs = fnmatch.filter(os.listdir(short_folder), '*.npy')

    #extract the trajectories from each folder
    extractTrajectoriesT(long_folder , dtrajs, ttrajs, long_trajs , ensemble, pair2state)
    extractTrajectoriesT(short_folder, dtrajs, ttrajs, short_trajs, ensemble, pair2state)

    return


def performDTRAM():
    #setup and perform discrete tram using all the trajectories 

    #get the pair2state dictionary mapping
    with open("pair2state", 'rb') as f:
        pair2state = pickle.load(f)

    #specify the number of states, and the index of the target state
    num_states = len(pair2state)
    target_index = pair2state[(12,30)]
    print("Target state index is {}".format(target_index))
    print("Simulations have visited {} distinct states\n".format(num_states))

    #init empty lists for trajectories
    dtrajs = []
    ttrajs = []

    #set the relative path to the trajectories folder
    traj_dir = "../trajectories/"

    #get list of each directory in the trajectories folder
    dir_list = glob.glob(traj_dir + '*')
    dir_list.sort() #low to high energy values
    K = len(dir_list)  #number of ensembles

    #init storage for parameters
    E = np.zeros(K)
    S = np.zeros(K)

    #loop over each ensemble
    for k in range(K):
        #get the folder
        folder = dir_list[k]

        #get the corresponding parameters from the foldername
        split_temp1 = folder.split('E')
        split_temp2 = split_temp1[-1].split('S')
        E[k] = float(split_temp2[0])
        S[k] = float(split_temp2[1])

        #print a progress message 
        print("Appending the trajectories from ensemble {} of {}".format(k,K-1))

        #go through the folder and extract all trajectories
        analyze_folder_npy(folder, dtrajs, ttrajs, k, pair2state)

    #invert the dictionary map for help in constructing bias array - it is 1 to 1
    inv_map = {v: k for k, v in pair2state.items()}

    #construct the bias arrays
    bias = np.zeros((K, num_states))
    unbiased = 0

    #print progress message
    #print('\nComputing bias energies\n')

    #get the unbiased energies
    for state in range(num_states):
        p = inv_map[state]
        energy = -p[0] * S[unbiased] - p[1] * E[unbiased] * 2
        bias[unbiased][state] = energy

    #get energies from every other row and subtract the first row
    for k in range(K):
        if (not k == unbiased):
            for state in range(num_states):

                p = inv_map[state]
                energy = -p[0] * S[k] - p[1] * E[k] * 2
                bias[k][state] = (energy - bias[unbiased][state])
                #bias[k][state] = 0

    #set the unbiased row to 0
    for state in range(num_states):
        bias[unbiased][state] = 0

    #run tram
    print("Running dTRAM")
    lag_time = 2
    dtram_obj = pyemma.thermo.dtram(ttrajs, dtrajs, bias, lag = lag_time, maxerr = 1e-4,
                                   maxiter=30000, connectivity='summed_count_matrix',
                                   unbiased_state=unbiased)
    #print(tram_obj.log_likelihood())
    #print(tram_obj.count_matrices)
    #print(tram_obj.meval('stationary_distribution'))
    #print(tram_obj.meval('f'))
    #print(tram_obj.models)

    #print out the transition matrix at each ensemble
    for k in range(K):
        msm = dtram_obj.models[k]
        a_set = msm.active_set
        a_index = np.where(a_set == target_index)
        print(a_set)
        print(msm.P[a_index][0][a_index])

    #save the tram object for use in other codes
    #dtram_obj.save('dtram_out.pyemma', model_name='dodec', overwrite=True)
    with open("dtram_obj", 'wb') as f:
        pickle.dump(dtram_obj, f)


    return













if __name__ == "__main__":

    #simpleTest()
    #performDTRAM()
    count_tests()