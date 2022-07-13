'''
This script will provide implementations of the adaptive sampling of MSMs via eigenvalue 
uncertainty analysis algorithm, described in the 2006 Pande paper, "Calculation of the 
distribution of eigenvalues and eigenvectors in Markovian state models for molecular 
dynamics". 

The algorithm takes in an MSM and computes the sensitivity of some "important" eigenvalues
to the entries of the transition matrix. We determine the biggest contributors to the 
uncertainty in these eigenvalues, and construct a distribution to sample for initial 
configurations for a ratcheting algorithm. 

We also provide simpler candidates to start ratcheting, such as all states that are observed 
on a reactive trajectory to a specified target state. 
'''

import numpy as np
import sys
import os
import fnmatch

import pickle

import scipy
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

import gsd.hoomd

#import the MSM class object and needed functions
from manualMSM import MSM
from manualMSM import loadStateDict
from manualMSM import extractTrajectories
from manualMSM import createMSM
from manualMSM import loadMSMs

#################################################################################
#################### Method 5 - Important Probability Based #####################
#################################################################################

'''
For a selection of MSMs, determine which states go beyond a threshold probability 
at any point during the assembly. Add these to a list that will be used as a 
checklist for sampling in the disassembly regime. 
'''

def locateInFolder(folder, stateLocation, toFind, stateDict, refineList):
    #for each trajectory in the folder, locate states present in toFind

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
        for i in range(len(path)):

            state = path[i]

            #get the reduced representation
            reducedState = (state[0], state[1], state[-1])

            #check if performing refinement
            if refine:

                #check if this state is to be refined. if so use 5 coordinate rep
                if reducedState in refineList:
                    reducedState = state

                
            #get the index of this state
            index = stateDict[tuple(reducedState)]

            #check if this state is in toFInd
            if index in toFind:

                #write the filename and frame number to the dict
                stateLocation[index] = [npy_file, i]
                print("State {} found in {}, frame={}".format(index,npy_file,i))

                #remove this index from toFind
                toFind.remove(index)
                print("{} states left to find".format(len(toFind)))

        #if toFind is empty, can return
        if len(toFind) == 0:
            return

    return

def mapStateToFile(states, traj_folders, stateDict, refineList):
    #take a list of states and map them to frames and files from npy trajs

    #create an empty dict of states to store location info
    stateLocation = dict()
    toFind = []
    for state in states:
        stateLocation[state[0]] = []
        toFind.append(state[0])

    #loop over the folders containing the npy files, locating states
    for folder in traj_folders:

        #check each folder for state locations for snapshots
        locateInFolder(folder, stateLocation, toFind, stateDict, refineList)

        #check if we can stop looking
        if len(toFind) == 0:
            print("Found all requested states. Exiting loop.")
            break

    #return locations of these states
    return stateLocation

def getImportantStates(prob_cut_max=0.5e-3, prob_cut_avg=1e-5, data_version='', refine=True):
    #determine states exceedinging a probability threshold during the assembly process

    #set universal parameters
    animate_time = 25

    #set which parameter values to consider - near top left to center of domain
    restrict = [6,11,13,19]

    #load the corresponding MSMs
    MSMs, K, E2 = loadMSMs(data_version=data_version, refine=refine, restrict=restrict)

    #load the dictionary of states
    stateDict, refineList = loadStateDict(data_version=data_version, refine=refine)
    num_states = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set the initial and target state indices using the stateDict
    initial_index = stateDict[(0,0,0)]

    #init storage for important states
    important = []
    traj_folders = []

    #loop over MSMs, tracking which states exceed threshold probs
    for msm in MSMs:

        #compute the probability as fn of time
        lag = msm.lag
        num_steps = int(900000.0 / (animate_time*lag))
        t, p = msm.solveForwardEquationActive(initial_index, num_steps)

        #get all row indices with a column exceeding threshold - max and average
        important_indicesMAX = np.where(sum(p>prob_cut_max,0)>0)[0]
        important_indicesAVG = np.where(sum(p,0)> (prob_cut_avg * num_steps))[0]

        X = sum(p,0)
        
        #for each index, convert to a true id and get the state description
        for idx in important_indicesMAX:
            true_id    = np.where(msm.fullToNew == idx)[0][0]
            state_full = inv_map[true_id]
            important.append([true_id, state_full])
            print(state_full, X[idx]/num_steps)

        for idx in important_indicesAVG:
            true_id    = np.where(msm.fullToNew == idx)[0][0]
            state_full = inv_map[true_id]
            important.append([true_id, state_full])
            print(state_full, X[idx]/num_steps)

        #append a trajectory folder
        p = msm.Ep
        h = msm.Eh
        traj_folders.append("../trajectories/P{}H{}/".format(p,h))
        
    #remove duplicates from the list
    important = [i for n, i in enumerate(important) if i not in important[:n]]
    
    #get in file locations for all the important states
    state_location = mapStateToFile(important, traj_folders, stateDict, refineList)

    #dump the state location data
    with open("ratchetStateLocations", 'wb') as f:
            pickle.dump(state_location, f)
    
    return 

def getFileList():
    #get the set of files needed to extract all snapshots

    #open the location dictionary
    with open("ratchetStateLocations", 'rb') as f:
        state_location = pickle.load(f)

    #get a set of all the needed files
    files = np.sort(list(set([x[0] for x in list(state_location.values())])))

    #make a dict to separate the different directories
    files_per_folder = dict()
    for file in files:
        s = file.split('/')
        folder = s[2]
        traj_file = s[3].split('.')[0]+".gsd"

        #check if in the dict and add
        if folder not in files_per_folder.keys():
            files_per_folder[folder] = [traj_file]
        else:
            files_per_folder[folder].append(traj_file)

    #make a separate text file with states for each directory
    for folder in list(files_per_folder.keys()):

        filepath = folder+"gsdList.txt"
        with open(filepath, 'w') as file_handler:
            for item in files_per_folder[folder]:
                file_handler.write("{}\n".format(item))

    return

def storeSnapshots():
    #create a dictionary that associates a state index with a snapshot
    #uses the ratchetStateLocations dictionary to find the file and frame location

    #open the location dictionary
    with open("ratchetStateLocations", 'rb') as f:
        state_location = pickle.load(f)

    #create dictionary for storing snaps
    snapDict = dict()

    #loop over keys, open the file and find the relevant frame
    all_keys = list(state_location.keys())
    for keynum in range(len(all_keys)):

        #get the key and get file loc and frame number
        key = all_keys[keynum]
        file, frame = state_location[key]
        gsd_file = file[0:len(file)-3] + "gsd"

        #print progress message
        print("Finding snapshot for key {} of {}".format(keynum, len(all_keys)))
        print("File: {}, Frame {}".format(gsd_file, frame))

        #load the gsd file
        snaps = gsd.hoomd.open(name=gsd_file, mode="rb")

        #extract the snap at frame and add to dict
        snap = snaps.read_frame(frame)
        snapDict[key] = snap

    #write the dictionary to disk
    with open("snapDict", 'wb') as f:
        pickle.dump(snapDict, f)

    return

            







#################################################################################
#################### Method 4 - Critical Nucleus Based ##########################
#################################################################################

'''
Compute the committor function for the given MSM and determine all states with 
forward committor probability greater than around 0.5. Most of these initial states
will result in successful assembly, and some will sample disassembly. 
'''

def computeCommittor(P, initial, target, num_states):
    #compute the committor probability for each state

    #init the system matrix and rhs
    R = np.array(P.todense())
    b = np.zeros(num_states)

    #subtract identity from P to get R
    for i in range(num_states):
        R[i][i] -= 1.0

    #set boundary conditions
    #initial state
    for i in range(num_states):
        if (i == initial):
            R[initial,i] = 1.0
        else:
            R[initial,i] = 0.0

    #target state
    for i in range(num_states):
        if (i == target):
            R[target,i] =  1.0
        else:
            R[target,i] = 0.0

    #set vector
    b[target] = 1.0

    #do the solve and return the result
    x = np.linalg.solve(R,b)
    return x


def ratchetByCriticalNucleus(folder):
    #determine good candidate states for ratchet trajectories by computing 
    #a critical nucleus and only considering states past this reaction coordinate

    #get the stateDict dictionary mapping
    stateDict = loadStateDict()

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial state and target state
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,30,30)]

    #set a lag and construct an MSM
    lag = 50
    animate_time = 25
    MSM = manual_create_MSM(folder, lag)

    #get the target state index in the active system
    target_active = MSM.fullToNew[target_index]
    print("The active index of the target state is {}".format(target_active))
    if (target_active == -1):
        print("The requested target state is not in the active set. Exiting.")
        return -1, -1

    #print debug info
    num_states = MSM.num_states
    nsa        = MSM.num_states_active
    nz         = MSM.count_matrix.count_nonzero()
    print("{} nonzero entries out of {}, {} %".format(nz, num_states*num_states, float(nz)/float(num_states)))
    print("Absorbing Sets: {}".format(MSM.absorbing_active))

    #get the committor 
    q = computeCommittor(MSM.P_active, initial_index, target_active, nsa)

    #init a list for states to sample
    statesToSample = []

    #loop over committor entries, adding states with >0.5 committor prob
    for i in range(len(q)):
        if q[i] > 0.5:
            state = inv_map[np.where(MSM.fullToNew == i)[0][0]]
            prob  = q[i]
            print("State {}, Committor {}".format(state, prob))

            statesToSample.append(state)


    #return list of states to sample
    return statesToSample



#################################################################################
#################### Method 3 - Eigenvalue Sensitivity ##########################
#################################################################################

'''
Determine which states are responsible for the largest uncertainty in the larger 
eigenvalues. Choose a state weighted by this uncertainty.
'''


def excludeSample(state, count, value):
    #decide if the given state should be excluded from sampling 

    #dont sample if pentamers>12 or hexamers>30
    if (state[0] > 12 or state[1] > 30):
        return True

    #dont sample if distribution value doesnt exceed a tol
    sens_tol = 1e-8
    if (value < sens_tol):
        return True

    #dont sample if the state is already highly sampled
    count_cut = 10000 #gives accuracy to about 1 percent probability
    if (count > count_cut):
        return True

    #dont sample if the configuration is unhelpful
    if (state[0] > 8 and state[1] > 18 and state[2] < state[1]/2):
        return True

    return False


def getSensitivity(msm, num_states, eigenvalue):
    #get the sensitivity of the given eigenvalue to all nonzero transition probs

    #set some shorthands
    ns = num_states
    ev = eigenvalue

    #construct A = P-lambda*I
    A = msm.P_active - scipy.sparse.identity(ns) * ev
    
    #construct an LU decomp of A^T, extract L and U
    #the decomp assumes L has unit diagonal, but we need U to have unit diagonal, hence A^T
    LU = scipy.sparse.linalg.splu(A.transpose())
    U = LU.L.transpose()
    L = LU.U.transpose()

    #construct the permutation matrices
    Pr = scipy.sparse.csc_matrix((np.ones(ns), (LU.perm_r, np.arange(ns))))
    Pc = scipy.sparse.csc_matrix((np.ones(ns), (np.arange(ns), LU.perm_c)))

    #determine which diagonal element of L is 0
    d = L.diagonal()
    smallest = 100
    for i in range(len(d)):
        diag_val = abs(d[i])
        if diag_val < smallest:
            smallest  = diag_val
            small_idx = i

    #solve the linear system involing U, multiply by column perm matrix
    e_k = scipy.sparse.lil_matrix((ns, 1),dtype=float)
    e_k[small_idx, 0] = 1.0
    e_k.tocsr()
    xr = np.array([scipy.sparse.linalg.spsolve(U, e_k)]) #this is a row vector
    v = Pr.transpose() * xr.T  #this is a column vector

    #solve the linear system involving L^T
    z = scipy.sparse.csr_matrix((ns, 1),dtype=float)
    # make the diagonal and corresponding vector entry 1 so the solution is 1
    L[small_idx, small_idx] = 1.0
    z[small_idx] = 1.0
    xl = np.array([scipy.sparse.linalg.spsolve(L.transpose(), z)]) #this is row vector
    u = xl * Pc.transpose() #this is a row vector

    #compute the outer product, normalize by the inner product
    ip = u.dot(v)
    if (abs(ip) < 1e-8 or np.isnan(ip)):
        print("The sensitivity vector is empty or contains NaNs. Skipping")
        return -1
    S = u.transpose().dot(v.transpose()) / ip

    #apply a mask such that only nonzero transition matrix values have sensitivities
    nonzero_mask = msm.P_active.nonzero()
    nzm_values = S[nonzero_mask]
    SS = scipy.sparse.csr_matrix((nzm_values, nonzero_mask), shape=((ns, ns)))
    SS.eliminate_zeros()

    return SS

def importantEigs(msm, E, target_active):
    #compute first E eigenvalues and determine which are 'important' by some criteria

    #compute eigenvalues and eigenvectors
    eigs_vecs = msm.computeEigenvalues(E)
    e_vects = np.array(eigs_vecs[1])
    e_vals  = np.zeros(E, dtype=complex)

    #define list for important values
    important  = []
    imp_tol    = 0.05    #importance tolerance (target component / max value)

    #test each for importance
    for i in range(E):
        e_vals[i] = eigs_vecs[0][i]
        e_vect    = e_vects[:,i]

        #check if the component corresponding to target is significant
        component = e_vect[target_active]

        if abs(component)/np.amax(np.abs(e_vect)) > 0.05 and e_vals[i] < 1-1e-12:
            important.append(e_vals[i])
            print(i, e_vals[i], component, abs(component)/np.amax(np.abs(e_vect)))

    #return important eigenvalues
    return important

def buildSensitivityDistribution(msm, important, nsa, inv_map, K = 45):
    #get a distribution to sample states from by looping over important eigenvalues
    #and seeing which Pij entries the eigenvalue is most sensitive to
    #consider at most K eigenvalues

    #set parameters
    added_tol  = 50  #check how much uncertainty improves if adding this many observations
    top        = 6   #consider the top this many states contributing to uncertainty

    #create storage for the ratcheting sampling distribution
    ratchet_states = []
    ratchet_probs  = []

    #pick an eigenvalue, perform sensitivity calculation
    num_eigs = min(len(important),K)
    eig_num  = 0
    for ev in important[0:K]:

        #print progress message
        eig_num += 1
        print("Analyzing eigenvalue of interest number {} of {}".format(eig_num, num_eigs))

        #deal with complex eigenvalues
        if (abs(np.imag(ev)) > 1e-8):
            continue
        else:
            ev = np.real(ev)

        #get the sensitivity matrix
        SS = getSensitivity(msm, nsa, ev)
        if (type(SS) == int):
            continue

        #now we need to compute the uncertainties using these computed sensitivities
        q = np.zeros(nsa, dtype=float)
        diff = np.zeros(nsa, dtype=float)

        for i in range(nsa):

            #get the row of the transition matrix, and of the sensitivity matrix
            Prow = msm.P_active.getrow(i)
            Srow = SS.getrow(i)

            #construct the middle term of the quadratic form
            M = msm.P_active.diagonal() - Prow.transpose().dot(Prow)

            #evaluate the uncertainty
            q[i] = Srow * M * Srow.transpose()

            #print the uncertainty scaled by the row counts
            diff[i] = q[i] * (1.0 / msm.row_counts_active[i] - 1.0 / (msm.row_counts_active[i]+added_tol))
            full_state = np.where(msm.fullToNew == i)[0][0]
            state = inv_map[full_state]

            #print(i, diff[i], state, M1.row_counts_active[i])

        #find the top states
        #print('\n')
        top_ind = np.argpartition(diff, -top)[-top:]
        for ind in top_ind:
            full_state = np.where(msm.fullToNew == ind)[0][0]
            state = inv_map[full_state]
            count = msm.row_counts_active[ind]
            #print(ind, diff[ind], state, M1.row_counts_active[ind])

            #remove states that are exceptions from being considered
            exclude = excludeSample(state, count, diff[ind])
            if (exclude):
                continue
            else:
                #add state to distrubition w/ prob prop to lambda * s
                ratchet_states.append(full_state)
                ratchet_probs.append(ev * abs(diff[ind]))
                print("Appending state {} with value {}".format(state, ratchet_probs[-1]))


    #normalize the ratchet_probs distribution
    ratchet_states = np.array(ratchet_states)
    ratchet_probs  = np.array(ratchet_probs)
    ratchet_probs /= ratchet_probs.sum()

    #return the states and the distribution over them
    return ratchet_states, ratchet_probs


def ratchetByEigUncertainty(folder, lag = 100):
    #compute uncertainty in eigenvalues to entries of the transition matrix
    #choose ratchet initial states by largest uncertainty in important eigenvalues

    #get the stateDict dictionary mapping
    stateDict = loadStateDict()

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial state and target state
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,30,30)]

    #set a lag and construct an MSM
    animate_time = 50
    M1 = manual_create_MSM(folder, lag)

    #get the target state index in the active system
    target_active = M1.fullToNew[target_index]
    print("The active index of the target state is {}".format(target_active))
    if (target_active == -1):
        print("The requested target state is not in the active set. Exiting.")
        return -1, -1

    #print debug info
    num_states = M1.num_states
    nsa        = M1.num_states_active
    nz         = M1.count_matrix.count_nonzero()
    print("{} nonzero entries out of {}, {} %".format(nz, num_states*num_states, float(nz)/float(num_states)))
    print("Absorbing Sets: {}".format(M1.absorbing_active))
    
    #get the list of eigenvectors, determine the eigenvalues
    E = min(nsa-10, 400)
    #E = 15 #for faster runs for debug
    important = importantEigs(M1, E, target_active)
    
    #get the distribution from the sensitivity algo
    ratchet_states, ratchet_probs = buildSensitivityDistribution(M1, important, nsa, inv_map)
    
    #return the distribution
    return ratchet_states, ratchet_probs

#################################################################################
#################### Method 2 - Reactive States Only ############################
#################################################################################

'''
Determine which states appear on trajectories that actually form a given target state.
Returns a list of these states, which are the possible initial configurations for
new simulations, chosen either uniformly or based on frequency.
'''

def getReactiveStates(folder):
    #get a list of all reactive states, i.e. states that appear on pathways that end up
    #in the target state. 

    #get the stateDict dictionary mapping
    stateDict = loadStateDict()

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get info about the state space and target
    num_states = len(stateDict)
    target_index = stateDict[(12,30,30)]
    print("Target state index is {}".format(target_index))
    print("There are {} states".format(num_states))

    #make list for all trajectories
    dtrajs = []

    #get all npy files
    trajs = fnmatch.filter(os.listdir(folder), 'traj*.npy')
    try:
        short_trajs = fnmatch.filter(os.listdir(folder+"short/"), 'traj*.npy')
    except Exception as e:
        print(e)
        short_trajs = []

    #append trajectories from sims to dtrajs 
    print("Extracting long trajectories. Found {}".format(len(trajs)))
    extractTrajectories(dtrajs, folder,  trajs,  stateDict)
    print("Extracting short trajectories. Found {}".format(len(short_trajs)))
    extractTrajectories(dtrajs, folder+"short/",  short_trajs,  stateDict)

    #determine which end in the target state and get all the states in that trajectory
    reactiveStates = set()
    for i in range(len(dtrajs)):
        if (dtrajs[i][-1] == target_index):
            for index in dtrajs[i]:
                state = inv_map[index]
                reactiveStates.add(state)
                #print("adding ", state)

    return reactiveStates




if __name__ == "__main__":

    folder = "../trajectories/P1.4H1.45/"

    #print(getReactiveStates(folder))
    #a,b = ratchetByEigUncertainty(folder)
    # print(a)
    # print(b) 

    #ratchetByCriticalNucleus("../trajectories/P1.2H1.4/")

    #getImportantStates()
    #getFileList()
    storeSnapshots()