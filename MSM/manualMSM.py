'''
This file contains the class definition for my custom MSM class. 

Unlike pyEMMA, my version does not compute the connected set of the underlying
transition matrix before doing the estimation. It uses the full state space of 
observed transitions. There are however, parameters to cut out particular states
and transitions based on a supplied minimum count number. I have set the defaults 
to what I have found works best. 

To construct MSMs, I...
1) Compute count matrix from a list of discrete trajectories
2) Cut out states according to minimum count thresholds
3) Normalize into a probability transition matrix by computing row sums
4) Compute a reduced representation only over the active states in the system
5) Compute absorbing sets (only if requested), this takes some time.

I then provide methods for solving the forward equation, computing eigenvalues, 
and computing implied timescales. 

The rest of the file contains functions for taking in trajectory data, converting to
discrete trajectories according to the supplied state definitions, and computing MSMs
as well as analyzing them. 
'''


import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import fnmatch
import glob
import itertools
import random

import pickle

import scipy
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

from scipy.sparse import dok_matrix
from collections import defaultdict

import warnings



##################################################################
############# MSM Class Functions  ###############################
##################################################################

class MSM:

    #class methods
    def __init__(self, dtrajs, lag, Eh, Ep, num_states, min_count=1, min_frame=2, 
                 absorbingSets = False):
        #set all the input parameters, make count matrix, and probability matrix

        #init parameters that do not get set by MSM computation routines
        self.timescales       = []
        self.absorbing        = set()
        self.absorbing_active = set()

        #set the id parameters 
        self.lag = int(lag)
        self.Eh  = Eh
        self.Ep  = Ep

        #set cutoff parameters
        self.min_count = int(min_count)
        self.min_frame = int(min_frame)

        #init the sparse matrix storage
        self.num_states   = num_states
        self.count_matrix = scipy.sparse.dok_matrix((num_states, num_states), dtype=int)
        self.P            = scipy.sparse.csr_matrix((num_states, num_states), dtype=float)
        self.row_counts   = np.zeros(num_states, dtype=float)

        #init state info lists
        self.active_set   = []
        self.inactive     = []
        self.col_rem      = []

        #init the sparse matrix storage for active states
        self.num_states_active   = 0
        self.count_matrix_active = []
        self.P_active            = []
        self.row_counts_active   = []
        self.fullToNew           = np.zeros(num_states, dtype=int)

        #print a start message
        print("Estimating Markov Model")

        #compute the count matrix using dtraj data
        self.constructCountMatrix(dtrajs)
        print("Count Matrix Constructed")
        print("Total Counts = {}".format(self.row_counts.sum()))

        #apply cutoffs for frame and transition counts
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') #sparse efficiency warning, cant be avoided
            self.applyCutoffs()
        print("Cutoffs Applied:")
        print("Min Transitions = {}, Min Frames = {}".format(self.min_count, self.min_frame))

        #determine inactive state
        self.findInactive()
        print("Inactive States have been identified")

        #get the transition matrix
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') #divide by 0 warning, fine b/c those rows are purged later
            self.getProbabilityMatrix()
        print("Rows re-normalized into a probability matrix")

        #compute a reduced representation over the active set
        self.computeActive()
        print("A reduced transition matrix has been constructed over the active set")

        #determine absorbing sets if requested
        if (absorbingSets):
            print("Identifying Absorbing states by repeated squaring")
            self.findAbsorbing()
            print('Absorbing States have been identified')

    def applyCutoffs(self):
        #apply the two cutoffs to the count matrix

        #remove rows with little sampling and transitions with few counts
        for i in range(self.num_states):
            #set states below min_count to 0
            r = self.count_matrix.getrow(i)
            nonzero_mask = np.array(r[r.nonzero()] < self.min_count)
            if nonzero_mask.size > 1:
                idx = r.nonzero()[1][nonzero_mask[0]]
                self.count_matrix[i,idx] = 0

            #set rows with few frames to zero
            r = self.count_matrix.getrow(i)
            if (r.sum() < self.min_frame):
                idx = r.nonzero()[1]
                self.count_matrix[i,idx] = 0
                self.col_rem.append(i)
                

        #remove all column entries corresponding to removed rows
        for i in self.col_rem:
            c = self.count_matrix.getcol(i)
            idx = c.nonzero()[0]
            if (len(idx)) > 0:
                #print("Removing {} entries from column {}".format(len(idx),i))
                self.count_matrix[idx,i] = 0


        #change the sparsity structure (removes memory for all 0 entries)
        self.count_matrix.eliminate_zeros()

        #get the new number of counts in each row of the matrix for normalization
        self.row_counts = np.asarray(self.count_matrix.sum(axis=1)).squeeze()

    def constructCountMatrix(self, dtrajs):
        #construct the count matrix using the given trajectories and parameters

        '''
        for speed, store (i,j) pairs as keys in a dict, create matrix from there
        '''

        matrix_dict = defaultdict(int)

        #loop over each path in dtrajs
        for path in dtrajs:

            #loop over all entries in the path, get transition for i+lag
            for i in range(len(path)-self.lag):

                #get a transition from frame i to i+lag
                state1 = path[i]
                state2 = path[i+self.lag]

                #update the count dict
                matrix_dict[state1, state2] += 1

        #use the dict to set the matrix
        dict.update(self.count_matrix, matrix_dict)

        #convert to csr
        self.count_matrix = self.count_matrix.tocsr()

        #get row counts
        self.row_counts = np.asarray(self.count_matrix.sum(axis=1)).squeeze()

    def getProbabilityMatrix(self):
        #use the count matrix and counts to get a normalized probability matrix

        #create a sparse matrix with reciprocal rows sums on the diagonal
        c = scipy.sparse.diags(1/self.row_counts.ravel())

        #empty rows will get an inf. Set these to zero and log them as inactive
        find_inf = scipy.sparse.find(c > 1)[0]
        c.tocsr()
        c.data[0][find_inf] = 0
        
        #compute dot product to get PTM
        self.P = c.dot(self.count_matrix)
        diagonal = self.P.diagonal()

        #check which rows have no entries, set diag to 1 there
        for i in range(self.num_states):
            r = self.P.getrow(i)
            if (r.sum() < 1e-4):
                diagonal[i] = 1.0

        self.P.setdiag(diagonal)

    def findInactive(self):
        #determine inactive states in transition matrix

        #print info message to user
        print("Performing passes over the states to determine inactive states")

        #perform 100 attempts. should be sufficient for even large systems
        for attempt in range(100):

            #init an identified list to store which inactive states are found each attempt
            identified = []
            for i in range(self.num_states):
                if i not in self.inactive:

                    #check if the i-th row has no entries
                    cr = self.count_matrix.getrow(i)
                    if cr.sum() < self.min_frame:

                        #add to lists -> inactive, remove column, identified
                        self.inactive.append(i)
                        self.col_rem.append(i)
                        identified.append(i)

                        #get the column for the inactive state and remove transitions to it
                        c = self.count_matrix.getcol(i)
                        idx = c.nonzero()[0]
                        if (len(idx)) > 0:
                            print("Removing {} entries from column {}".format(len(idx),i))
                            self.count_matrix[idx,i] = 0

            #print out how many inactive states were identified this pass
            print("Pass {} identified {} inactive states".format(attempt, len(identified)))

            #if no new states found, we are done
            if (len(identified)) == 0:
                return

        return


    def computeActive(self):
        #compute count and transition matrices over the active set

        #compute the active set by removing inactive states
        self.active_set = list(set(range(self.num_states)) - set(self.inactive))
        self.num_states_active = len(self.active_set)
        nsa = self.num_states_active

        #compute a map from full index to active index
        for full_state in range(self.num_states):
            try:
                new_index = self.active_set.index(full_state)
            except:
                new_index = -1
            self.fullToNew[full_state] = new_index

        #init the dok matrix for the reduced count matrix
        self.count_matrix_active = scipy.sparse.dok_matrix((nsa, nsa), dtype=int)

        #build the dok matrix entry by entry
        for active_state in range(nsa):

            #convert the active state index to full state index
            fsi = self.active_set[active_state]

            #get the nonzeros in the full count matrix row
            r = self.count_matrix.getrow(fsi).nonzero()[1]

            #iterate over each, check if the state is in active set, append entry
            for full_state in r:
                new_index = self.fullToNew[full_state]
                if new_index > -1:
                    count     = self.count_matrix[fsi,full_state]
                    self.count_matrix_active[active_state,new_index] = count

        #convert to csr
        self.count_matrix_active = self.count_matrix_active.tocsr()

        #get row counts
        self.row_counts_active = np.asarray(self.count_matrix_active.sum(axis=1)).squeeze()

        #create a sparse matrix with reciprocal rows sums on the diagonal
        c = scipy.sparse.diags(1.0 / self.row_counts_active.ravel())

        #empty rows will get an inf, but there should be none. check this
        find_inf = scipy.sparse.find(c > 1)[0]
        if len(find_inf > 0):
            print(find_inf)
            print(self.count_matrix_active.getrow(find_inf[0]))
            print("Warning: An inactive state has survived pruning. Check this manually.")
            sys.exit()

        #convert to csr
        c.tocsr()
        
        #compute dot product to normalize rows and get PTM
        self.P_active = c.dot(self.count_matrix_active)

    def findAbsorbing(self):
        #search for an absorbing set by taking powers of the transition matrix

        #start by squaring the transition matrix over the active set
        M = self.P_active * self.P_active

        #keep squaring the transition matrix to reach stationary distribution
        n = 36 #number of squarings. P_n = P^(2^n)
        for i in range(n):
            #do squaring
            M = M * M

            #eliminiate entries sufficiently close to 0
            nonzero_mask = np.array(M.data[M.data.nonzero()] < 1e-5)
            M.data[nonzero_mask] = 0
            M.eliminate_zeros()
            print("Transition matrix squared {} times".format(i+1))


        #get the first row and its nonzero entries. record them
        for a in range(self.num_states_active):
            r = M.getrow(a)
            nz = r.nonzero()[1]
            for active_state in nz:
                #convert to state in full state space
                full_state = np.where(self.fullToNew == active_state)[0][0]

                #find the indices of the nonzero destinations
                active_stationary = scipy.sparse.find(M.getrow(active_state))[1]

                #convert these indices to the full state space
                full_stationary = []
                for i in range(len(active_stationary)):
                    full_stationary.append(np.where(self.fullToNew == active_stationary[i])[0][0])
                
                #add a tuple to the absorbing set
                self.absorbing.add(tuple(full_stationary))
                self.absorbing_active.add(tuple(active_stationary))


    def computeEigenvalues(self, k = 100):
        #compute largest k eigenvalues of transition matrix over active states

        #compute the largest real eigenvalues of P.
        eigs = scipy.sparse.linalg.eigs(self.P_active.transpose(), k=k, which="LR", maxiter=500000,
                                        ncv = self.num_states_active)

        #eigs = scipy.linalg.eig(self.P_active.toarray(), left=True, right=False)

        return eigs
                
    def computeTimescales(self, k):
        #use eigenvalues to compute timescales
        #use top k eigenvalues (excluding 1)

        #compute the largest real eigenvalues of P. 
        eigs = self.computeEigenvalues()
        eigs = eigs[0] #toss out the eigenvectors

        #take the real part for timescales
        e = np.real(eigs)

        #sort the eigenvalues, get rid of the eigenvalues equal to 1
        e.sort()
        e = e[::-1]
        if (len(self.absorbing) > 0):
            K = len(self.absorbing)
            e = e[K:]
        else:
            for i in range(len(e)):
                if np.abs(e[i]-1) > 1e-8:
                    K = i
                    break
            e = e[K:]

        #take the largest k and use them to compute timescales
        E = e[0:k]
        self.timescales = - self.lag / np.log(E)

        return self.timescales

    def solveForwardEquation(self, initial, T):
        #solve forward equation until T*lag to get probability vector as fn of time

        #set the initial condition
        p0 = np.zeros(self.num_states, dtype=float)
        p0[initial] = 1.0

        #init storage for probabilities for all time and set ic
        probs = np.zeros((T+1, self.num_states), dtype=float)
        probs[0,:] = p0

        #iteratively multiply by transition matrix
        for i in range(T):
            probs[i+1,:] = probs[i,:] * self.P

        #construct temporal discretization
        t = np.linspace(0, T*self.lag, T+1)

        return t, probs

    def solveForwardEquationActive(self, initial, T):
        #solve forward equation until T*lag to get probability vector as fn of time

        #set the initial condition
        p0 = np.zeros(self.num_states_active, dtype=float)
        initial_active = self.fullToNew[initial]
        if (initial_active > -1):
            p0[initial_active] = 1.0
        else:
            raise("The chosen initial state is not part of the active set")

        #init storage for probabilities for all time and set ic
        probs = np.zeros((T+1, self.num_states_active), dtype=float)
        probs[0,:] = p0

        #iteratively multiply by transition matrix
        for i in range(T):
            probs[i+1,:] = probs[i,:] * self.P_active

        #construct temporal discretization
        t = np.linspace(0, T*self.lag, T+1)

        return t, probs

    def solveForwardEquationActiveSpectral(self, initial, T):
        #solve forward equation by spectral decomp until T*lag to get 
        #probability vector as fn of time

        #set the initial condition
        p0 = np.zeros(self.num_states_active, dtype=float)
        initial_active = self.fullToNew[initial]
        if (initial_active > -1):
            p0[initial_active] = 1.0
        else:
            raise("The chosen initial state is not part of the active set")

        #init storage for probabilities for all time and set ic
        probs = np.zeros((T+1, self.num_states_active), dtype=float)
        #probs[0,:] = p0

        #construct temporal discretization
        t = np.linspace(0, T*self.lag*50, T+1)

        #get the left and right eigenvectors
        k = 250
        eigsL, vecsL = scipy.sparse.linalg.eigs(self.P_active.transpose(), k=k, which="LR")

        eigsR, vecsR = scipy.sparse.linalg.eigs(self.P_active, k=k, which="LR")

        print(eigsL)
        print(eigsL-eigsR)

        #sort them and search
        swaps = np.argsort(np.abs(eigsL))
        eig_sort = eigsL[swaps][::-1]
        print(eig_sort)

        #compute the spectral decomp
        for i in range(k):

            #get the eigenvalue 
            ev = eig_sort[i]
            print(i,ev)
            # if np.abs(np.imag(ev)) > 1e-6:
            #     print("Skipping complex")
            #     continue 

            #get the left  and right eigenvectors
            indexL = np.where(np.abs(eigsL - ev) < 1e-6)[0][0]
            indexR = np.where(np.abs(eigsR - ev) < 1e-6)[0][0]
            evL    = vecsL[:,indexL]
            evR    = vecsR[:,indexR]

            #check left eigvec for target component
            target_active = self.fullToNew[1182]
            # if (evL[target_active]/np.amax(np.abs(evL))) < 0.01:
            #     print("Skipping unimportant")
            #     continue


            #compute inner product between p(0) and right evect. scale by evL * evR
            inner = np.dot(p0, evR)
            scale = np.dot(evR,evL)

            #multiply by the left eigenvector and scale
            inner *= evL / scale

            #add in the time evolution term, eigenvale to the power t
            probs += np.real(np.outer(np.power(ev,t/50), inner))

        return t, probs

    def plotYieldCurves(self, initial_index, target_index, final_time, animate_time,
                        inv_map=None, samples_folder=None, sampling_target=None):
        #plot a yield curve for target state given an initial state and final time
        #can print out states with large probabilities if the inverse mapping is provided
        #can compare yield curve to sampling data if trajectory folder is provided
        #set the lag 
        lag = self.lag

        #compute probability for each state as a function of time
        num_steps = int(final_time / (animate_time*lag))
        t, p = self.solveForwardEquationActive(initial_index, num_steps)
        t = t * animate_time 

        #get a time scaling by getting number of digits
        num_digits = int(np.floor(np.log10(t[-1])))
        time_scaling = 10**num_digits
        t /= time_scaling

        #print out the notable final probabilities. Only if inv_map is provided
        if (inv_map is not None):

            #set the cutoff for a "notable" probability
            p_cut = 0.02
            print("Notable Final Probabilities:")

            #print the state and probability for all above the threshold
            for i in range(self.num_states_active):
                if p[-1,i] > p_cut:
                    state_full = np.where(self.fullToNew == i)[0][0]
                    print("State: {}, Prob: {}".format(inv_map[state_full], p[-1,i]))


        #plot the yield curve from initial state to target state
        new_idx = self.fullToNew[target_index]
        fig = plt.figure(1)
        ax  = fig.add_subplot(111)
        plt.plot(t,p[:,new_idx])
        legend_text = ["MSM Estimate"]

        #compare to sampling data if the trajectory folder is provided
        if (samples_folder is not None and sampling_target is not None):
            
            print("Computing Yield Curve estimate from sampling data")
            t_s, p_s, samples = computeSampledTargetProbability(samples_folder, 
                                                                sampling_target,
                                                                animate_time)
            if samples > 0:
                t_s /= time_scaling
                ax.plot(t_s, p_s)
                legend_text.append("Brownian Dynamics ({} samples)".format(samples))

        #format the plot axes and what-not
        ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
        ax.set_ylabel("Yield", fontsize = 20)

        #set num ticks
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=4)

        #set tick label sizes
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        #add legend
        ax.legend(legend_text, prop={'size':14})

        #remove whitespace and plot
        plt.tight_layout()

        #show the plot
        plt.show()

        return


##################################################################
############# Helper Functions   #################################
##################################################################

def getTrajType(type_string):
    #set boolean flags for each trajectory type using the supplied string

    #default each bool to false
    longT, shortT, otherT = False, False, False

    #check the string for identifying characters
    if ("L" in type_string):
        longT = True
    if ("S" in type_string):
        shortT = True
    if ("D" in type_string):
        otherT = True

    return longT, shortT, otherT


def getParameterMap():
    #read the parameter map text file to determine which parameter sets
    #correspond to which parameters

    #set the file location and open it
    param_file_loc = "../MSM/parameterMap.txt"
    param_file = open(param_file_loc, 'r')
    Lines = param_file.readlines()

    #init storage for the two parameter values and trajs types and read them in by line
    H = []
    P = []
    traj_types = []
    for line in Lines:
        line = line.split()
        H.append(line[1])
        P.append(line[2])
        traj_types.append(line[3])

    #close the parameter file and return the parameter values
    param_file.close()
    return H, P, traj_types


def loadStateDict(data_version = '', refine = False):
    '''
    Load the state dictionary. This is typically done from the MSM folder already, 
    but ratcheting will try to do this from the simulation folder. Start by going back
    a level, then into the MSM directory to always reach it. 

    If refinement is requested, also load the list of states that needed refinement
    as well as the refined dictionary.
    '''

    #check for refinement
    if refine:

        #set the file location for the refined states list
        refined_states_loc = "../MSM/data" + data_version + "/refineStates"
        refined_dict_loc   = "../MSM/data" + data_version + "/stateDictRefined"

        try:

            #load the refined states
            with open(refined_states_loc, 'rb') as f:
                refineList = pickle.load(f)
            print("Set of refinement candidates loaded from {}".format(refined_states_loc))

            #load the dict with refined states
            with open(refined_dict_loc, 'rb') as f:
                stateDict = pickle.load(f)

        except:
            print("A refinement file {} was not found.".format(refined_states_loc),\
            "Please create this before performing requesting a refined discretization.")
            raise()

    #otherwise load an empty refine list and the base state dictionary
    else:

        #create empty list and set the dict location
        refineList = []
        base_dict_loc = "../MSM/data" + data_version + "/stateDict"

        #load the dict with base states
        with open(base_dict_loc, 'rb') as f:
            stateDict = pickle.load(f)

    return stateDict, refineList

def loadMSMs(data_version = '', refine = True, restrict = [], verbose = True):
    #load the MSMs and place them in a list. Gather relevant parameters.

    #check if restricting the MSMs to a subset
    r_flag = False
    if len(restrict) > 0:
        if (verbose):
            print("Restricting MSMs to ensembles...")
            print(restrict)
        r_flag = True

    #init list for MSMs
    MSMs = []

    #search the msm folder for all the MSMs, add to list
    msm_folder = 'msm' + data_version + '/'
    if (refine):
        name_convention = "msmR"
    else:
        name_convention = "msm"
    msm_files = fnmatch.filter(os.listdir(msm_folder), name_convention+"*")
    msm_files = np.sort(msm_files)
    for msm_file in msm_files:

        #write the file path, get the numbered index
        full_path = msm_folder+msm_file
        msm_index = int(msm_file.split(name_convention)[1])

        #check if being restricted
        if (r_flag and msm_index not in restrict):
            continue

        #load the MSM and append to list
        try:
            with open(full_path, 'rb') as f:
                msm = pickle.load(f)
            MSMs.append(msm)

            if (verbose):
                spaces = 6-len(msm_file)
                print("Loaded {}.{} Active set length is {}.".format(msm_file,' '*spaces, len(msm.active_set)))
        except:
            print("Could not find {}".format(msm_file))
            raise()

    #get the number of ensembles found
    K = len(MSMs)

    #access the energy parameters for each ensemble, (P,H)
    E2 = np.zeros(K, dtype=object)
    for k in range(K):
        p = MSMs[k].Ep
        h = MSMs[k].Eh
        E2[k] =[p,h]

    #return the data
    return MSMs, K, E2


##################################################################
############# Trajectory Extraction   ############################
##################################################################


def exclude(pair):
    #define rules to exclude a trajectory

    return False


def manualMapToState(stateDict, traj, frames, refine=False, refineList=[]):
    #map the trajectory in state space to a non-negative state index

    #init an array for the path in state space
    state_path = np.zeros(frames, dtype=int)

    #loop over tuples, get state index
    for i in range(frames):
        reducedState = [traj[i][0], traj[i][1], traj[i][4]]

        #check if performing refinement
        if refine:

            #check if this state is to be refined. if so use 5 coordinate rep
            if tuple(reducedState) in refineList:
                reducedState = traj[i]

        pair = tuple(reducedState)
        if (exclude(pair)):
            return np.zeros(0)
        state = stateDict[pair]
        state_path[i] = state

    return state_path


def extractTrajectories(dtrajs, folder, traj_files, stateDict, refineList = []):
    #extract trajectories stored in npy files, append to dtrajs

    #check for refinement
    refine = False
    if len(refineList) > 0:
        refine = True

    #loop over trajectories to get dtrajs
    for file in traj_files:

        #append the folder to get full path
        path = folder + file

        #filter the files under some criteria
        file_num = int(path.split('traj')[2].split('.')[0])
        if file_num < 0:
            continue

        #try to load the file (may fail due to blowup for given seed)
        try:
            data = np.load(path)
        except:
            continue

        #get the number of frames in the trajectory
        frames = len(data)

        #convert to an integer state trajectory
        #print(file)
        traj = manualMapToState(stateDict, data, frames, refine, refineList)
        #print(traj)
        #print(file, data[-1])

        #append to dtraj if there is a non-trivial trajectory
        if (len(traj)):
            dtrajs.append(traj)

    return

def getDtrajs(folder, stateDict, refineList, longT=True, shortT=False, otherT=False):
    #extract all the desired trajectories and return a list of them

    #init list to store trajectories
    dtrajs = []

    #check which trajectories we want to load
    if longT:

        #get list of all long trajectories
        try:
            trajs = fnmatch.filter(os.listdir(folder), 'traj*.npy')
            print("Extracting long trajectories. Found {}".format(len(trajs)))
            extractTrajectories(dtrajs, folder,  trajs,  stateDict, refineList)

        except:
            print("Long trajectories requested but could not be found")
            raise()

    if shortT:

        #get list of all short trajectories
        try:
            trajs = fnmatch.filter(os.listdir(folder+"short/"), 'traj*.npy')
            print("Extracting short trajectories. Found {}".format(len(trajs)))
            extractTrajectories(dtrajs, folder+"short/",  trajs,  stateDict, refineList)

        except:
            print("Short trajectories requested but could not be found")
            raise()
            
    if otherT:

        #get list of all other trajectories (disassembly trajs)

        #loop over relevant folders
        for traj_folder in glob.glob(folder+"state12*"):
            
            trajs = fnmatch.filter(os.listdir(traj_folder+"/"), 'traj*.npy')
            print("Extracting trajectories from {}. Found {}".format(traj_folder, len(trajs)))
            extractTrajectories(dtrajs, traj_folder+"/",  trajs,  stateDict, refineList)

    
    #count how many trajectories were loaded
    num_trajs = len(dtrajs)
    if (num_trajs == 0):
        print("No trajectories could be loaded from folder {}. Exiting".format(folder))
        raise()
    else:
        print("{} trajectories were loaded and discretized.".format(num_trajs))

    #return the list of dtrajs
    return dtrajs


############################################################################
################### MSM Misc Testing #######################################
############################################################################


def getTrajFolder(msm, initial_state):
    #determine which folder to search for trajectories for the specified test

    #get energy parameters
    p, h = msm.Ep, msm.Eh

    #get the base folder from these parameters
    traj_folder = "../trajectories/P{}H{}/".format(p,h)

    #get any modifications based on the starting state
    init_state_3 = (initial_state[0], initial_state[1], initial_state[-1])
    if init_state_3 != (0,0,0):
        traj_folder += "state{}_{}_{}/".format(*init_state_3)

    #return the folder
    return traj_folder


def MSMtesting(msm, initial_state, target_state, data_version, refine):
    #test MSM by computing a yield curve and comparing to sampling data

    #set animate time for the MSM and final time for the tests
    animate_time = 25
    final_time = 800000

    #load the dictionary of states
    stateDict, refineList = loadStateDict(data_version=data_version, refine=refine)
    num_states = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set the initial and target state indices using the stateDict
    initial_index = stateDict[initial_state]
    target_index  = stateDict[target_state]

    #print info on the target transitions
    print("Ways to enter target state:")
    print(msm.P.getcol(target_index))
    print(msm.count_matrix.getcol(target_index))

    print("Ways to exit target state:")
    print(msm.P.getrow(target_index))
    print(msm.count_matrix.getrow(target_index))

    #print info on number of entries in transition matrix
    num_states = msm.num_states
    nz = msm.count_matrix.count_nonzero()
    print("{} nonzero entries out of {}, {} %".format(nz, num_states*num_states, float(nz)/float(num_states)))

    #plot yield curves
    traj_folder = getTrajFolder(msm, initial_state)
    sampling_target = (target_state[0], target_state[1], target_state[-1])
    msm.plotYieldCurves(initial_index, target_index, final_time, animate_time, 
                        inv_map, traj_folder, sampling_target)

    return


def MSMtestingLoad(msm_loc,initial_state=(0,0,0), target_state=(12,30,30),
               data_version='', refine = False):
    #perform MSM testing by loading from pickle

    #try loading the MSM
    try:
        with open(msm_loc, 'rb') as f:
            msm = pickle.load(f)
    except:
        print("Could not load MSM at location {}".format(msm_loc))
        raise()

    #do the testing
    MSMtesting(msm, initial_state, target_state, data_version, refine)

    return




def MSMtestingScratch(folder, lag, initial_state=(0,0,0), target_state=(12,30,30),
               data_version='', refine = False, 
               longT=True, shortT=False, otherT=False):
    #do msm testing by constructing it manually

    #create the msm using given folder and lag
    msm = createMSM(folder, lag, data_version=data_version, refine=refine, 
                   longT=longT, shortT=shortT, otherT=otherT)

    #do the testing
    MSMtesting(msm, initial_state, target_state, data_version, refine)

    #save the msm so we can load it in the future
    with open("msm_tests/msmP{}H{}".format(msm.Ep,msm.Eh), 'wb') as f:
        pickle.dump(msm, f)

    return


def computeSampledTargetProbability(folder, target_state, animate_time):
    #compute p(t) estimated using sampled trajectories

    #get all npy files in the given folder
    try:
        trajs = fnmatch.filter(os.listdir(folder), '*.npy')
    except:
        print("No such file or directory: {}".format(folder))
        return np.array([0]), np.array([0]), 0

    #if none are found, return empty lists
    if (len(trajs) == 0):
        print("No npy files were found in {}".format(folder))
        return np.array([0]), np.array([0]), 0

    #init storage
    frames = 0

    for npy_file in trajs:

        #load the npy file
        try:
            #print(npy_file)
            path = np.load(folder+npy_file)
        except:
            continue

        if (frames == 0): #do the first time setup
            frames     = len(path)
            new_frames = frames
            p          = np.zeros(frames, dtype=float)
            samples    = np.zeros(frames, dtype=float)
            samples[0] = 0

        else: #test if later path has more frames
            new_frames = len(path)
            if new_frames > frames:
                p_new = np.zeros(new_frames)
                p_new[0:(frames)] = p
                p = p_new

                samples_new = np.zeros(new_frames)
                samples_new[0:(frames)] = samples
                samples = samples_new

                frames = new_frames

        count = 0
        for pair in path:
            if (pair[0] == target_state[0] and pair[1] == target_state[1] and pair[-1] == target_state[2]):
                p[count] += 1
                #break

            samples[count] += 1
            count += 1

    #get time discretization and normalize p
    t = np.linspace(0, animate_time*frames, frames)
    p = p / samples

    print(p)
    print(samples)

    return t, p, int(np.max(samples))

def computeCommittor(P, initial, target, num_states):
    #compute the committor probability for each state

    #convert single valued initial and target to lists
    initials = list(np.array([initial]).flat)
    targets  = list(np.array([target]).flat)

    #init the system matrix and rhs
    R = np.array(P.todense())
    b = np.zeros(num_states)

    #subtract identity from P to get R
    for i in range(num_states):
        R[i][i] -= 1.0 - 1e-16

    #set boundary conditions
    #initial state
    for i in initials:
        for j in range(num_states):
            if j == i:
                R[i,j] = 1.0
            else:
                R[i,j] = 0.0

    #target state
    for t in targets:
        for j in range(num_states):
            if j == t:
                R[t,j] = 1.0
            else:
                R[t,j] = 0.0

    #set vector
    b[targets] = 1.0

    #do the solve and return the result
    x = np.linalg.solve(R,b)
    return x

def computeMFPT(P, initial, target, num_states):
    #compute mean first passage time from initial states to target states

    #need to solve the linear system tau=1+Q*tau

    #convert single valued initial and target to lists
    initials = list(np.array([initial]).flat)
    targets  = list(np.array([target]).flat)

    #get list of absorbing states to remove
    absorbing = []
    for i in range(num_states):
        if P[i,i] > 1-3e-2:
            absorbing.append(i)
            if i in initials:
                initials.remove(i)

    # print(targets)
    # print("Absorbing: {}".format(absorbing))

    #init the system matrix and rhs
    Q = np.array(P.todense())

    #construct Q by removing rows and columns in the target from P
    to_remove = list(set(targets) | set(absorbing))
    Q = np.delete(Q, to_remove, 0)
    Q = np.delete(Q, to_remove, 1)

    #solve the system
    off_target = num_states - len(to_remove)
    b = np.ones(off_target)
    A = np.identity(off_target)*(1+1e-10)-Q
    # Ainv = np.linalg.pinv(A)
    # tau = np.matmul(Ainv,b)
    tau = np.linalg.solve(A, b)
    print(A*tau)
    print(tau)

    #get the indexing map due to removing the target states
    new_indices = []
    count = 0
    for i in range(num_states):
        if i in targets or i in absorbing:
            count +=1
            new_indices.append(-1)
        else:
            new_indices.append(i-count)

    #average the mean first passage times over the initial states
    avg = 0
    for i in range(len(initials)):
        new_index = new_indices[initials[i]]
        value = tau[new_index]
        print(i, new_index, value)
        avg += value

    avg /= len(initials)
    return avg


def computeMFPTsampling(P, initial, target, num_states, inv_map, globalActive):
    #estimate mean first passage times by sampling

    #convert single valued initial and target to lists
    initials = list(np.array([initial]).flat)
    targets  = list(np.array([target]).flat)

    #set sampling parameters
    num_samples = 100
    max_iters = 256

    #choose initial state at random
    i = random.choice(initials)

    #do sampling
    average = 0
    for sample in range(num_samples):
        current_state = i
        # input()
        for t in range(max_iters):

            #get the row of P cooresponding to current state
            row = P.getrow(current_state)
            row = row.A[0]

            #sample from the row
            current_state = np.random.choice(range(len(row)),p=row)
            # print(t,inv_map[globalActive[current_state]])
            if current_state in targets:
                print("Hit target in {} steps".format(t))
                average += t+1
                break

    average /= num_samples

    return average






##################################################################
############# MSM Convergence Testing  ###########################
##################################################################


def time_scale_analysis(dtraj, LAGS, Eh, Ep, num_states, k = 10):
    #compute MSMs at various lag times, plot timescales, do markovity test

    #get number of tests to do and init storage for timescales
    num_lags = len(LAGS)
    timescales = np.zeros((k, num_lags), dtype=float)

    for i in range(num_lags):
        lag = LAGS[i]

        #print progress message
        print("Computing MSM {} of {}".format(i+1,num_lags))

        M = MSM(dtraj, lag, Eh, Ep, num_states)
        t = M.computeTimescales(k)
        timescales[:,i] = np.array(t)

   
    #plot the timescales
    plt.figure()
    for i in range(3,k):
        plt.plot(LAGS, timescales[i])

    plt.show()


    return

def convergenceLag(folder, initial_state, target_state, data_version='', refine=False):
    '''
    Test convergence of the MSMs by constructing yield curves from initial_state to 
    target_state as a function of lag time. The curves involve all the timescales
    important for that particular assembly, so this should be more robust than a 
    CK test

    Set a selection of lag times to test apriori, construct the MSMs for each lag time,
    then save them together for later re-use.
    '''

    #set the desired lag times and other constant parameters
    LAGS         = [1, 25, 50, 75, 100, 125, 150]
    animate_time = 25
    Ep           = float(folder.split('P')[1].split('H')[0])
    Eh           = float(folder.split('P')[1].split('H')[1].split('/')[0])

    #get the stateDict dictionary mapping
    stateDict, refineList = loadStateDict(data_version=data_version, refine=refine)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get indices of the initial and target states
    num_states = len(stateDict)
    initial_index = stateDict[initial_state]
    target_index  = stateDict[target_state]
    print("Target state index is {}".format(target_index))
    print("There are {} states".format(num_states))

    #try to load an existing collection of MSMs at the lag time
    msm_conv_data_loc = "../MSM/msm_tests/convP{}H{}".format(Ep,Eh)
    try:

        with open(msm_conv_data_loc, 'rb') as f:
            MSM_list = pickle.load(f)
            print("MSM convergence data loaded from {}".format(msm_conv_data_loc))

    except:

        #create the MSMs at each lag from scratch
        MSM_list = []
        for k in range(len(LAGS)):
            lag = LAGS[k]
            M = createMSM(folder, lag, data_version=data_version, refine=refine,
                          longT=True, shortT=True, otherT=True)
            MSM_list.append(M)

        #pickle the list of MSMs for later use
        with open(msm_conv_data_loc, 'wb') as f:
            pickle.dump(MSM_list, f)

    #loop over the MSM list and compute yield curves for the desired states
    target_times = []
    target_probs = []
    use_lags = [25, 125, 150]
    for k in range(len(LAGS)):

        lag = LAGS[k]
        if lag not in use_lags:
            continue

        #construct the MSM object
        M = MSM_list[k]

        #get the target probability evolution
        num_steps = int(800000.0 / (animate_time*lag))
        t, p = M.solveForwardEquationActive(initial_index, num_steps)
        t = t * animate_time
        t /= 1e5
        target_times.append(t)
        global_t = M.fullToNew[target_index]
        target_probs.append(p[:,global_t])

    #get the sampled values
    print("Computing fc estimate from sampling")
    traj_folder = getTrajFolder(M, initial_state)
    sampling_target = (target_state[0], target_state[1], target_state[-1])
    t_s, p_s, samples = computeSampledTargetProbability(folder, sampling_target, animate_time)
    t_s /= 1e5

    #plot the data and format it
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    #plot sampled data
    plt.plot(t_s, p_s, '-')

    #plot each MSM data
    for k in range(len(target_probs)):
        plt.plot(target_times[k],target_probs[k])

    #label 
    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("Yield", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    # plt.legend(["Sampled data", "MSM Lag 1", "MSM Lag 25","MSM Lag 50","MSM Lag 75",
    #             "MSM Lag 100","MSM Lag 125","MSM Lag 150"], prop={'size':15})
    plt.show()

    return


def convergenceSamples(folder, data_version='', refine=False):
    #test the convergence of the MSM as a function of number of samples

    #get the stateDict dictionary mapping
    stateDict, refineList = loadStateDict(data_version=data_version, refine=refine)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get info about the state space and target
    num_states = len(stateDict)
    initial_index = stateDict[(12,29,27)]
    target_index  = stateDict[(12,29,27)]
    target_state  = (12,29,27)
    print("Target state index is {}".format(target_index))
    print("There are {} states".format(num_states))

    #get the ensemble energies
    Ep = float(folder.split('P')[1].split('H')[0])
    Eh = float(folder.split('P')[1].split('H')[1].split('/')[0])

    #set lag and animation period
    lag = 100
    animate_time = 25
    num_steps = int(700000.0 / (animate_time*lag))

    #make list for all trajectories
    dtrajs = []

    #get all npy files
    trajs = fnmatch.filter(os.listdir(folder), 'traj*.npy')

    #append trajectories from sims to dtrajs 
    print("Extracting long trajectories. Found {}".format(len(trajs)))
    extractTrajectories(dtrajs, folder,  trajs,  stateDict)

    #create MSMs gradually using more data
    num_trajs = [3, 6, 12, 24, 48, 96, 188]
    target_probs = []
    for k in range(len(num_trajs)):
        nt = num_trajs[k]

        #construct the MSM object
        M = MSM(dtrajs[0:nt], lag, Eh, Ep, num_states)

        #get the target probability evolution
        t, p = M.solveForwardEquationActive(initial_index, num_steps)
        t = t * animate_time
        t_ind = M.fullToNew[target_index]
        target_probs.append(p[:,t_ind])

    #get the sampled values
    print("Computing fc estimate from sampling")
    sampling_target = (target_state[0], target_state[1], target_state[-1])
    t_s, p_s, samples = computeSampledTargetProbability(folder, sampling_target, animate_time)

    #plot
    plt.figure()

    #plot sampled data
    plt.plot(t_s, p_s, '-')

    #plot each MSM data
    for k in range(len(num_trajs)):
        plt.plot(t,target_probs[k])

    #label 
    plt.xlabel("Time")
    plt.ylabel("Target State Probability")
    plt.legend(["Sampled data", "MSM-3", "MSM-6","MSM-12", "MSM-24","MSM-48", "MSM-96","MSM-188"])

    
    #show
    plt.show()

    return


##################################################################################
################## Refine State Space Discretization #############################
##################################################################################


def comparePijByStartState(folder1, folder2):
    '''
    Test how transition matrix entries chage in an MSM when adding additional sampling
    data that was initialized in a different initial configuration. Compare values in the
    augmented MSM to base values, and check if they are within some multiple of the 
    standard sampling error. If so, mark that state as a problem state. Return all such 
    states. 

    This serves as a test for which states the reaction coordinate used is defining
    a state ambiguously, i.e. two distinct microstates with same macrostate. 
    '''

    #get the stateDict dictionary mapping
    stateDict, refineList = loadStateDict()
    num_states = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set a lag
    lag = 100
    animate_time = 25

    #get the base set of trajs to construct an MSM
    dtrajs = []
    trajs = fnmatch.filter(os.listdir(folder1), 'traj*.npy')
    extractTrajectories(dtrajs, folder1,  trajs,  stateDict)

    #construct the MSM object
    msm1 = MSM(dtrajs, lag, 1.2, 1.4, num_states)
    r,c = msm1.P.nonzero()

    #construct the second MSM with additional sampling data used and make another MSM
    trajs = fnmatch.filter(os.listdir(folder2), 'traj*.npy')
    extractTrajectories(dtrajs, folder2,  trajs,  stateDict)
    msm2 = MSM(dtrajs, lag, 1.2, 1.4, num_states)

    #init a list to store the problem states
    identified = []

    #set a tolerance for how many standard errors from the estimate is acceptable
    tol = 3

    #loop over nonzeros and print differences
    for i in range(len(r)):

        #get probability in this index for each MSM
        p1 = msm1.P[r[i],c[i]]
        p2 = msm2.P[r[i],c[i]]

        #get counts in base MSM to compute standard error
        c1 = msm1.count_matrix[r[i],c[i]]
        std = np.sqrt(p1*(1-p1)/float(c1))

        #check if above the tolerance for being considered
        if (np.abs(p1-p2) > tol*std and std > 1e-5):
            state1 = inv_map[r[i]]
            state2 = inv_map[c[i]]
            print("{} -> {}, p1 = {}, p2 = {}, std = {}".format( \
                  state1, state2, round(p1,4), round(p2,4), round(std,4)))
            identified.append(state1)

    #return the identified states
    return identified

def getStatesToRefine(base_folder):
    #get a list of candidate states for refinement 

    #list all starting states for the long sims
    start_states = ["12_29_27", "12_30_30", "12_30_26", "12_31_27", "12_31_29"]
    # start_states = ["12_29_27", "12_30_26", "12_31_29"]

    #try to load a file for candidate sets. if no such file, create from scratch
    set_name = "data/refineStates"
    try:
        with open(set_name, 'rb') as f:
            candidates = pickle.load(f)
        print("Set of candidates loaded from {}".format(set_name))
    except:
        print("File {} not found. Creating candidates from scratch".format(set_name))
        candidates = set()

    #loop over each pair to get states to look at 
    for i in range(len(start_states)):
        for j in range(len(start_states)):
            if i != j:

                #set the folders containing the data
                f1 = base_folder + "state" + start_states[i] + "/"
                f2 = base_folder + "state" + start_states[j] + "/"

                #get the states for this pair
                print("Comparing data in {} to {}".format(f1, f2))
                states = comparePijByStartState(f1, f2)

                #loop over and add to set
                for state in states:
                    candidates.add(state)

                #display size of new set
                print("Candidate set now has {} states".format(len(candidates)))

    #save the set to disk
    with open(set_name, 'wb') as f:
        pickle.dump(candidates, f)
        print("Wrote set of candidate states to {}".format(set_name))


############################################################################
################### Create MSMs  ###########################################
############################################################################

def createMSM(folder, lag, data_version='', refine=False, longT=True, shortT=False,
              otherT=False):
    #manually construct an MSM using the trajectories in folder and specified lag

    #load the dictionary of states
    stateDict, refineList = loadStateDict(data_version=data_version, refine=refine)
    num_states = len(stateDict)

    #get the energy parameters from the folder name
    Ep = float(folder.split('P')[1].split('H')[0])
    Eh = float(folder.split('P')[1].split('H')[1].split('/')[0])

    #get a list of discrete trajectories using data in specified folders
    dtrajs = getDtrajs(folder, stateDict, refineList, longT=longT, shortT=shortT,
                       otherT=otherT)

    #set the minimum observation thresholds based on the trajectory types
    min_count = 1
    min_frame = 2
    if (otherT):
        min_count = 2
        min_frame = 4

    #construct the MSM object
    M = MSM(dtrajs, lag, Eh, Ep, num_states, min_count=min_count, min_frame=min_frame)

    #return MSM object
    return M

def batch_make_MSM(lag, data_version = '', refine = False):
    #make an MSM for every input in the parameter map file

    #get the parameter values as a function of input number
    H, P, traj_types = getParameterMap()

    #loop over the parameter to construct MSMs
    for i in range(35,len(H)):

        #get parameter values
        h = H[i]
        p = P[i]
        type_string = traj_types[i]

        #parse the type_string into meaningful values
        if ("N" in type_string): 

            #this parameter set is null. Skip MSM creation
            continue

        else:

            #get booleans for each sim type
            longT, shortT, otherT = getTrajType(type_string)


        #set folder path
        folder = "../trajectories/P{}H{}/".format(p,h)

        #check that the folder exists, continue to next set if not
        if (not os.path.exists(folder)):
            continue

        #create the MSM
        print("Creating MSM {} using parameters P{}H{}".format(i,p,h))
        M = createMSM(folder, lag, data_version=data_version, refine=refine,
                      longT=longT, shortT=shortT, otherT=otherT)

        #pickle the MSM according to its index number and the run type
        out_file = "msm" + data_version + "/msm"
        if (refine):
            out_file += "R"
        out_file += "{}".format(i)
        with open(out_file, 'wb') as f:
            pickle.dump(M, f)

    return



if __name__ == "__main__":

    #create one MSM
    folder = "../trajectories/P1.4H1.45/"
    lag = 125
    # initial_state = (12,29,60,57,27)
    # initial_state = (12,30,60,60,26)
    # initial_state = (12,31,29)
    # initial_state = (12,30,30)
    # initial_state = (12,31,27)
    initial_state = (0,0,0)
    # initial_state = (10,20,12)

    # target_state = (12,30,30)
    target_state = (12,29,60,57,27)
    # target_state = (12,30,60,60,26)
    # target_state = (12,30,26)
    # target_state = (12,31,29)
    # target_state=initial_state

    #createMSM(folder,lag,"V1",False)

    #do testing on MSM - compare dynamics curves or assess convergence

    # MSMtestingScratch(folder, lag, data_version='',refine=True,
    #                   longT=True, shortT=True, otherT=True, 
    #                   initial_state=initial_state,target_state=target_state)

    # MSMtestingLoad("msm_tests/msmP1.25H1.4", data_version='', refine=True,
    #                initial_state=initial_state, target_state=target_state)

    #27 and 6 are adjacent
    # MSMtestingLoad("msm/msmR7", data_version='', refine=True,
    #                initial_state=initial_state, target_state=target_state)

    #convergenceSamples(folder+"state12_29_27/")
    convergenceLag(folder, initial_state, target_state, refine=True)

    #refine the state space discretization by comparing disassembly trajs in folder
    # getStatesToRefine(folder)

    #batch make MSMs without any extraneous output
    # batch_make_MSM(lag, refine=True)


    #make a refine list manually
    # stateDict, refineList = loadStateDict()
    # num_states = len(stateDict)
    # inv_map = {v: k for k, v in stateDict.items()}

    # refineList = [inv_map[i] for i in range(num_states)]
    # print(refineList)

    # set_name = "data/refineStates"
    # with open(set_name, 'wb') as f:
    #     pickle.dump(refineList, f)
    #     print("Wrote set of candidate states to {}".format(set_name))

