'''
This block of code creates a matrix of rbf interpolants for the probability transition
matrix given the results from the MSM in each ensemble. The MSM is assumed to be the 
custom MSM class, not the pyemma MSM.

This transition matrix acts on the full state space, i.e. the union of states visited 
in each ensemble. It is frequently the case that a particular transition is not observed 
in every ensemble (i.e. those with low probabilities), so we allow empty values for the
interpolation. These values will simply be interpolated, instead of being set to 0.

There are a few coded methods for handling entries that have estimates in one ensemble 
but maybe not others. The default is to set such entries to zero and to give all estimates
zero noise. Having noise in the interpolants smooths things out too much in 2 dimensions,
especially if there is only one ensemble with an estimate. Doing it this way gives the best
interpolations of the MSM between nodes, as well as nearly perfectly preserving the 
dynamics at the nodes.
'''

import sys
import os
import fnmatch
import inspect

#set a path to the simulation folder to do relative importing of database class in ratchet
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir) 
assemblydir = parentdir + "/simulation/"
sys.path.insert(0, assemblydir) 

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse

from collections import defaultdict
from collections import Counter

import multiprocessing
from multiprocess import Process, Manager

import shelve
import pickle
import sqlite3
import timeit
import time

import rbf
from rbf.interpolate import RBFInterpolant

import manualMSM
from manualMSM import MSM
from manualMSM import loadMSMs

############################################################################
########### Helper Functions For Probability Extraction ####################
############################################################################

def getEnsembleIndex(state, MSMs, k):
    #get the index of state in the active set in ensemble k 

    index = MSMs[k].fullToNew[state]

    return index

def getGlobalActive(MSMs, K):
    #get the union of all states active in any ensemble

    global_active = set()
    for k in range(K):
        global_active = global_active.union(set(MSMs[k].active_set))
    global_active = list(global_active)

    return global_active


def getTargetSetUnion(natural_state_i, MSMs, K):
    #get the union of states that state i interacts with across all ensembles

    non_zeros = set()
    for k in range(K):
        #get index of natural state i in MSM k
        i_ind = getEnsembleIndex(natural_state_i, MSMs, k)

        #if index is non-negative, get nonzero entries of the row
        if i_ind > -1:
            nzk = MSMs[k].P_active.getrow(i_ind).nonzero()[1]
        
            #get the natural index of the nonzeros
            nzk_nat = np.array(MSMs[k].active_set)[nzk]

            #take union wth previously found non-zero states
            non_zeros = non_zeros.union(set(nzk_nat))

    return non_zeros

############################################################################
########### Probability Extraction from MSMs ###############################
############################################################################

def getProbList(natural_state_i, natural_state_j, MSMs, K, zero_tol):
    #get a list of transition probabilities from states i to j in each ensemble

    #init an nparray to store values and check for zeros
    probs = np.zeros(K)
    zeroI = np.zeros(K)

    #fill the nparray with values from the MSMs in each ensemble
    for k in range(K):

        #get the ensemble index of natural states i and j
        i_ind = getEnsembleIndex(natural_state_i, MSMs, k)
        j_ind = getEnsembleIndex(natural_state_j, MSMs, k)

        #check if the states are in the active set or not
        if (i_ind == -1 or j_ind == -1): 
            p = []
            zeroI[k] = 1
        else:
            #states are present. get probability and number of samples from row
            p = MSMs[k].P_active[i_ind, j_ind]
            probs[k] = p
            if (p < zero_tol):
                zeroI[k] = 1

    #check if there are zero values - if so, replace with empty lists
    #todo - maybe modify how the empty lsts are handled?
    zero_check = np.sum(zeroI)
    if (zero_check == K):
        probList = list(probs)

    elif (zero_check > zero_tol):
        #replace zeros with empty list
        probList = []
        for k in range(K):
            if (zeroI[k] == 1):
                probList.append([])
            else:
                probList.append(probs[k])
    else:
        probList = list(probs)

    return probList

def getStandardError(natural_state_i, natural_state_j, pList, MSMs, K):
    #get the standard error for each transition as the standard deviation of a multinomial
    #distribution over the row i

    #init list for standard errors
    s = []

    #loop over ensembles and fill with standard error
    for k in range(K):

        #get the ensemble indices
        i_ind = getEnsembleIndex(natural_state_i, MSMs, k)
        j_ind = getEnsembleIndex(natural_state_j, MSMs, k)

        #check if the states are in the active set or not, get counts
        if (i_ind == -1 or j_ind == -1): 
            c = 0
        else:
            c = MSMs[k].count_matrix_active[i_ind, j_ind]

        #get the k-th element of p and compute standard error
        element = pList[k]
        if (type(element) is list or c == 0):
            #if no observation, make stddev 0
            s.append(0.0)
        else:
            s.append(np.sqrt(element*(1.0-element) / float(c)))

    return s

def getFullPTMs(MSMs, K):
    #construct probability transition matrices on the same state space for each ensemble
    #PTM will be a sparse array of lists containing entry in each ensemble
    #also constructs lists of the standard error of each measurement using multinomials

    #define a tolerance to be considered 0
    zero_tol = 1e-7

    #init default dicts for the matrices
    PTM = defaultdict(int)
    SEM = defaultdict(int)

    #compute the union of states in the active sets in all ensembles
    global_active = getGlobalActive(MSMs, K)

    #get the total number of active states
    nsa = len(global_active)

    #loop over each active state in global_active
    for i in range(nsa):

        #print progress message
        if (i % 100 == 0):
            print("Constructing row {} of {} of global transition matrix".format(i,nsa))

        #get the natural index of state i
        natural_state_i = global_active[i]

        #get the set of states that natural state i communicates with across all ensembles
        non_zeros = getTargetSetUnion(natural_state_i, MSMs, K)

        #loop over destination states with non zero entries
        for natural_state_j in non_zeros:

            #get a probability list for the transitions from nsi to nsj
            probList = getProbList(natural_state_i, natural_state_j, MSMs, K, zero_tol)

            #get index of nsj in active state and input and place the list in the PTM
            active_j = np.where(global_active == natural_state_j)[0][0]
            PTM[(i, active_j)] = probList

        #loop back over and construct the standard error as sqrt(p_i(1-p_i)/N)
        for natural_state_j in non_zeros:
            
            #get active state j
            active_j = np.where(global_active == natural_state_j)[0][0]

            #get pList to construct the std dev
            pList = PTM[(i,active_j)]

            #compute std dev
            SEM[(i, active_j)] = getStandardError(natural_state_i, natural_state_j, 
                                                  pList, MSMs, K)

    #return the matrices with probabilities and standard deviations
    return PTM, SEM, global_active

############################################################################
########### Interpolation Functions ########################################
############################################################################

def findOptimalShape(x, y, basis, sample_points=30, sig=0, verbose=False):
    #determine the optimal shape parameter for the rbf by sampling
    #returns the rbf interpolant

    #set parameters for the optimization
    mag_tol = 8    #base 10 exponent for largest coefficient to consider
    shapeLB = -1   #lower bound for search space (logspace)
    shapeUB = 2  #upper bound for ...
    default = 9 #default shape parameter value, for when it is unclear what to use

    #construct a set of shape parameters to perform error minimization over
    e_set = np.logspace(shapeLB, shapeUB, sample_points)
    e_set = np.flip(e_set, axis=0)
    errs  = np.ones(sample_points)*100

    #set up variables for minimum error storage
    min_error = 100

    #get number of data points
    num_points = len(x)

    #loop over the shape parameters to compute errors
    for i in range(sample_points):

        #get the interpolant and basis coefficients
        e = e_set[i]
        I = RBFInterpolant(x, y, sigma = sig, phi=basis, order=-1, eps=e)
        c = I.phi_coeff
        if (c is None or np.log10(np.abs(c[0])) > mag_tol):
            #the matrix has become non spd, perform minimization on given examples

            #cut off values not being considered
            errs = errs[0:i]
            e_set = e_set[0:i]

            #print information message if requested
            if (verbose):
                print("The system becomes unstable when eps={}".format(e))
                print("Optimization performed over eps > {}".format(e))

            #end the sampling loop
            break

        Kinv = I.inv
        if (verbose):
            print(i, c[0])

        #init accumulator for LOOCV error sum
        S = 0
        #compute the error as sum over basis coefficients - use L1 norm for outliers
        for k in range(num_points):
            e_term = c[k] / Kinv[k][k]
            S += e_term * e_term
            #S += np.abs(e_term) 

        #set the error and check for minimum
        errs[i] = S
        #print(S)
        if (S < min_error):
            min_error = S
            bestInterp = I
            bestShape  = e

    # plt.loglog(e_set, errs)
    # plt.show()

    #get the derivative of the errors
    d = np.gradient(errs, e_set)
    # print(e_set)
    # plt.plot(e_set, d)
    # plt.axis([0,100,-1,1])
    # plt.show()

    #remove noise from the gradient to avoid spurious minima
    noise_tol = 1e-7
    d[np.where(np.abs(d) < noise_tol)[0]] = 0.0

    #search for where derivative changes sign from minus to plus, get local minima
    minima = np.where(np.sign(d[:-1]) > np.sign(d[1:]))[0] + 1
    min_vals = errs[minima]

    # print(e_set[minima])

    #if there are no local minima, use default shape parameter
    if (len(minima) == 0):
        min_index = np.abs(e_set - default).argmin()
        min_error = errs[min_index]
        bestShape = e_set[min_index]
        bestInterp = RBFInterpolant(x, y, sigma = sig, phi=basis, order=-1, eps=bestShape)
        return bestInterp, bestShape, min_error, e_set, errs

    #get smallest local min and compare to global sampling
    min_min  = np.min(min_vals)

    #determine which critical point to use - smallest error
    min_index = minima[np.where(min_vals == min_min)[0][0]]

    #check if the smallest local min is sufficiently larger than the global min
    #if it is, there is likely an error plateau -> choose default shape parameter
    if (min_min > 1.1 * min_error):
        if (verbose):
            print("Global min =/= local min. Likely plateu - choosing default eps")
        min_index = np.abs(e_set - default).argmin()

    min_error = errs[min_index]
    bestShape = e_set[min_index]
    bestInterp = RBFInterpolant(x, y, sigma = sig, phi=basis, order=-1, eps=bestShape)

    #return the interpolant and errors
    return bestInterp, bestShape, min_error, e_set, errs

def getFullRBF(x_vals, y_vals, sig, K):
    #get radial basis function interpolator given the data
    
    #set parameters for the interpolation
    basis = 'ga'
    sample_points = 250
    noise_tol = 0.9

    #set method for dealing with poorly sampled states
    method = 0  #set zeros, set no noise method
    # method = 1  #delete high noise, set no noise method
    # method = 2  #delete high noise and missing, set no noise method
    # method = 3  #delete high noise, set missing to 0 with noise

    #do pre-processing - eliminate empty lists from the data
    x = []
    y = []
    s = []

    if method == 0:
        for i in range(K):
            if (type(y_vals[i]) is list): 
                #append a 0 prob and 0 noise for missing entries
                x.append(x_vals[i])
                y.append(0.0)
                s.append(0.0)
            else:
                #append entry as normal, use 0 noise
                x.append(x_vals[i])
                y.append(y_vals[i])
                s.append(0.0)

    if method == 1:
        for i in range(K):
            if (type(y_vals[i]) is list):
                #append a 0 prob and 0 noise for missing entries
                x.append(x_vals[i])
                y.append(0.0)
                s.append(0.0)
            else:
                #if entry is above noise threshold, skip it
                p = y_vals[i]
                noise = sig[i] + 1e-12

                est_count = p * (1-p) / (noise * noise)
                if est_count < noise_tol:
                    continue
                else:
                    x.append(x_vals[i])
                    y.append(y_vals[i])
                    s.append(0.0)

    if method == 2:
        for i in range(K):
            if (type(y_vals[i]) is list):
                #skip
                continue
            else:
                #if entry is above noise threshold, skip it
                p = y_vals[i]
                noise = sig[i] + 1e-12

                est_count = p * (1-p) / (noise * noise)
                if est_count < noise_tol:
                    continue
                else:
                    x.append(x_vals[i])
                    y.append(y_vals[i])
                    s.append(0.0)

    if method == 3:
        for i in range(K):
            if (type(y_vals[i]) is list):
                #append a zero with noise
                x.append(x_vals[i])
                y.append(0.0)
                s.append(0.001)
            else:
                #if entry is above noise threshold, skip it
                p = y_vals[i]
                noise = sig[i] + 1e-12

                est_count = p * (1-p) / (noise * noise)
                if est_count < noise_tol:
                    continue
                else:
                    x.append(x_vals[i])
                    y.append(y_vals[i])
                    s.append(0.0)



    #check if all the entries are 0s, if so, skip this element
    S = np.sum(np.array(y))
    if S < 1e-10:
        return 0, False


    # print(x)
    # print(y)
    # print(s)

    #get the optimal shape interpolator
    I, eps, min_e, e_set, errs = findOptimalShape(x, y, basis, sample_points, s, 
                                                  verbose=False)
    # print(eps)

    #return interpolant
    return I, True


def fitEntry(key, values, stdev, E2, K):
    #fit the given values using RBF

    return getFullRBF(E2, values, stdev, K)

############################################################################
########### Interpolation Drivers (Serial, Parallel, Pickle, SQL) ##########
############################################################################

def getExtractedTransitionProbs(MSMs, K, data_version=''):
    #get all the extracted transition probabilities and errors to pass into the
    #interpolation code

    #todo add in refine option

    #try to load the array of list of transition elements
    try:

        #load the sparse matrix of transition probability elements
        with open("data" + data_version + "/ptm_full", 'rb') as f:
            PTM = pickle.load(f)

        #load sparse matrix of standard errors
        with open("data" + data_version + "/sem_full", 'rb') as f:
            SEM = pickle.load(f)

        #get global active set
        global_active = getGlobalActive(MSMs, K)

        #print success message
        print("Loaded the transition probability elements from disk")

    #if not found, compute it instead
    except:

        #get entries for the probability transition matrix without removing states
        print("Transition probabilities not found on disk at {}.".format("data"+ data_version))
        print("Gathering all non-zero transition matrix entries from scratch.")
        PTM, SEM, global_active = getFullPTMs(MSMs, K)

        #pickle them so we can load them next time
        with open("data" + data_version + "/ptm_ex", 'wb') as f:
            pickle.dump(PTM, f)

        with open("data" + data_version + "/sem_ex", 'wb') as f:
            pickle.dump(SEM, f)

    #return the data
    return PTM, SEM, global_active



def fitTransitionMatrixFull(data_version = ''):
    #Fit the entries of the transition matrix using all ensembles - serial
    #state space is union of all observed states from each ensemble

    #get the MSMs and useful parameters
    MSMs, K, E2 = loadMSMs(data_version=data_version)

    #get entries for the probability transition matrix without removing states
    PTM, SEM, global_active = getExtractedTransitionProbs(MSMs, K, data_version=data_version)

    #fit each entry of the transition matrix and return an array of RBF objects
    #the defaultdict return should be None, as it will not be called ideally
    Pfit = defaultdict()

    #loop over the keys in the PTM
    print("Constructing interpolants for all non-zero entries")
    num_keys = len(PTM.keys())
    c = 0
    loop = 1

    start = time.time()
    for key in PTM.keys():

        #get the interpolant
        rbf, success = fitEntry(key, PTM[key], SEM[key], E2, K)

        #store the interpolant if successfully constructed
        if (success):
            Pfit[key] = rbf

        #increment loop counter and print progress in 10% increments
        c += 1
        if int(100*float(c)/num_keys) > 10:
            print("Interpolated {}% of entries".format(10*loop))
            loop += 1
            c = 0
    end = time.time()

    #print elapsed time message
    print("Elapsed time is {} seconds for the interpolation".format(end-start))

    #pickle the array of interpolators
    with open("msm" + data_version + "/matrixI_new", 'wb') as f:
        pickle.dump(Pfit, f)


def getDict(MSMs, K, E2, num_procs, data_version = ''):
    #do the interpolation in parallel to get a dict of RBF objects

    #create a pool of parallel workers
    p = multiprocessing.Pool(num_procs)

    #get entries for the probability transition matrix without removing states
    PTM, SEM, global_active = getExtractedTransitionProbs(MSMs, K, data_version=data_version)

    #get list of keys
    keys = list(PTM.keys())

    #create a generator for the input to the fit function for each worker
    fit_input = ((key, PTM[key], SEM[key], E2, K) for key in keys)

    #start timer
    start = time.time()

    #compute the rbfs for each key - get boolean filter for successful ones
    rbfs = p.starmap(fitEntry, fit_input)
    bool_map = [b[1] for b in rbfs]

    #filter the keys and rbfs according to bool_map
    keys = [i for (i,v) in zip(keys, bool_map) if v]
    rbfs = [i[0] for (i,v) in zip(rbfs, bool_map) if v]

    #make a dictionary out of the keys and rbfs
    Pfit = dict(zip(keys, rbfs))

    #end timer
    end = time.time()

    #print elapsed time message
    print("Elapsed time is {} seconds for the interpolation".format(end-start))

    #return the dict of interpolants
    return Pfit

def getSQL(db_name, MSMs, K, E2, num_procs, data_version=''):
    #do the interpolation in parallel and store in a SQL database

    #create a pool of parallel workers
    p = multiprocessing.Pool(processes=num_procs)

    #create an sql database
    conn = sqlite3.connect(db_name)

    #create a table for interpolants
    conn.execute('''CREATE TABLE IF NOT EXISTS INTERP
         (ROW    INT    NOT NULL,
          COL    INT    NOT NULL,
          FIT    BLOB   NOT NULL,
          UNIQUE(ROW, COL) ON CONFLICT REPLACE);''')

    #get entries for the probability transition matrix without removing states
    PTM, SEM, global_active = getExtractedTransitionProbs(MSMs, K, data_version=data_version)

    #get list of keys
    keys = list(PTM.keys())

    #set a batch size 
    batch_size = 50000
    num_batches = int(len(keys) / batch_size) + 1

    #do the interpolation in parallel, in batches
    for i in range(num_batches):

        #print progress message
        print("Beginning batch {} of {}".format(i+1,num_batches))

        #set the start and end values and get the keys
        a = batch_size * i
        b = batch_size * (i+1)
        batch_keys = keys[a:b]

        #create a generator for the input to the fit function for each worker
        fit_input = ((key, PTM[key], SEM[key], E2, K) for key in batch_keys)

        #start timer
        start = time.time()

        #compute the rbfs for each key - get boolean filter for successful ones
        rbfs = p.starmap(fitEntry, fit_input)
        bool_map = [b[1] for b in rbfs]

        #filter the keys and rbfs according to bool_map
        idxs = [i for (i,v) in zip(batch_keys, bool_map) if v]
        rbfs = [i[0] for (i,v) in zip(rbfs, bool_map) if v]

        #add to the database
        for entry in range(len(idxs)):
            R = idxs[entry][0]
            C = idxs[entry][1]
            I = rbfs[entry]
            pI = pickle.dumps(I, pickle.HIGHEST_PROTOCOL)
            conn.execute('''INSERT INTO INTERP (ROW, COL, FIT) \
                VALUES (?, ?, ?)''',(R,C,sqlite3.Binary(pI)))

        #commit to the database
        conn.commit()

        #end timer
        end = time.time()

        #print elapsed time message
        print("Elapsed time for interp batch {} is {} seconds".format(i,end-start))

    #close the database
    conn.close()

    return


def fitTransitionMatrixFullParallel(num_procs = 6, refine = True, SQL = True, data_version=''):
    #Fit the entries of the transition matrix using all ensembles
    #state space is union of all observed states from each ensemble
    #use a parallel worker pool

    #load the MSMs
    MSMs, K, E2 = loadMSMs(data_version=data_version, refine=refine)

    #determine how we want to store the data
    if SQL:

        #set the database name
        db_name = "msm" + data_version + "/interp"
        if refine:
            db_name += "R"
        db_name += ".db"

        #store the results in a SQL database
        getSQL(db_name, MSMs, K, E2, num_procs, data_version=data_version)

    else:

        #do the interpolation in RAM and get a dict of keys and rbfs
        Pfit = getDict(MSMs, K, E2, num_procs, data_version=data_version)

        #pickle the array of interpolants
        Iname = "msm" + data_version+ "/matrixI"
        if refine:
            Iname += "R"
        Iname += "_new"
        with open(Iname, 'wb') as f:
            pickle.dump(Pfit, f)


if __name__ == "__main__":

    #perform interpolation of the full transition matrix
    #fitTransitionMatrixFull()
    #fitTransitionMatrixFullParallel()

    #use old data
    # fitTransitionMatrixFullParallel(refine = False, data_version="V1")

    #use new data and SQL database
    fitTransitionMatrixFullParallel(refine = True, data_version="", SQL=True)


    #test the conversion of pickle to shelve
    # PfitLoc = "msm/matrixI"
    #pickleToShelve(PfitLoc)
    #pickleToSQL(PfitLoc)



