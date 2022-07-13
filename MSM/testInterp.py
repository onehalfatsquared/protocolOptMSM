'''
Code to test the interpolation of of transition matrices resulting from my custom
MSM class. 

Goal is to provide unit tests for functions in constructing the matrix
of values to be interpolated.

Also tests for the interpolants themselves, by looking at particular examples and examing
the resulting interpolants. 

Test how the interpolation of missing entries works out

Note: some of these may be out of date after some refactoring of the code, as of 
March 7, 2022. May return to this at a later date, but everything works as intended
as far as I can tell during the dev process.
'''

import sys
import os
import fnmatch
import inspect

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse
from scipy import interpolate
from collections import defaultdict
from collections import Counter

import pickle
import timeit
import time

import rbf
from rbf.interpolate import RBFInterpolant

import manualMSM
from manualMSM import MSM
from manualMSM import loadStateDict

import interpolateMSM
from interpolateMSM import *

from interpolantEval import *

from sparseTesting import highestFluxPathways
from sparseTesting import computeTPTflux

os.environ['OPENBLAS_NUM_THREADS'] = '1'

def computeAllProbability(P, T, initial, num_states):
    #compute the probability of each state as a function of time by solving FKE

    #initial condition
    p0 = np.zeros(num_states)
    p0[initial] = 1.0

    #set the probability vector storage for all time
    p = np.zeros((T+1, num_states))
    p[0,:] = p0

    #iteratively multiply my P
    for i in range(T):
        p[i+1,:] = p[i,:] * P

    #set the time discretization
    t = np.linspace(0,T,T+1)
    
    return t, p


def computeTargetProbability(P, T, initial, target, num_states):
    #compute the probability of being in the target as a function of time using P

    #get all the probabilities
    t, p = computeAllProbability(P, T, initial, num_states)

    #return target prob
    return t, p[:,target]

def computeSampledTargetProbability(folder, target_state):
    #compute p(t) estimated using sampled trajectories

    #count number of npy files
    trajs = fnmatch.filter(os.listdir(folder), '*.npy')

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
            p          = np.zeros(frames+1, dtype=float)
            samples    = np.zeros(frames+1, dtype=float)
            samples[0] = 1

        else: #test if later path has more frames
            new_frames = len(path)
            if new_frames > frames:
                p_new = np.zeros(new_frames+1)
                p_new[0:(frames+1)] = p
                p = p_new

                samples_new = np.zeros(new_frames+1)
                samples_new[0:(frames+1)] = samples
                samples = samples_new

                frames = new_frames

        count = 1
        for pair in path:
            if (pair[0] == target_state[0] and pair[1] == target_state[1] and pair[-1] == target_state[2]):
                p[count] += 1
                #break

            samples[count] += 1
            count += 1

    #get time discretization and normalize p
    t = np.linspace(0, 25*frames, frames+1)
    p = p / samples

    print(p)
    print(samples)

    return t, p


def loadTest():
    #test loading the MSMs 

    #test 1 - load all found MSMs
    MSMs, K, E2 = loadMSMs(verbose=False)
    assert(K == 20)

    #test 2 - load only a few MSMs
    doLoad = [1,7,9]
    MSMs, K, E2 = loadMSMs(doLoad,False)
    E2true = np.zeros(3, dtype=object)
    E2true[0] = [1.6, 1.2]
    E2true[1] = [1.6, 1.3]
    E2true[2] = [1.8, 1.3]
    assert((E2 == E2true).all())

    print("MSM load tests passed")

    return 0


def getPTMspeed():
    #test the speed of constructing the list of transition matrix elements across ensembles

    #get the MSMs
    MSMs, K, E2 = loadMSMs(verbose=False)

    #get entries for the probability transition matrix without removing states
    print("Gathering all non-zero transition matrix entries")

    #time it
    start = time.time()
    PTM, SEM, global_active = getFullPTMs(MSMs, K)
    end = time.time()
    print("Elapsed time is {} seconds".format(end-start))

    #pickle the resulting PTM to test on without reconstructing
    with open("data/ptm_ex_full", 'wb') as f:
        pickle.dump(PTM, f)

    #pickle the resulting PTM to test on without reconstructing
    with open("data/sem_ex_full", 'wb') as f:
        pickle.dump(SEM, f)

    #result notes: using 4 ensembles, it took 130 seconds. 


    return 0

def testPTMentries():
    #test that the entries in the list of the PTM matrix are consistent with 
    #indiviudal MSMs

    #load the example sparse matrix of transition probability elements
    with open("data/ptm_ex_full", 'rb') as f:
        PTM = pickle.load(f)

    #load the MSMs
    MSMs, K, E2 = loadMSMs()

    #get the list of global active states
    global_active = getGlobalActive(MSMs, K)
    print(global_active)

    #test 1 - common transition 0 -> 0
    pList = PTM[(0,0)]
    testList = []
    for k in range(K):
        testList.append(MSMs[k].P_active[0,0])
    assert(pList==testList)

    #test 2 - state with different mappings in each ensemble
    keys = list(PTM.keys())
    for idx in range(len(keys)):
        key = keys[idx]
        global_key = (global_active[key[0]], global_active[key[1]])
        pList = PTM[key]
        testList = []
        for k in range(K):
            i = getEnsembleIndex(global_key[0], MSMs, k)
            j = getEnsembleIndex(global_key[1], MSMs, k)
            if (i == -1 or j == -1):
                prob = []
            else:
                prob = MSMs[k].P_active[(i,j)]
                if prob < 1e-9:
                    prob = []
            testList.append(prob)
        assert(pList==testList)
        if idx % 1000 == 0:
            print("Testing key {} of {}".format(idx+1, len(keys)))


    print("All tests passed")

    return 0

def plotInterpolant(x, y, z, x_itp, u_itp):
    #plot a general interpolant and scatter plot

    plt.figure()
    vm = np.min(z)
    vM = np.max(z)
    plt.tripcolor(x_itp[:,0], x_itp[:,1], u_itp, vmin=vm, vmax=vM, cmap='viridis', alpha=0.9)
    plt.scatter(x, y, s=50, c=z, vmin=vm, vmax=vM,cmap='viridis', edgecolor='k', alpha=1)
    plt.xlabel("P-H Bond Strength")
    plt.ylabel("H-H Bond Strength")
    plt.colorbar()
    plt.show()

    return

def plotTransitionProbInterpolant(rbf, state1, state2, global_active, inv_map):
    #plot the given rbf interpolant of the transition probability between state 1 and 2

    #generate 2d grid from 1 to 2 in each dimension
    x1, x2 = np.linspace(1.2, 2, 100), np.linspace(1.1, 1.6, 100)
    x_itp = np.reshape(np.meshgrid(x1, x2), (2, 100*100)).T

    #compute the interpolant on the grid
    interp = rbf
    u_itp = interp(x_itp)
    
    #get the centers and known values
    z = interp.d
    x = interp.y + interp.center

    #separate out the parameters into each dimension
    y = [x[i][1] for i in range(len(x))]
    x = [x[i][0] for i in range(len(x))]
    
    #get and print info on what transition is being plotted
    s = interp.sig
    s1 = global_active[state1]
    s2 = global_active[state2]
    print("Plotting P[{}][{}]. Transition {} -> {}".format(state1,state2,inv_map[s1],inv_map[s2]))
    for i in range(len(x)):
         print((x[i],y[i]), z[i], s[i])
    
    #plot it
    plotInterpolant(x, y, z, x_itp, u_itp)

    return 0


    
def interpTargetProb():
    #get probability of target for each MSM and interpolate

    #load the MSMs
    MSMs, K, E2 = loadMSMs()

    #get global active set
    global_active = getGlobalActive(MSMs, K)

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial and target states
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,30,30)]

    #set params
    animate_time = 25
    lag = 100

    #init target prob storage at final time
    probs = []

    #compute the target prob for each MSM
    for k in range(K):
        num_steps = int(500000.0 / (animate_time*lag))
        t, p = MSMs[k].solveForwardEquationActive(initial_index, num_steps)
        true_target = MSMs[k].fullToNew[target_index]
        if true_target == -1:
            probs.append(0)
        else:
            probs.append(p[-1,true_target])

    x = [v[0] for v in E2]
    y = [v[1] for v in E2]
    sig = [0 for i in range(len(x))]
    # sig[0] = 0.01
    # sig[10] = 0.001
    # sig[13] = 0.01
    # sig[21] = 0.01
    # sig[6] = 0.01
    # sig[20] = 0.01
    # sig[12] = 0.01
    # sig[5] = 0.01
    # sig[7] = 0.001

    for i in range(len(x)):
        print(i,x[i],y[i],sig[i])

    # for i in range(len(E2)):
    #     probs[i] = 0
    #     sig[i]   = 0.001

    # probs[0]  = 0.33
    # sig[0]    = 0.27/500.0
    
    #do an rbf interpolation
    rbf = getFullRBF(E2, probs, sig, K)

    #generate 2d grid from 1 to 2 in each dimension
    x1, x2 = np.linspace(1.2, 2, 100), np.linspace(1.1, 1.6, 100)
    x_itp = np.reshape(np.meshgrid(x1, x2), (2, 100*100)).T

    #compute the interpolant on the grid
    interp = rbf[0]
    u_itp = interp(x_itp)
    
    #get the centers and known values
    z = interp.d
    x = interp.y + interp.center

    #separate out the parameters into each dimension
    y = [x[i][1] for i in range(len(x))]
    x = [x[i][0] for i in range(len(x))]

    #plot 
    plotInterpolant(x, y, z, x_itp, u_itp)

    return 0



def testMSMinterpolant():
    #test of the interpolated transition matrix repreoduces the MSM fc curve

    #load the MSMs
    MSMs, K, E2 = loadMSMs()

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial and target states
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,30,30)]

    #get the target indices in the global index
    global_i = np.where(global_active == initial_index)[0][0]
    global_t = np.where(global_active == target_index )[0][0]

    #set params
    animate_time = 25
    lag = 100
    num_steps = int(500000.0 / (animate_time*lag))

    #test case - P1.6H1.4, input 13
    
    #find the index of the desired MSM
    for idx in range(len(E2)):
        param = E2[idx]

        if param == [1.4,1.45]:
            msm_index = idx
            break

    #get the target probability for this MSM
    t, p = MSMs[msm_index].solveForwardEquationActive(initial_index, num_steps)
    true_target = MSMs[msm_index].fullToNew[target_index]
    t = t * animate_time
    t_prob = (p[:,true_target])


    #get the interpolated transition matrix
    with open("msm/matrixI", "rb") as f:
        matrixI   = pickle.load(f)
        print("Matrix of interpolants loaded")

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(matrixI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    global work_list
    work_list = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        j_arrays = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key[1])
            j_arrays.append(matrixI[key])

        work_list[i] = [j_ids, j_arrays] 

    print('Created global array of jobs for parallel workers')

    #delete matrixI
    del matrixI
    print('Deleted original matrix')

    #evaluate it at the parameter value
    start = time.time()
    I = eval_matrix_sparsePar([1.4,1.45], nsa, nsg, 6)
    end = time.time()

    print("Evalution time is {} seconds".format(end-start))

    print(global_i, global_t)
    print(inv_map[global_active[global_t]])
    print(I.getrow(global_t))

    #evaluate the target probabilities
    tI, pI = computeTargetProbability(I, int(500000 / (lag*animate_time)), 
                                          global_i, global_t, nsg)
    tI *= lag * animate_time

    #compute target prob estimate from sampling
    target_state = (12,30,30)
    folder = "../trajectories/P1.4H1.45/"
    ts, ps = computeSampledTargetProbability(folder, target_state)


    #plot stuff
    plt.figure()
    plt.plot(t, t_prob, tI, pI, ts, ps)
    plt.xlabel("Time")
    plt.ylabel("Target State Probability")
    plt.legend(["MSM", "Interpolant", "Sampling"])
    plt.show()


    return 0


def computeRowPar(row_idx, E):
    #create the 3 arrays that define the transition matrix row

    #make lists to store non-zero columns as well the probability in that column
    j_list = work_list[row_idx]
    thisRowData = []
    nonZeroCols = j_list[0]

    j_array = j_list[1]

    #loop over non-zero columns in the keys
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-12:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            # print(j, nonZeroCols)
            # print(nonZeroCols[j])
            del nonZeroCols[j]
            continue

    
    #perform a re-normalization to ensure rows sum to 1
    npRow = np.array(thisRowData, dtype=float)
    S     = np.sum(npRow)
    if (S > 1e-6):
        npRow /= S

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    return [r, nonZeroCols, npRow.tolist()]

def eval_matrix_sparsePar(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E) for i in range(nsa))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowPar, rowInput)

    #close pool
    p.close()
    p.join()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    
    #flatten all the lists
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    print(len(row), len(col), len(dat))
    print(max(row), max(col), max(dat))
    print(nsg)

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    return imat

def computeRowDPar(row_idx, E):
    #create the 3 arrays that define the transition matrix row

    #make lists to store non-zero columns as well the probability in that column
    j_list = work_list[row_idx]
    thisRowData = []
    nonZeroCols = j_list[0]

    j_array = j_list[1]

    #make a storage array for the derivatives
    dims = len(E)
    d = [[] for i in range(dims)]

    #loop over non-zero columns in the keys - backwards in case of deletion
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-12:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            # print(j, nonZeroCols)
            # print(nonZeroCols[j])
            del nonZeroCols[j]
            continue

        #compute each derivative and append it
        for dim in range(dims):
            #get a tuple of the proper dimension to diff
            diff = np.zeros(dims, dtype=int)
            diff[dim] = 1

            deriv = I([E],diff)[0]
            d[dim].insert(0,deriv)

    
    #perform a re-normalization to ensure rows sum to 1
    npRow   = np.array(thisRowData, dtype=float)
    npDRows = np.array(d, dtype=float)
    S       = np.sum(npRow)
    D       = np.sum(npDRows,1)

    if (S > 1e-12):
        npRow /= S
        for i in range(dims):
            npDRows[i] = (npDRows[i] - npRow * D[i]) / (S)

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    # if row_idx == 3653:
        # print(npRow)
        # print(npDRows)
        # print(S)
        # print(D)
        # print(npDRows[0])
        # sys.exit()

    return [r, nonZeroCols, npRow.tolist(), npDRows.tolist()]

def eval_matrix_deriv_sparsePar(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  
    #also evaluate the derivative

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E) for i in range(nsa))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowDPar, rowInput)

    #close pool
    p.close()
    p.join()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    der = [item[3] for item in results]
    
    #flatten all the lists except the derivatives list
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    #handle derivatives separately
    derivs = []
    for i in range(len(E)):
        deriv = [item for sublist in der for item in sublist[i]]
        derivs.append(scipy.sparse.coo_matrix((deriv, (row, col)), shape = (nsg,nsg)).tocsr())

    #return the transition matrix and its partial derivatives
    return imat, derivs

def testMSMinterpolantPar():
    #test the accuracy and efficiency of parallel interpolant eval against serial

    #get the interpolated transition matrix
    with open("msm/matrixI_test0", "rb") as f:
        matrixI   = pickle.load(f)

    print('loaded matrix')

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(matrixI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    global work_list
    work_list = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        j_arrays = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key[1])
            j_arrays.append(matrixI[key])

        work_list[i] = [j_ids, j_arrays] 

    print('created work list')

    # #test just evaluating interpolants
    # start = time.time()
    # eval_rbfParG([1.35,1.5], 4)
    # end = time.time()

    # print("Par evalution only time is {} seconds".format(end-start))

    #evaluate it at the parameter value
    # start = time.time()
    # I = eval_matrix_sparse(matrixI, [1.35,1.5], 3714)
    # end = time.time()

    # print("Serial evalution time is {} seconds".format(end-start))
    # print(I.getrow(782))

    # #delete matrixI
    del matrixI

    #evaluate it at the parameter value - parallel
    print('beginning eval')
    start = time.time()
    I = eval_matrix_sparsePar([1.35,1.5], nsa, 3714, 6)
    end = time.time()

    print("Parallel evalution time is {} seconds".format(end-start))
    print(I.getrow(782))


    return 0


def testMSMinterpolantBetweenNodes():

    #load the MSMs
    MSMs, K, E2 = loadMSMs()

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial and target states
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,30,30)]

    #get the target indices in the global index
    global_i = np.where(global_active == initial_index)[0][0]
    global_t = np.where(global_active == target_index )[0][0]

    #set params
    animate_time = 25
    lag = 100
    num_steps = int(500000.0 / (animate_time*lag))

    #get the interpolated transition matrix
    with open("msm/matrixI", "rb") as f:
        matrixI   = pickle.load(f)
        print("Matrix of interpolants loaded")

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(matrixI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    global work_list
    work_list = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        j_arrays = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key[1])
            j_arrays.append(matrixI[key])

        work_list[i] = [j_ids, j_arrays] 

    print('Created global array of jobs for parallel workers')
    print('Deleting original matrix')

    #delete matrixI
    del matrixI

    #set up the intermediate points to test at
    initial_params = [1.5,1.5]
    update_dir = [0,-1]
    test_params = []
    num_steps   = 8
    step_size   = 0.1 / (num_steps-1)
    step_size   = 0.0075
    for i in range(num_steps):
        new_params = [0,0]
        new_params[0] = initial_params[0] + i * step_size * update_dir[0]
        new_params[1] = initial_params[1] + i * step_size * update_dir[1]
        test_params.append(new_params)

    #print the test parameters
    print("The parameters to be tested are:")
    print(test_params)

    #evaluate the interpolants at each test point, get fc for target
    plt.figure()
    labels = []
    for i in range(num_steps):
        print("Evaluating interpolant {} of {}".format(i+1,num_steps))
        I = eval_matrix_sparsePar(test_params[i], nsa, nsg, 4)

        #evaluate the target probabilities
        tI, pI = computeTargetProbability(I, int(500000 / (lag*animate_time)), 
                                              global_i, global_t, nsg)
        tI *= lag * animate_time

        plt.plot(tI, pI)
        labels.append('P'+str(test_params[i][0])+"H"+str(test_params[i][1]))


    #label plots
    plt.xlabel("Time")
    plt.ylabel("Target State Probability")
    plt.legend(labels)
    plt.show()




    return 0



def testDerivativeMatrix():
    #test the matrix of partial derivatives against finite differences

    #load the MSMs
    MSMs, K, E2 = loadMSMs()

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the interpolated transition matrix
    with open("msm/matrixI_test1_2", "rb") as f:
        matrixI   = pickle.load(f)

    print('loaded matrix')

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(matrixI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    global work_list
    work_list = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        j_arrays = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key[1])
            j_arrays.append(matrixI[key])

        work_list[i] = [j_ids, j_arrays] 

    print('created work list')

    # #delete matrixI
    del matrixI

    #evaluate it at the parameter value - parallel
    start = time.time()
    I0, D = eval_matrix_deriv_sparsePar([1.64,1.4], nsa, nsg, 6)
    end = time.time()

    print("Parallel evalution time is {} seconds".format(end-start))

    #evaluate it at the parameter value + h
    h = 1e-4
    start = time.time()
    I1 = eval_matrix_sparsePar([1.64+h,1.4], nsa, nsg, 6)
    end = time.time()

    print("Parallel evalution time is {} seconds".format(end-start))

    #loop over keys to check deviations
    bad = 0
    total = 0
    for row,col in zip(*I0.nonzero()):
        fd = (I1[row,col] - I0[row,col] ) / h
        d  = D[0][row,col]

        #get difference for all values greater than 1e-5
        diff = np.abs(fd-d)
        rel_e = diff / np.abs(d)
        total += 1
        if rel_e > 1e-1:
            bad += 1
            #print("({},{}). Matrix: {}, FD: {}".format(row,col,d, fd))


    print("{} bad approximations of {}".format(bad,total))

    #save the values to gauge memory use
    td = dict()
    td['test_key'] = [I0,D]
    with open("data/interp_test", 'wb') as f:
        pickle.dump(td, f)

    return 0


def denseMult(mats, num_steps, num_states):

    p = np.zeros((num_steps+1, num_states))
    p[0, 0] = 1

    #solve the forward equation by applying transition matrix T times
    for i in range(num_steps):
        p[i+1,:] = np.dot(p[i,:], mats[i].toarray())

    return p

def sparseMult(mats, num_steps, num_states):

    p = np.zeros((num_steps+1, num_states))
    p[0, 0] = 1

    #solve the forward equation by applying transition matrix T times
    for i in range(num_steps):
        p[i+1,:] = p[i,:] * mats[i]

    return p

def decompMult(mats, num_steps, num_states):

    p = np.zeros((num_steps+1, num_states))
    p[0, 0] = 1

    k = 10

    #solve the forward equation
    for i in range(num_steps):

        eigsL, vecsL = scipy.sparse.linalg.eigs(mats[i].transpose(), k=k, which="LR")
        eigsR, vecsR = scipy.sparse.linalg.eigs(mats[i], k=k, which="LR")

        #sort them and search
        swaps = np.argsort(np.abs(eigsL))
        eig_sort = eigsL[swaps][::-1]

        #compute the spectral decomp
        for j in range(k):

            #get the eigenvalue 
            ev = eig_sort[j]

            #get the left  and right eigenvectors
            indexL = np.where(np.abs(eigsL - ev) < 1e-6)[0][0]
            indexR = np.where(np.abs(eigsR - ev) < 1e-6)[0][0]
            evL    = vecsL[:,indexL]
            evR    = vecsR[:,indexR]

            #compute inner product between p(0) and right evect. scale by evL * evR
            inner = np.dot(p[i,:], evR)
            scale = np.dot(evR,evL)

            #multiply by the left eigenvector and scale
            inner *= evL / scale

            #add in the time evolution term, eigenvale to the power t
            # print(p[i+1,:])
            # print(np.real(np.outer(np.power(ev,25), inner)))
            p[i+1,:] += np.real(np.outer(np.power(ev,25), inner))[0]

        

    return p


def testMVmult():
    #test the speed and accuracy of three matrix vector multiply techniques

    #load the MSMs
    MSMs, K, E2 = loadMSMs()

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the interpolated transition matrix
    with open("msm/matrixI_test1_2", "rb") as f:
        matrixI   = pickle.load(f)

    print('loaded matrix')

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(matrixI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    global work_list
    work_list = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        j_arrays = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key[1])
            j_arrays.append(matrixI[key])

        work_list[i] = [j_ids, j_arrays] 

    print('created work list')

    # #delete matrixI
    del matrixI

    #evaluate it at the parameter value - parallel
    start = time.time()
    I0, D = eval_matrix_deriv_sparsePar([1.64,1.4], nsa, nsg, 6)
    end = time.time()

    num_steps = 100
    mat_list = []
    for i in range(num_steps):
        mat_list.append(I0)

    #perform each of 3 methods
    start = time.time()
    p0 = denseMult(mat_list, num_steps, nsg)
    end = time.time()
    print("Dense time is {}".format(end-start))

    start = time.time()
    p1 = sparseMult(mat_list, num_steps, nsg)
    end = time.time()
    print("Sparse time is {}".format(end-start))

    start = time.time()
    p2 = decompMult(mat_list, num_steps, nsg)
    end = time.time()
    print("Decomp time is {}".format(end-start))


    #plot the evolution to compare computed values
    plt.figure()
    plt.plot(p0[:,887])
    plt.plot(p1[:,887])
    plt.plot(p2[:,887])
    plt.show()

    print(p0[:,887])
    print(p1[:,887])
    print(p2[:,887])



def testShelveCaching():
    #test the relative speed of pickle and shelve caching for the interpolants

    # #time the set up phase
    # print("Beginning pickle set up phase")
    # start = time.time()
    # #get the interpolated transition matrix
    # with open("msm/matrixI", "rb") as f:
    #     matrixI   = pickle.load(f)

    # #get a list of all the keys in the dict, sorted
    # all_keys = sorted(matrixI.keys())

    # #get a list of the first index of each key
    # id1 = [t[0] for t in all_keys]

    # #count how many nonzero entries each row index has, and the number of active states
    # nz_counts = Counter(id1)
    # nsa = len(nz_counts)
    # print(nsa)

    # #create a list of offsets to get to a particular state
    # offsets = np.zeros(nsa+1,dtype=int)
    # for i in range(1,nsa):
    #     offsets[i] = offsets[i-1] + nz_counts[i-1]

    # #for each i, precompute all the work that needs to be done
    # global work_list
    # work_list = np.zeros(nsa, dtype=object)
    # for i in range(nsa):
    #     num_non_zeros_i = nz_counts[i]
    #     j_ids    = []
    #     j_arrays = []
    #     for j in range(num_non_zeros_i):
    #         key = all_keys[offsets[i]+j]
    #         j_ids.append(key[1])
    #         j_arrays.append(matrixI[key])

    #     work_list[i] = [j_ids, j_arrays] 

    # print('created work list')

    # # #delete matrixI
    # del matrixI

    # #end start up time
    # end = time.time()
    # print("Pickle start up phase is {} seconds".format(end-start))

    # #set up some test values 
    # E_test = [[1.4,1.4], [1.3,1.5], [1.6,1.5], [1.54,1.65], [1.5,1.5]]

    # #evaluate matrices for the 5 tests and get average time - pickle
    # print('beginning eval time test')
    # start = time.time()
    # for i in range(len(E_test)):
    #     I = eval_matrix_sparsePar(E_test[i], 3801, 3801, 6)
    # end = time.time()
    # print(I)
    # print("Average eval time is {} seconds for pickle version".format(0.2*(end-start)))

    #evaluate matrices for the 5 tests and get average time - cache
    print('beginning eval time test')
    E_test = [[1.4,1.4],[1.5,1.5],[1.3,1.5],[1.5,1.3],[1.76,1.56],[1.35,1.45],[1.8,1.5]]
    start = time.time()
    for i in range(len(E_test)):
        I1 = eval_matrix_sparsePar_cache(E_test[i], 3801, 3801, 6)
    end = time.time()
    c = 1.0/float(len(E_test))
    print("Average eval time is {} seconds for cache version".format(c*(end-start)))
    print(I1)
    #print(I.getrow(782))
    #print(I1.getrow(782))

def testSQLcaching():
    #do a speed test on using an SQL database to store interpolants

    print('beginning eval time test')
    E_test = [[1.4,1.4],[1.5,1.5],[1.3,1.5],[1.5,1.3],[1.76,1.56],[1.35,1.45],[1.8,1.5]]
    E_test = [[1.2,1.4]]
    start = time.time()
    for i in range(len(E_test)):
        I1 = eval_matrix_sparsePar_SQL(E_test[i], 5726, 5726, 6)
    end = time.time()
    c = 1.0/float(len(E_test))
    print("Average eval time is {} seconds for cache version".format(c*(end-start)))
    print(I1)


def verifySQL():
    #test if the SQL database is accurate

    conn = sqlite3.connect("msmV1/interp.db")

    cursor = conn.execute("SELECT ROW,COL FROM INTERP WHERE ROW=0")
    for item in cursor.fetchall():
        print(item[0], int.from_bytes(item[1],sys.byteorder))

def plotMajorDifferences(t, p1, p2, I1, I2, num_states, stateDict, inv_map, global_active):
    #determine which yeld curves have major differences and plot them

    #get the difference between distributions
    D = np.abs(p1-p2)

    #loop over p2 to see which states have sufficient yields at any time
    visited = []
    for i in range(num_states):
        p = p2[:,i]
        if any(p > 0.01):
            visited.append(i)


    for i in range(num_states):
        stateProbs = p2[:,i]
        M_ind = np.argmax(stateProbs)
        M = stateProbs[M_ind]
        if M > 0.01:
            print(i, inv_map[global_active[i]], M, M_ind)
            print(I1[i,:])
            print()
            print(I2[i,:])

    #find the states with large differences - states are columns
    threshold = 0.001
    beyondThresh = (D > threshold)
    beyondThreshS = np.sum(beyondThresh, axis=0)
    beyondThreshStates = np.where(beyondThreshS > 0)[0]

    #find the state in the flux list
    test_states = []

    ex_index = stateDict[(11,22,13)]
    global_test = np.where(global_active == ex_index)[0][0]
    # test_states.append(global_test)

    ex_index = stateDict[(9,16,8)]
    global_test = np.where(global_active == ex_index)[0][0]
    # test_states.append(global_test)

    # ex_index = stateDict[(12,30,60,60,26)]
    # ex_index = stateDict[(12,29,60,57,27)]
    # ex_index = stateDict[(12,31,29)]
    ex_index = stateDict[(12,30,30)]
    global_test = np.where(global_active == ex_index)[0][0]
    test_states.append(global_test)

    # for state in beyondThreshStates:
    for state in test_states:
    # for state in visited:
        gState = global_active[state]
        fullState = inv_map[gState]
        print(state, fullState)
        plt.plot(t, p1[:,state], t, p2[:,state])
        plt.show()


    return 

def determineThreshold(M, global_active, inv_map, stateDict):

    #init storage for max
    max_prob = 0
    max_state = 0

    #loop over first 10 rows to determine threshold
    for i1 in range(1000):
        initial_state = inv_map[global_active[i1]]
        if initial_state[0] + initial_state[1] <8:
            print("Checking row {}, which is state {}".format(i1,initial_state))

            #get the nonzero entries and loop over
            rows,cols = M.getrow(i1).nonzero()
            for col in cols:

                #get the state descriptor
                state = inv_map[global_active[col]]

                #check for sufficient progress 
                if state[0] + state[1] > 8:
                    val = M[i1,col]
                    if val > max_prob:
                        max_prob = val
                        max_state = state
                        print("New Max: State {}, Prob {}".format(state, val))



def compareDynamics():
    #compare dynamics at node with those slightly away from it

    #load the MSMs
    MSMs, K, E2 = loadMSMs(refine=True)
    lag = 125
    animate_time = 25

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the state mapping from pickle
    stateDict, dummy = loadStateDict(refine=True)
    nsa = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial and target states
    # initial_index = stateDict[(12,31,29)]
    initial_index = stateDict[(0,0,0)]
    # initial_index = stateDict[(12,30,60,60,26)]
    # initial_index = stateDict[(12,29,60,57,27)]
    target_index  = stateDict[(12,30,30)]
    # target_index = initial_index

    #get the target indices in the global index
    global_i = np.where(global_active == initial_index)[0][0]
    global_t = np.where(global_active == target_index )[0][0]

    #load the transition matrices
    with open("msm_tests/testP1.3H1.5", 'rb') as f:
        I1 =  pickle.load(f)

    with open("msm_tests/testP1.305H1.5", 'rb') as f:
        I2 = pickle.load(f)

    print("Loaded transition matrices")
    print(I1)
    print(I2[5000:,:])
    print(nsa,nsg)

    i1 = np.where(global_active == stateDict[(2,0,0)])[0][0]
    i2 = np.where(global_active == stateDict[(9,26,15)])[0][0]
    print(I2[i1,i2])
    i1 = np.where(global_active == stateDict[(0,1,0)])[0][0]
    i2 = np.where(global_active == stateDict[(8,23,15)])[0][0]
    print(I2[i1,i2])
    i1 = np.where(global_active == stateDict[(2,1,0)])[0][0]
    i2 = np.where(global_active == stateDict[(7,25,11)])[0][0]
    print(I2[i1,i2])
    # i1 = np.where(global_active == stateDict[(1,1,0)])[0][0]
    # i2 = np.where(global_active == stateDict[(9,26,13)])[0][0]
    # print(I2[i1,i2])

    # determineThreshold(I2, global_active, inv_map, stateDict)
    # sys.exit()

    #get eq from solving forward equation for long time
    # t, p = computeAllProbability(I2, int(8000000 / (lag*animate_time)), global_i, nsg)
    # eq = p[-1,:]
    # f = computeTPTflux(I2, eq, global_i, global_t, nsg)
    # highestFluxPathways(f, global_i, global_t, nsg, inv_map, global_active)
    # sys.exit()
        
    # I2[:,1461] = 0
    # I2[44,1461] = 1.0
    # I2[:,1612] = 0
    # I2[47,1612] = 1.0
    # I2[9,678] = 0
    # I2[37,1531] = 0
    # I2[40,494] = 0
    # I2[41,1531] = 0
    # I2[44,494] = 0
    # I2[169,910] = 0
    # I2[164,910] = 0
    # I2[64,3669] = 0
    # I2[98,3669] = 0
    # I2[259,3669] = 0
    # I2[214,2669] = 0


    # # I2[:,910] = 0
    # # I2[:,678] = 0

    # I2[:,1008] = 0
    # I2[:,360] = 0

    # rows, cols = I1.nonzero()
    # for row,col in zip(rows,cols):
    #     if I1[row,col] < 3e-4: #2.1e-3
    #         I1[row,col] = 0

    # for i in range(nsg):
    #     S = np.sum(I1[i,:]) 
    #     if (S > 0):
    #         I1[i,:] = I1[i,:] / S

    # rows, cols = I2.nonzero()
    # for row,col in zip(rows,cols):
    #     if I2[row,col] < 8e-4:   #7e-3
    #         I2[row,col] = 0

    # for i in range(nsg):
    #     S = np.sum(I2[i,:]) 
    #     if (S > 0):
    #         I2[i,:] = I2[i,:] / S

    t1, p1 = computeAllProbability(I1, int(800000 / (lag*animate_time)), 
                                          global_i, nsg)
    t1 *= lag * animate_time

    t2, p2 = computeAllProbability(I2, int(800000 / (lag*animate_time)), 
                                          global_i, nsg)
    t2 *= lag * animate_time

    plotMajorDifferences(t1, p1, p2, I1, I2, nsg, stateDict, inv_map, global_active)

def testSQLaccuracy():

    #load the MSMs
    MSMs, K, E2 = loadMSMs(refine=True)

    #set the SQL DB location
    setSQLdb(refine=True)

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the state mapping from pickle
    stateDict, dummy = loadStateDict(refine=True)
    nsa = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial and target states
    # initial_index = stateDict[(12,31,29)]
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,30,30)]

    #get the target indices in the global index
    global_i = np.where(global_active == initial_index)[0][0]
    global_t = np.where(global_active == target_index )[0][0]

    #get a matrix at the node 
    E_test = [[1.3,1.5]]
    start = time.time()
    for i in range(len(E_test)):
        I1, dummy = eval_matrix_deriv_sparsePar_SQL(E_test[i], nsg, nsg, 3)
    end = time.time()
    c = 1.0/float(len(E_test))
    print("Average eval time is {} seconds for cache version".format(c*(end-start)))

    # #get a matrix slightly away to compare
    E_test = [[1.305,1.5]]
    start = time.time()
    for i in range(len(E_test)):
        I2, dummy = eval_matrix_deriv_sparsePar_SQL(E_test[i], nsg, nsg, 4)
    end = time.time()
    c = 1.0/float(len(E_test))
    print("Average eval time is {} seconds for cache version".format(c*(end-start)))

    # #save these matrices for easier access later
    # with open("msm_tests/testP1.3H1.5", 'wb') as f:
    #     pickle.dump(I1, f)

    with open("msm_tests/testP1.305H1.5", 'wb') as f:
        pickle.dump(I2, f)


    # #compare the matrices
    NZR, NZC = I2.nonzero()
    for i in range(len(NZR)):
        R = NZR[i]
        C = NZC[i]
        p1 = I1[R, C]
        p2 = I2[R, C]

        # if np.abs(p1-p2) > 0.01:
        #     print("Row = {}, Col = {}, p1 = {}, p2 = {}".format(R,C,p1,p2))

    lag = 125
    animate_time = 25

    t1, p1 = computeAllProbability(I1, int(800000 / (lag*animate_time)), 
                                          global_i, nsg)
    t1 *= lag * animate_time

    t2, p2 = computeAllProbability(I2, int(800000 / (lag*animate_time)), 
                                          global_i, nsg)
    t2 *= lag * animate_time

    #compute target prob estimate from sampling
    # target_state = (12,30,30)
    # # folder = "../trajectories/P1.2H1.4/state12_31_29/"
    # folder = "../trajectories/P1.4H1.4/"
    # ts, ps = computeSampledTargetProbability(folder, target_state)
    # ts, ps = [0], [0]


    #plot stuff
    # plt.figure()
    # plt.plot(tI, pI, ts, ps)
    # plt.xlabel("Time")
    # plt.ylabel("Target State Probability")
    # plt.legend(["Interpolant", "Sampling"])
    # plt.show()

def getTestKeys(PTM):
    #get a list of test keys to try out

    #get all keys
    keys = list(PTM.keys())
    # return keys

    #get a subset via some criteria
    interesting = []
    for key in keys:

        #get the transition probabilities
        p = PTM[key]

        #get the number of empty lists
        empty = 0
        vals = []
        for entry in p:
            if entry == []:
                empty += 1
            else:
                vals.append(entry)

        if empty > 1:
            continue

        if any(y > 0.15 for y in vals):
            m = min(vals)
            M = max(vals)
            if (M-m > 0.05):
                interesting.append(key)

    return interesting






def testSingleInterp():
    #perform interpolation over individual entries of the matrix and observe the fit

    #load the example sparse matrix of transition probability elements
    with open("data/ptm_ex", 'rb') as f:
        PTM = pickle.load(f)

    #load sparse matrix of standard errors
    with open("data/sem_ex", 'rb') as f:
        SEM = pickle.load(f)

    #load the MSMs
    MSMs, K, E2 = loadMSMs(refine=True)

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the state mapping from pickle
    stateDict, dummy = loadStateDict(refine=True)
    nsa = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    # for i in range(nsg):
    #     print(i, PTM[(14,global_active[i])])

    #get interesting keys to test out
    keys = getTestKeys(PTM)
    # keys = [(global_active[51],global_active[69])]
    for key in keys:
    
        i1 = key[0]
        i2 = key[1]

        #construct the interpolant
        rbf = fitEntry(key, PTM[key], SEM[key], E2, K)[0]

        #plot the interpolant
        plotTransitionProbInterpolant(rbf, i1, i2, global_active, inv_map)



    return 0

def testCK():
    #do a test on the CK diagonal prob decay

    MSM, K, E2 = loadMSMs(refine=True)
    test = MSM[0]

    #get the state mapping from pickle
    stateDict, dummy = loadStateDict(refine=True)
    nsa = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #set initial and target states
    # initial_index = stateDict[(12,31,29)]
    initial_index = stateDict[(0,0,0)]
    target_index  = stateDict[(12,31,27)]

    matrix = test.P

    # for i in range(1000):
    #     print(i, matrix[i,i])

    prob = [1]
    for m in range(10):

        prob.append(matrix[301,301])
        matrix = matrix * test.P


    plt.plot(prob)
    plt.show()

    return

def searchDict():

    MSM, K, E2 = loadMSMs(refine=True)
    test = MSM[11]

    stateDict, dummy = loadStateDict(refine=True)
    keys = list(stateDict.keys())

    initial = stateDict[(0,0,0)]
    t, p = test.solveForwardEquation(initial, 200)

    
    for key in keys:
        if key[0] == 12 and key[1] == 30 and key[-1] == 26:
            index = stateDict[key]
            print(key, p[:,index])
            print(test.P.getrow(index))


    return 





if __name__ == "__main__":

    #do stuff

    #loadTest()

    #getPTMspeed()

    #testPTMentries()

    # testSingleInterp()

    #interpTargetProb()

    #testMSMinterpolant()

    #testMSMinterpolantPar()

    #testMSMinterpolantBetweenNodes()

    #testDerivativeMatrix()

    #testMVmult()

    #testShelveCaching()

    # testSQLcaching()

    # verifySQL()

    # testSQLaccuracy()

    #compareDynamics()

    # testCK()

    searchDict()