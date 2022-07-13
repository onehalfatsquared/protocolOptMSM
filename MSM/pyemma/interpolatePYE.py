import sys
import os
import fnmatch
import inspect

#set a path to the simulation folder to do relative importing of database class in ratchet
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir) 
assemblydir = parentdir + "/simulation/"
sys.path.insert(0, assemblydir) 

# from ratchet import State
# from ratchet import Database
# from ratchet import obs_to_distribution
# from ratchet import sample_discrete_rv

import matplotlib.pyplot as plt
import numpy as np
import pyemma
import scipy
from scipy import sparse
from collections import defaultdict
from collections import Counter

import pickle
import timeit

import rbf
from rbf.interpolate import RBFInterpolant

import manualMSM
from manualMSM import MSM


'''
This block of code creates a matrix of rbf interpolants for the probability transition
matrix given the results from the MSM in each ensemble. 

This transition matrix acts on the full state space, i.e. the union of states visited 
in each ensemble. It is frequently the case that a particular transition is not observed 
in every ensemble (i.e. those with low probabilities), so we allow empty values for the
interpolation. 

If an empty value occurs between two existing values, the interpolant is constructed as 
normal, and provides a guess of what the value may be. If an empty value occurs near the 
boundary of the ensemble parameters (i.e. no data for E=4 when the allowed set is [4,9])
then we explicitly set a value of 0 for this transition, and all following ensembles, 
until a measurement is reached. For example, if the active ensembles are {6,8}, we fill 
in zeros for 4,5, and 9, but leave 7 to be filled in. 
'''

def getEnsembleIndex(state, a_sets, k):
    #get the index of natural state in the k-th active set
    #return -1 if not present

    index = np.where(a_sets[k] == state)[0]

    if len(index > 0):
        return index[0]
    else:
        return -1


def getFullPTMs(MSMs, a_sets, K, num_states):
    #construct probability transition matrices on the same state space for each ensemble
    #PTM will be a sparse array of lists containing entry in each ensemble
    #also constructs lists of the standard error of each measurement using multinomials

    #define a tolerance to be considered 0
    zero_tol = 1e-7

    #init default dicts for the matrices
    PTM = defaultdict(int)
    SEM = defaultdict(int)


    #compute the union of natural states in the active set
    #also convert and store count matrices as csr
    active_nat = set()
    count_matrices = []
    for k in range(K):
        active_nat = active_nat.union(set(a_sets[k]))
        count_matrices.append(MSMs[k].count_matrix_active.tocsr())
    active_nat = list(active_nat)

    #get the total number of natural active states
    nsa = len(active_nat)

    #loop over each entry of the new array
    for i in range(nsa):

        #print progress message
        if (i % 100 == 0):
            print("Constructing row {} of {} of transition matrix".format(i,nsa))

        #get the i-th natural state
        natural_state_i = active_nat[i]

        #determine the set of states that natural state communicates with
        #this is the union of the nonzero entries of MSMs.P[ns]
        non_zeros = set()
        for k in range(K):
            #get index of natural state in MSM k
            i_ind = getEnsembleIndex(natural_state_i, a_sets, k)

            #if index is non-negative, get nonzero entries of the row
            if i_ind > -1:
                nzk = MSMs[k].P.getrow(i_ind).nonzero()[1]
            
                #get the natural index of the nonzeros
                nzk_nat = a_sets[k][nzk]

                #take union wth previously found non-zero states
                non_zeros = non_zeros.union(set(nzk_nat))


        #loop over states with non zero entries
        for natural_state_j in non_zeros:

            #init an nparray to store values and check for zeros
            probs = np.zeros(K)
            zeroI = np.zeros(K)

            #fill the nparray with values from the MSMs in each ensemble
            for k in range(K):

                #get the ensemble index of natural states i and j
                i_ind = getEnsembleIndex(natural_state_i, a_sets, k)
                j_ind = getEnsembleIndex(natural_state_j, a_sets, k)

                #check if the states are in the active set or not
                if (i_ind == -1 or j_ind == -1): 
                    p = []
                    zeroI[k] = 1
                else:
                    #states are present. get probability and number of samples from row
                    p = MSMs[k].P[i_ind, j_ind]
                    probs[k] = p
                    if (p < zero_tol):
                        zeroI[k] = 1

            #check if there are zero values - if so, replace with empty lists
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

            #place the list in the PTM - need to get active state j
            active_j = np.where(active_nat == natural_state_j)[0][0]
            PTM[(i,active_j)] = probList

        #loop back over and construct the standard error as sqrt(p_i(1-p_i)/N)
        for natural_state_j in non_zeros:

            #get active state j
            active_j = np.where(active_nat == natural_state_j)[0][0]

            #get p and init the standard dev list
            p = PTM[(i,active_j)]
            s = []

            #loop over ensembles and fill with standard error
            for index in range(K):

                #get the ensemble indices
                i_ind = getEnsembleIndex(natural_state_i, a_sets, index)
                j_ind = getEnsembleIndex(natural_state_j, a_sets, index)

                #check if the states are in the active set or not, get counts
                if (i_ind == -1 or j_ind == -1): 
                    c = 0
                else:
                    c = count_matrices[index][i_ind, j_ind]

                #get the k-th element of p and compute standard error
                element = p[index]
                if (type(element) is list or c == 0):
                    #if no observation, make stddev 0
                    s.append(0.0)
                else:
                    s.append(np.sqrt(element*(1.0-element) / float(c)))
            SEM[(i,active_j)] = s


    #return the matrices with probabilities and standard deviations
    return PTM, SEM, active_nat

def findOptimalShape(x, y, basis, sample_points=30, sig=0, verbose=False):
    #determine the optimal shape parameter for the rbf by sampling
    #returns the rbf interpolant

    #set parameters for the optimization
    mag_tol = 8    #base 10 exponent for largest coefficient to consider
    shapeLB = -3   #lower bound for search space (logspace)
    shapeUB = 0.5  #upper bound for ...
    default = 0.02 #default shape parameter value, for when it is unclear what to use

    #construct a set of shape parameters to perform error minimization over
    e_set = np.logspace(shapeLB, shapeUB, sample_points)
    e_set = np.flip(e_set, axis=0)
    errs  = np.ones(sample_points)*1000

    #set up variables for minimum error storage
    min_error = 1000

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
            if (verbose):
                print("The system becomes unstable when eps={}".format(e))
                print("Optimization performed over eps > {}".format(e))
            break
        Kinv = I.inv
        if (verbose):
            print(i, c[0])

        #init accumulator for LOOCV error sum
        S = 0
        #compute the error as sum over basis coefficients - use L1 norm for outliers
        for k in range(num_points):
            e_term = c[k] / Kinv[k][k]
            #S += e_term * e_term
            S += np.abs(e_term) 

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

    #remove noise from the gradient to avoid spurious minima
    noise_tol = 1e-7
    d[np.where(np.abs(d) < noise_tol)[0]] = 0.0

    #search for where derivative changes sign from minus to plus, get local minima
    minima = np.where(np.sign(d[:-1]) > np.sign(d[1:]))[0] + 1
    min_vals = errs[minima]

    #if there are no local minima, use default shape parameter
    if (len(minima) == 0):
        min_index = np.abs(e_set - default).argmin()
        min_error = errs[min_index]
        bestShape = e_set[min_index]
        bestInterp = RBFInterpolant(x, y, sigma = sig, phi=basis, order=-1, eps=bestShape)
        return bestInterp, bestShape, min_error, e_set, errs

    #get smallest local min and compare to global sampling
    min_min  = np.min(min_vals)

    #get the smallest stddev for checking which criteria to use
    smallest_sig = np.min(sig)

    #determine which critical point to use - very ad-hoc
    if len(minima) == 1:
        #choose the only local minumum
        min_index = minima[0]

    elif len(minima) == 2:
        #choose the smaller local minimum - in terms of shape parameter values
        min_index = minima[-1]

    else:
        #if there are forced zeros, choose second smallest shape param
        #this step avoids linearization of data with only a few entries
        if smallest_sig < 1e-6:  
            min_index = minima[-2]

        #if no forced zeros, choose the local min with smallest error value
        else: 
            min_index = minima[np.where(min_vals == min_min)[0][0]]

    #check if the smallest local min is sufficiently larger than the global min
    #if it is, there is likely an error plateau -> choose default shape parameter, 0.02
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
    sample_points = 200

    #do pre-processing - eliminate empty lists from the data
    x = []
    y = []
    s = []
    for i in range(K):
        if (type(y_vals[i]) is list):
            continue
        else:
            x.append(x_vals[i])
            y.append(y_vals[i])
            s.append(sig[i])

    # print(x)
    # print(y)
    # print(s)

    #get the optimal shape interpolator
    I, eps, min_e, e_set, errs = findOptimalShape(x, y, basis, sample_points, s, 
                                                  verbose=False)

    #return interpolant
    return I




def fitTransitionMatrixFull():
    #test fitting the transition matrix from all ensembles

    #set some parameters
    MSMs     = []                          #array of MSMs
    a_sets   = []                          #active set for each ensemble
    msmIndex = []                          #index for the msm

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get number of states found
    num_states = len(stateDict)
    print("{} states have been visited among all simulations".format(num_states))

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get the index to parameter map
    P, H = getParameterMap()

    #define the msm folder, get sorted list of all pickled msm files
    msm_folder = "msmPYE/"
    msm_files = fnmatch.filter(os.listdir(msm_folder), 'msm*')
    msm_files.sort()

    #go through the list of msm files, append relevant quantities to lists
    for msm_file in msm_files:
        try:

            #try to unpickle
            full_path = msm_folder + msm_file
            with open(full_path, 'rb') as f:
                msm = pickle.load(f)

            #if success, append msm, active set, and index label
            MSMs.append(msm)
            a_sets.append(msm.active_set)
            msmIndex.append(int(msm_file.split('msm')[1]))

            #print that this msm ha sbeen found
            print("Opened {}. Active set length is {}".format(msm_file, len(msm.active_set)))
        
        except: #could not unpickle msm, raise error
            print("Could not find {}".format(msm_file))
            raise()

    #get the number of ensembles found
    K = len(MSMs)

    #get the energy parameters for each msm found, using the index to parameter map
    E2 = np.zeros(K, dtype=object)
    for k in range(K):
        index = msmIndex[k]
        p = P[index]
        h = H[index]
        E2[k] =[p,h]


    #get entries for the probability transition matrix without removing states
    print("Gathering all non-zero transition matrix entries")
    PTM, SEM, active_nat = getFullPTMs(MSMs, a_sets, K, num_states)

    #pickle the active_nat list
    with open("global_active", 'wb') as f:
        pickle.dump(active_nat, f)

    #fit each entry of the transition matrix and return an array of RBF objects
    #the defaultdict return should be None, as it will not be called ideally
    Pfit = defaultdict()

    #loop over the keys in the PTM
    print("Constructing interpolants for all non-zero entries")
    num_keys = len(PTM.keys())
    c = 0
    loop = 1
    for key in PTM.keys():

        #get the probabilities and standard deviations
        values = PTM[key]
        stdev  = SEM[key]

        print(key)
        print((loop-1)*c +c)

        #construct the interpolant
        rbf = getFullRBF(E2, values, stdev, K)
        Pfit[key] = rbf
        c += 1

        if int(100*float(c)/num_keys) > 10:
            print("Interpolated {}% of entries".format(10*loop))
            loop += 1
            c = 0



    #pickle the array of interpolators
    with open("matrixI", 'wb') as f:
        pickle.dump(Pfit, f)



def eval_matrix(ptmI, E):
    #evaluate an interpolated probability transition matrix at the given parameter value

    #set a tolerance for being considered 0
    zero_tol = 1e-5

    #get the number of states in the system
    num_states = len(ptmI)

    #init an array for the matrix at E
    imat = np.zeros((num_states, num_states))

    #loop over each entry of the matrix
    for i in range(num_states):
        for j in range(num_states):

            #get the interpolant
            I = ptmI[i][j]

            #check if the interpolant is non-trivial
            y = I.d
            if (np.all(y < zero_tol)):
                imat[i][j] = 0
                continue

            #if non-trivial, compute the inteerpolant and set the value
            prob = I([[E,0]])
            imat[i][j] = prob

        #perform a re-normalization to ensure rows sum to 1
        S = np.sum(imat[i])
        if (S > zero_tol):
            imat[i] /= S

    return imat

def eval_matrix_sparse(ptmI, E, nsg):
    #do the matrix evaluation using sparse storage

    #set a tolerance for being considered 0
    zero_tol = 1e-6

    #init storage for rows, cols, and data lists
    row = []
    col = []
    dat = []

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(ptmI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #init a counter that keeps track of which key block we are analyzing
    current_block = 0

    #loop over each entry of the matrix, get non-zero entries, normalize
    for i in range(nsa):

        #get the number of nonzero entries in this row
        num_non_zeros_i = nz_counts[i]

        #make lists to store non-zero columns as well the probability in that column
        thisRowData = []
        nonZeroCols = []

        #loop over non-zero columns in the keys
        for j in range(num_non_zeros_i):

            #get the current key from the block
            key = all_keys[current_block+j]

            #get the interpolant
            I = ptmI[key]

            #check if the interpolant is non-trivial
            y = I.d
            if (np.all(y < zero_tol)):
                continue

            #if non-trivial, compute the interpolant
            prob = max(I([E])[0],0.0)

            #append the probability and column
            thisRowData.append(prob)
            nonZeroCols.append(key[1])

        current_block += num_non_zeros_i

        #check if there are any non-zeros
        if len(nonZeroCols) > 0:
            #perform a re-normalization to ensure rows sum to 1
            npRow = np.array(thisRowData)
            S     = np.sum(npRow)
            if (S > zero_tol):
                npRow /= S

            #make a row entry that is i repeated len(npRow) times
            r = [i] * len(npRow)

            #append to the global lists
            row.append(r)
            col.append(nonZeroCols)
            dat.append(npRow.tolist())


    #flatten all the lists
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    return imat


def eval_matrix_diff(ptmI, E, diff):
    #evaluate transition matrix and its derivative

    #set a tolerance for being considered 0
    zero_tol = 1e-5

    #get the number of states in the system
    num_states = len(ptmI)

    #init an array for the matrix at E
    imat = np.zeros((num_states, num_states))
    ider = np.zeros((num_states, num_states))

    #loop over each entry of the matrix
    for i in range(num_states):
        for j in range(num_states):

            #get the interpolant
            I = ptmI[i][j]

            #check if the interpolant is non-trivial
            y = I.d
            if (np.all(y < zero_tol)):
                imat[i][j] = 0
                continue

            #if non-trivial, compute the inteerpolant and set the value
            prob = I([[E,0]])
            d    = I([[E,0]], diff)
            imat[i][j] = prob
            ider[i][j] = d

        #perform a re-normalization to ensure rows sum to 1
        S = np.sum(imat[i])
        if (S > zero_tol):
            imat[i] /= S
            ider[i] /= S

    return imat, ider

def getParameterMap():
    #read the parameter map file to determine which parameters correspond to each ensemble

    mfile_loc = "../simulation/input_hpcc/parameterMap.txt"
    m = open(mfile_loc, 'r')
    Lines = m.readlines()
    H = []
    P = []
    for line in Lines:
        line = line.split()
        H.append(line[1])
        P.append(line[2])

    return H, P

if __name__ == "__main__":

    fitTransitionMatrixFull()

    #load the matrix of interpolators
    # with open("matrixI", 'rb') as f:
    #     Pfit = pickle.load(f)

    # #create a matrix at an intermediate value
    # I, D = eval_matrix_diff(Pfit, 6.0, (1,0))
    # print(D[12])
    # print(np.sum(I[12]))
    # print(np.sum(D[12]))
    #M = 50
    #print(timeit.timeit('I = eval_matrix(Pfit, 7.5)', globals=globals(),number=M)/float(M))