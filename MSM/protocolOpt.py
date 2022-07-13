'''
This file contains the implementation of the optimal control algorithm over a discrete
time markov chain to compute an optimal time dependent protocol for forming a desired
target state. 

We determine the boundary of the feasible set by using the nodes for which MSMs are 
generated at, and restrict the optimization to the interior of the the resulting 
concave hull. 

Gradients are computed using the adjoint formulation, and we perform gradient descent,
with a selection of possible penalty functions, and the option for a line search. 

For efficiency, transition matrices are stored in a cache using the shelve module, which
requires some level of discretization to determine the values to store the matrices for. 
If a particular matrix is not yet cached, we compute it online and store it in the cache 
for efficient access in the future. 

We construct a list to store information about global optimization results if using
stochastic gradient descent. Stores a collection of local optima, which can be sorted 
by either the yield, or total objective function value. 

'''

import sys
import os
import fnmatch
import inspect
import gc

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse

import pickle
import time

from operator import itemgetter
import objgraph

#import modules to do calculations on polygons
import shapely
import alphashape
from descartes import PolygonPatch

#import the MSM class object and MSM loading functions
from manualMSM import MSM
from manualMSM import loadMSMs
from manualMSM import loadStateDict

#import functions for loading MSMs and getting active set info
from interpolateMSM import getGlobalActive

#import functions for evaluating the interpolants
import interpolantEval as iEval

#import shelve for reading the cache on hdd
import shelve

#import functions for plotting optimization results
import analyzeProtocol


def getPreComputed(data_version = '', refine=True, interp = 'SQL'):
    #read in all objects that have been pre-computed

    #load the MSMs, number of ensembles, and parameter space nodes
    MSMs, K, E2 = loadMSMs(data_version=data_version, refine=refine)

    #get global active set
    global_active = np.array(getGlobalActive(MSMs, K))
    nsg = len(global_active)

    #get the dictionary of states: state to index
    stateDict, refineList = loadStateDict(data_version=data_version, refine=refine)

    #determine how to access interpolants
    if (interp == 'SQL'):

        #print notification and efficiency warning
        print("Using SQL database for interpolant objects")
        print("Warning: This method is the default, but slower until sufficiently "+ \
            "many transition matrices have been cached. Much more memory efficient.")

        #set the SQL database location with read only option
        iEval.setSQLdb(data_version=data_version, refine=refine)

        #get nsa from the statedict
        nsa = nsg

    if (interp == 'matrix'):

        #issue notification and memory warning
        print("Using a sparse matrix of interpolant objects")
        print("Warning: this mode is RAM intensive. Consider using SQL option if "+ \
               "memory becomes an issue.")

        #load the matrix of interpolators
        interp_file = "msm" + data_version + "/interp"
        if refine:
            interp_file += "Rdict"
        else:
            interp_file += "Dict"

        with open(interp_file, 'rb') as f:
            matrixI = pickle.load(f)
            print("Matrix of interpolants successfully loaded.")

        #create a global WORK_LIST using the matrixI object
        nsa = iEval.createWorkList(matrixI)
        print(nsa,nsg)

        #delete the matrixI to free redundant memory
        print("Interpolants have been assigned to jobs. Deleting matrix.")
        del matrixI

    #return non-globals
    return E2, global_active, nsg, nsa, stateDict

def msmSetup(data_version = '', refine='True', interp='SQL',
             initial_state = (0,0,0), target_state = (12,30,30), 
             spacing = 0.0075, cache_flag = True):
    #perform necessary setup to use MSMs to evaluate probabilities

    #get pre-computed data
    E2, globalActive, nsg, nsa, stateDict = getPreComputed(data_version=data_version, 
                                                           refine=refine, interp=interp)
    # stateDict = stateDict[0]
    # print(E2)
    # print(globalActive)
    # print(nsg, nsa)
    # print(stateDict)

    #get the inverse mapping of the state dictionary: index to state
    inv_map = {v: k for k, v in stateDict.items()}

    #get the boundary of the sampling region to enforce bounds on parameter values
    boundary = getBoundary(E2)

    #set up the discretization
    disc = setupDiscretization(E2, boundary, spacing, test=False)

    #set up the cache for evaluated transition matrices
    Pcache = init_cache(cache_flag, spacing, data_version=data_version, refine=refine)

    #get indices for the initial and target states
    initial_index = stateDict[initial_state]
    target_index  = stateDict[target_state]

    #get the target indices in the global index
    global_i = np.where(globalActive == initial_index)[0][0]
    global_t = np.where(globalActive == target_index )[0][0]

    #print a message about state info
    m1 = "There are {} states among all parameter sets.\n".format(nsa)
    m2 = "Initial State is {}, index is {}, global index is {}.\n".format(initial_state, initial_index, global_i)
    m3 = "Target State is {}, index is {}, global index is {}.".format(target_state, target_index, global_t)
    print(m1+m2+m3)

    #return all the needed values
    return E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, spacing, \
           cache_flag, Pcache, global_i, global_t


def getBoundary(parameters):
    #determine a polygonal boundary for the region defined by nodal parameters

    #convert to list of tuples
    nodes = [tuple(entry) for entry in parameters]

    #get the boundary and return it
    return alphashape.alphashape(nodes,0)


def enforceBounds(protocol, boundary, disc):
    #enforce that each point in protocol is within the bounds of the feasible polygon

    #make a list of tuples out of the protocol
    protocol_pairs = [(protocol[0][i], protocol[1][i]) for i in range(len(protocol[0]))]

    #get a list of bools corresponding to if each point is within the boundary
    is_inside = [insideBoundary(p, boundary) for p in protocol_pairs]

    #if outside the boundary, project to closest point
    for i in range(len(protocol_pairs)):
        if not is_inside[i]:

            #this determines the closest point using just the boundary
            #does not work with manual adjustments to the region

            # p = shapely.geometry.Point(protocol_pairs[i])
            # closest = shapely.ops.nearest_points(boundary, p)
            # protocol[0][i] = closest[0].x
            # protocol[1][i] = closest[0].y

            #this determines the closest point using the entire discretization
            #slower but more general in terms of adjusting the region manually

            #get squared distance from every point in disc to the point
            x, y = protocol[0][i], protocol[1][i]
            distances = np.array([(x-x0)**2 + (y-y0)**2 for x0,y0 in disc])

            #get the min distance and set the closest point
            min_dist = np.argmin(distances)
            closest = disc[min_dist]
            protocol[0][i] = closest[0]
            protocol[1][i] = closest[1]
            

    return protocol


def solveForwardEquation(TMs, T, initial, num_states):
    #solve the forward equation 

    #init the solution to the forward equation
    p = np.zeros((T+1, num_states))
    p[0, initial] = 1

    #solve the forward equation by applying transition matrix T times
    for i in range(T):
        p[i+1,:] = p[i,:] * TMs[i]

    return p


def solveBackwardEquation(TMs, T, target, num_states):
    #solve backward equation

    #init the solution to the backward equation
    F = np.zeros((num_states,T+1))
    F[target,T] = 1

    #solve the backward equation by applying transition matrix T times
    for i in range(T):
        F[:,T-1-i] = TMs[T-1-i] * F[:,T-i]

    return F



def solveKolmogorovEqs(TMs, T, initial, target, num_states):
    #solve the forward and backward kolmogorov equations
    #note: these are not time-homogenous - decomp method not recommended

    p = solveForwardEquation( TMs, T, initial, num_states)
    F = solveBackwardEquation(TMs, T, target , num_states)

    return p, F

def insideBoundary(point, boundary):
    #determine if a point is inside the boundary, with manual modifications to the region

    #check if the point in within the boundary
    if not boundary.contains(shapely.geometry.Point(point)):
        return False

    #manually filter rectangular region below disassembly region
    if point[0] < 1.4 + 1e-2 and point[1] < 1.4:
        return False

    #manually filter points below the nucleation curve
    if (point[1] < 2.8 - point[0]) and point[0] > 1.4:
        return False

    #if we reach here, we are in the region, and not a filtered portion
    return True



def setupDiscretization(parameters, boundary, spacing, test = False):
    #set up a discretization of the allowed domain via rejection from the bounding box

    #first determine the min and max of each dimension to define the bounding box
    m = map(min, zip(*parameters))
    M = map(max, zip(*parameters))
    min_x, min_y = list(m)
    max_x, max_y = list(M)

    #determine the number of points needed in each dimension to get desired spacing
    dx = max_x - min_x
    dy = max_y - min_y
    nx = round(dx / spacing) + 1
    ny = round(dy / spacing) + 1

    #get all points in the bounding box grid
    mesh = np.array(np.meshgrid(np.linspace(min_x, max_x, nx), np.linspace(min_y, max_y, ny)))
    combos = mesh.T.reshape(-1,2)

    #filter out points that are outside the boundary
    valid = []
    for entry in combos:

        #check if in the region, append if so
        if insideBoundary(entry, boundary):
            valid.append(entry)


    if test:
        print("There are {} discretization points".format(len(valid)))
        print("This will use approximately {} GB of HDD space.".format(14.0*len(valid)/1000.0))
        plt.figure()
        x = [entry[0] for entry in valid]
        y = [entry[1] for entry in valid]
        plt.scatter(x,y,s=2)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    #return the valid points
    return valid



def solveHeatEq(beta, g, T, lam, h, tau):
    #solve the heat equation for the gradient update
    # todo - check if needs to be updated. dt might need t be tuned

    #define cfl constant
    dt = 2.5                #set via experimentation
    k = dt * tau
    cfl = (lam * h) / (float(k)*float(k))

    #evaluate the rhs of the linear system
    rhs = np.zeros(T)
    rhs = beta + h * g

    #init the thomas algorithm
    c = np.zeros(T-1)
    c[0] = -2.0 * cfl / (1.0 + 2.0 * cfl)
    d = np.zeros(T)
    d[0] = rhs[0] / (1.0 + 2.0 * cfl)
    soln = np.zeros(T)

    #iterate the thomas algorithm
    for i in range(1,T-1):
        denom = 1.0 + 2.0 * cfl + cfl * c[i-1]
        c[i] = -cfl / denom
        d[i] = (rhs[i] + cfl * d[i-1]) / denom

    #set final component of d
    d[T-1] = (rhs[T-1] + 2.0 * cfl * d[T-2]) / (1.0 + 2.0 * cfl + 2.0 * cfl * c[T-2])

    #construct solution by back-substitution
    soln[T-1] = d[T-1]
    for i in range(T-2,-1,-1):
        soln[i] = d[i] - c[i] * soln[i+1]

    #return soln to heat equation
    return soln

def evalPgrad(p, f, derivs, interval, param):
    #evaluate the derivative of probability given forward and backward solutions

    #compute the weighted inner product and return it
    d = np.dot(p[interval,:], derivs[interval][param] * f[:,interval+1])
    return d


def get_matrix(E, nsa, nsg, interp='SQL'):
    #evaluate the matrices at the supplied energy values and return them

    #compute the sparse matrices in parallel - check for interp mode
    if interp == 'matrix':
        P, D = iEval.eval_matrix_deriv_sparsePar(E, nsa, nsg, NUM_PROCS)

    if interp == 'SQL':
        P, D = iEval.eval_matrix_deriv_sparsePar_SQL(E, nsa, nsg, NUM_PROCS)

    #return as a list of all the matrices
    return [P, D[0], D[1]]


def getClosestPoint(E, disc):
    #get the closest point in the discretization to E

    #get the distance from E to all points in disc
    dists = np.linalg.norm(disc-np.array(E), axis=1)

    #get the argmin of dists
    closest_key = np.argmin(dists)

    #get the discretization point and return it
    return disc[closest_key]


def get_matrix_cached(E, nsa, nsg, Pcache, disc, interp='SQL'):
    #check if the matrices have been cached, if not evaluate and add

    #get the key correspinding to the closest point in the discretization
    interp_key = getClosestPoint(E, disc)
    interp_key_string = "matrix_" + str(interp_key[0]) + "_" + str(interp_key[1])

    #check for the entry with this key. if None, compute and add it
    try:
        spMats = Pcache[interp_key_string]
        #print("Key = {} -> Loaded data from cache".format(interp_key_string))
    except:
        print("Key = {} -> Computing from scratch and adding to cache".format(interp_key_string))
        a = time.time()
        spMats = get_matrix(interp_key, nsa, nsg, interp=interp)
        c = time.time()
        print("Interp total took {} seconds.".format(c-a))
        Pcache[interp_key_string] = spMats

    #return the transition matrix, and a list of derivative matrices
    return spMats[0], spMats[1:3]


def init_cache(cacheFlag, spacing, data_version='', refine=True):
    #setup a caching system for the probability transition matrices

    #check if a cache is desired, if not, exit
    if (cacheFlag == False):
        print("Cache has not been requested. Will compute transition matrices upon request")
        print("Warning: Run time my be drastically increased. Consider caching.")
        return None

    #set the path to the desired cache
    cName  = "msm" + data_version + "/cache"
    if (refine):
        cName += "R"
    cName += "/cache" + str(spacing)

    #check if a cache exists for the given parameters, and load it
    if (os.path.exists(cName+'.dat')):
        Pcache = shelve.open(cName)
        print("Cache successfully loaded from disk")
    else:
        #if path does not exist, create it
        print("Cache not found at {}. Creating a new cache.".format(cName))
        Pcache = shelve.open(cName)

    #return the cache
    return Pcache

def getProbAllStates(protocol, T, nsa, nsg, initial, cache_flag, Pcache, disc, interp='SQL'):
    #compute the probability as a function of time for all states

    #init the storage for transition matrices along the protocol
    TMs = []

    #build the sequence of transition matrices using protocol E
    for interval in range(T):
        #get a transition matrix to use at given parameter value in protocol

        point = [protocol[0][interval], protocol[1][interval]]

        if (cache_flag):
            TM, _ = get_matrix_cached(point, nsa, nsg, Pcache, disc, interp=interp)
        else:
            TM = get_matrix(point, nsa, nsg, interp=interp)
            TM = TM[0]

        TMs.append(TM)

    #evaluate the probability from this protocol
    p = solveForwardEquation(TMs, T, initial, nsg)

    #return probabilities
    return p
    

def getProb(protocol, T, nsa, nsg, initial, target, cache_flag, Pcache, disc,interp='SQL'):
    #compute the final probability of target given the protocol

    #get the probabilities as fn of time for all states
    p = getProbAllStates(protocol, T, nsa, nsg, initial, cache_flag, Pcache, disc, 
                         interp=interp)

    #extract probability of target at final time
    prob = p[T, target]

    return prob


def gradient_fd(E, T, nsa, nsg, initial, target, cache_flag, Pcache, disc, 
                spacing = 0, interp='SQL'):
    #approximate gradient via finite differences

    #set spacing for finite difference
    #basically must be the discretization spacing if caching
    if (cache_flag):
        h = spacing
    else:
        h = 1e-4

    #get the initial probability
    p0 = getProb(E, T, nsa, nsg, initial, target, cache_flag, Pcache, disc,interp=interp) 

    #init a gradient
    g = np.zeros((PARAM, T))

    for param in range(PARAM):
        for i in range(T):
            E_trial = E
            E_trial[param][i] += h

            p = getProb(E_trial, T, nsa, nsg, initial, target, cache_flag, Pcache, disc,
                        interp=interp)  

            g[param][i] = (p - p0) / h

    return g



def perform_line_search(E, g, boundary, T, initial, target, nsa, nsg, prev_prob, 
                        regularize, lam, tau, 
                        cache_flag, Pcache, disc, spacing,
                        interp='SQL'):
    #do a line search to determine a step size to increase the objective
    #this method will ignore the regularizer and just choose a step for probability

    default  = 0.05  #default step size in case line search fails
    h0       = 1.25   #initial step size
    h0       = 1.50
    max_iter = 15    #max number of reductions
    alpha    = 0.5   #reduction factor

    try_FD = False

    #set initial step
    h = h0

    #eval the old regularizing objective
    old_S = 0
    if (regularize):
        for i in range(T-1):
            old_S += ((E[0][i+1]-E[0][i])/float(tau))**2
            old_S += ((E[1][i+1]-E[1][i])/float(tau))**2
        old_S *= -lam / 2.0

    #loop to determine the max step size we can take
    for i in range(max_iter):

        #perform a step 
        if (regularize):
            E0 = solveHeatEq(E[0], g[0,:], T, lam, h, tau)
            E1 = solveHeatEq(E[1], g[1,:], T, lam, h, tau)
            E_trial = [E0,E1]

        else:
            E_trial = E + h * g

        #enforce bounds on the protocol
        E_trial = enforceBounds(E_trial, boundary, disc)

        #evaluate the probability using this protocol
        prob = getProb(E_trial, T, nsa, nsg, initial, target, cache_flag, Pcache, disc,
                       interp=interp)

        #get the difference between probs, check if right sign
        diff = prob - prev_prob

        #get the new S if regularizing
        S = 0
        if (regularize):
            for i in range(T-1):
                S += ((E_trial[0][i+1]-E_trial[0][i])/float(tau))**2
                S += ((E_trial[1][i+1]-E_trial[1][i])/float(tau))**2
            S *= -lam / 2.0

            #add diff in S to diff
            diff += (S-old_S)

        #print diff
        print("Step size h = {} gives obj value diff = {}".format(h,diff))

        if (diff > 0):
            return E_trial, prob, prob+S

        #if no increase, reduce and retry
        h = h * alpha

        #if alpha goes below spacing when using a cache, stop
        if (cache_flag and h < spacing):
            print("No step size was found to give increase.")
            break 

    if (try_FD):
        #if the line search fails, estimate gradient via fd and try again
        #WARNING: not updated, dont use

        print("Trying FD Method")

        fdg = gradient_fd(E, T, nsa, nsg, initial, target, cache_flag, Pcache, disc, spacing,
                          interp=interp)

        h = h0
        #loop to determine the max step size we can take
        for i in range(max_iter):

            #get the updated protocol, and apply bounds
            if (regularize):
                E0 = solveHeatEq(E[0], fdg[0,:], T, lam, h, tau)
                E1 = solveHeatEq(E[1], fdg[1,:], T, lam, h, tau)
                E_trial = [E0,E1]
            else:
                E_trial = E + h * fdg

            E_trial = enforceBounds(E_trial, boundary, disc)

            #evaluate the probability using this protocol
            prob = getProb(E, T, nsa, nsg, initial, target, cache_flag, Pcache, disc, 
                           interp=interp)

            #get the difference between probs, check if right sign
            diff = prob - prev_prob

            #get the new S if regularizing
            if (regularize):
                S = 0
                for i in range(T-1):
                    S += ((E_trial[0][i+1]-E_trial[0][i])/float(tau))**2
                    S += ((E_trial[1][i+1]-E_trial[1][i])/float(tau))**2
                S *= -lam / 2.0

                #add diff in S to diff
                diff += (S-old_S)

            #print diff
            #print("Step size h = {} gives diff = {}".format(h,diff))

            if (diff > 0):
                return h, fdg

            #if no increase, reduce and retry
            h = h * alpha

            #if alpha goes below spacing when using a cache, stop
            if (cache_flag and h < spacing):
                break 

    #if no step can be found, return a fail output
    return E, -1, -1



def perform_optimization_c(E2, nsa, nsg, T, initial, target, cache_flag, Pcache,
                           disc, interp='SQL'):
    #determine an optimal constant via sampling

    #begin by defining a discretization to sample over
    spacing = 0.025

    #get the max and min for each parameter
    zipped = [*zip(*E2)]
    min_x, max_x = min(zipped[0]), max(zipped[0]) 
    min_y, max_y = min(zipped[1]), max(zipped[1]) 

    #make modified boundary - hard coded for current setup
    E2m = []
    tol = 1e-4       #tolerance for determining unique point in discr. grid
    for point in E2:
        if np.abs((point[0] - max_x)) < tol:
            px = point[0] + 0.001
        elif np.abs((point[0] - min_x)) < tol:
            px = point[0] - 0.001
        else:
            px = point[0]

        if np.abs((point[1] - min_y)) < tol:
            py = point[1] - 0.001
        elif np.abs((point[1] - max_y)) < tol:
            py = point[1] + 0.001
        else:
            py = point[1]

        E2m.append([px,py])
    boundary = getBoundary(E2m)

    #get the discretization
    sampling_disc = setupDiscretization(E2, boundary, spacing, test=True)

    samples = len(sampling_disc)

    #define an array of probabilities for each of the sample points
    probs = np.zeros(samples)
    probsT = np.zeros(samples, dtype=object)

    #evaluate the probability for all values
    for i in range(samples):
    # for i in range(39):

        #init list storage for all transition matrices and their derivatives
        TMs = np.zeros(T,dtype=object)

        #get the parameter value for this sample
        point = sampling_disc[i]

        #get a transition matrix to use at given parameter value
        if (cache_flag):
            TM, _ = get_matrix_cached(point, nsa, nsg, Pcache, disc, interp=interp)
        else:
            TM = get_matrix(point, nsa, nsg, interp=interp)
            TM = TM[0]

        #make every matrix in the protocol TM
        for time in range(T):
            TMs[time] = TM

        #solve the forward equation for p
        p = solveForwardEquation(TMs, T, initial, nsg)
        probs[i]  = p[T, target]
        probsT[i] = p[:,target]
        print("i={}, E={}, P={}".format(i, point, probs[i]))


    #return the arrays of sample points and probabilities
    return sampling_disc, probs, probsT

def getMatsFromProtocol(protocol, nsa, nsg, T, cache_flag, Pcache, disc, interp='SQL'):
    #get a sequence of transition matrices according to the parameters in the protocol

    #init list storage for all transition matrices and their derivatives
    TMs = []
    derivs = []

    #loop over each interval and append the resulting matrices
    for interval in range(T):
        #get a transition matrix to use at given parameter value

        point = [protocol[0][interval], protocol[1][interval]]
        if (cache_flag):
            TM, deriv = get_matrix_cached(point, nsa, nsg, Pcache, disc, interp=interp)
        else:
            TM = get_matrix(point, nsa, nsg, interp=interp)
            deriv = TM[1:3]
            TM = TM[0]

        TMs.append(TM)
        derivs.append(deriv)

    #return the sequence of matrices
    return TMs, derivs

def applyStabilization(TMs, derivs, mu, T, nsa, nsg, target, p_f):
    #compute the gradient of the stabilizing penalty term

    #init an array of derivatives of probability at t_k wrt parameters
    PkDj = np.zeros((PARAM, T+1, T))

    #init the gradient 
    g = np.zeros((PARAM, T))

    #solve the backward equation for each final time, eval gradient
    for k in range(1,T+1):

        #solve backward equation with terminal condition at t_k
        f = solveBackwardEquation(TMs, k, target, nsg)
        for interval in range(k):
            for param in range(PARAM):
                PkDj[param][k][interval] = evalPgrad(p_f, f, derivs, interval, param)

    
    #use the PkDj derivatives to update the gradient vector
    for j in range(T):
        for param in range(PARAM):

            #init the sum for the PkDj
            S = 0
            for k in range(j+1,T+1): #j+2?
                S += mu[k-1] * (p_f[k,target] - 1.0) * PkDj[param][k][j]

            #add to the full gradient
            g[param][j] -= S

    #return the gradient
    return g

def updateOptima(optima_list, optimum_data, num_optima):
    #update the list of optima to include the newly found one. Sort by 1st value
    #drop any optima exceeding the num_optima parameter

    #append the data
    optima_list.append(optimum_data)

    #sort it by the first value - target probability
    optima_list = sorted(optima_list, key=itemgetter(0), reverse=True)

    #check if the list is longer than num_optima, and delete last value
    L = len(optima_list)
    if L > num_optima:
        optima_list.pop()

    return optima_list


def perform_optimization(tau, T, initial, target, nsa, nsg, boundary, cache_flag, Pcache,
                         disc, spacing, lam = 0, mu = None, interp='SQL'):
    #determine an optimal protocol for beta to maximize target probability

    #convenience time discretization
    t = np.linspace(0,1,T)

    #gradients descent parameters
    max_its = 1000          #max number of iterations before terminating
    #max_its = 1
    h0 = 0.05               #constant step size for gradient descent updates
    tol = 5e-7              #convergence criteria on objective fn updates
    tol = 0

    #flags for modifications to GD
    stabilize   = False      #perform stabilizing regularization
    regularize  = False      #perform smoothness regularization
    line_search = True       #perform line search for step size
    global_opt  = True       #perform global optimization
    plot_updates= False      #plot the protocol every update to check progress

    #re-set the flags based on provided regularization parameters
    if (lam > 1e-12):
        regularize = True

    if (mu is not None):
        stabilize = True

    #set global optimization parameters
    num_optima = 10

    #choose an initial guess for parameter protocols - two dimensions
    guess = 0
    if (guess == 0): #constant guess
        E = np.ones((PARAM, T))
        E[0] *= 1.6
        E[1] *= 1.45

    elif (guess == 1): #linear guess
        E = np.ones((PARAM, T))
        E[0] *= np.linspace(1.6,1.8,T)
        E[1] *= np.linspace(1.2,1.3,T)
        
    elif (guess == 2): #oscillatory guess
        E = np.ones((PARAM, T))

        #this one for 200 jumps
        # E[1] *= 0.2 * np.cos(8*np.pi*t) + 1.6
        # E[0] *= 0.2 * np.cos(4*np.pi*t) + 1.4

        #this one for 320 jumps
        E[1] *= 0.2 * np.cos(12*np.pi*t) + 1.4
        E[0] *= 0.2 * np.cos(6*np.pi*t) + 1.4

        #this one tests the dependence on frequency
        # E[1] *= 0.05 * np.cos(10*np.pi*t) + 1.4
        # E[0] *= 0.3 * np.cos(10*np.pi*t) + 1.6

        #this one for (12,29,27)
        # E[1] *= 0.2 * np.cos(8*np.pi*t) + 1.5
        # E[0] *= 0.2 * np.cos(15*np.pi*t) + 1.5

    elif (guess == 3): #two phase guess
        E = np.ones((PARAM, T))
        E[0][0:25] *= 1.5
        E[0][25:256] *= 1.3
        E[1] *= 1.5

    elif (guess == 4): #two phase guess with weaker interactions in phase 2
        E = np.ones((PARAM, T))
        E[0][0:25] *= 1.5
        E[0][25:256] *= 1.2

        E[1][0:25] *= 1.5
        E[1][25:256] *= 1.4

    elif (guess == 5): #linear guess between optimal initial and final
        E = np.ones((PARAM, T))
        E[0] *= np.linspace(1.5,1.3,T)
        E[1] *= np.linspace(1.5,1.4,T)

        

    elif (guess == -1): #linear guess
        
        with open("data/protocol_test", 'rb') as f:
            E = pickle.load(f)
        print("Protocol loaded.")

        #modify the protocol
        # E[0][100:150] = 1.35
        # E[1][100:150] = 1.45

        # E[0][1:319] = 1.35
        # E[1][1:319] = 1.45
        
    #set the initial probability of the target as 0. to gauge convergence of GD fn iterates
    prev_prob = 0
    prev_obj  = 0

    #init a list of lists to store collection of optima found for global optimization
    optima_list = []

    #print a begin message
    print("\nBeginning gradient descent")

    #convert the parameter protocol to the sequence of transition matrices + derivs
    TMs, derivs = getMatsFromProtocol(E, nsa, nsg, T, cache_flag, Pcache, disc, 
                                      interp=interp)

    #solve the primal and adjoint equations
    p, f = solveKolmogorovEqs(TMs, T, initial, target, nsg)

    #output the initial yield
    print("Initial guess yield is {}".format(p[T,target]))

    #start gradient descent iterations
    try:
        for i in range(max_its):

            #evaluate each component of the probability gradient
            g = np.zeros((PARAM, T))
            for interval in range(T):
                for param in range(PARAM):
                    g[param][interval] = evalPgrad(p, f, derivs, interval, param)

            #evaluate gradient of stabilizing penalty if desired
            if (stabilize):

                g += applyStabilization(TMs, derivs, mu, T, nsa, nsg, target, p)

            #check to do line search
            if (line_search):

                newE, newP, newO = perform_line_search(E, g, boundary, T, initial, target, 
                                                       nsa, nsg, prev_prob, regularize, 
                                                       lam, tau, cache_flag, Pcache, disc, 
                                                       spacing, interp=interp)

                #check for failure of line search
                if (newP < 0):

                    #print an info message
                    print("A step size could not be found to increase obj fn value.")
                    print("Treating as local optima and continuing.")

                    #collect the optima data into a list
                    optimum_data = [prev_prob, prev_obj, E]

                    #perform a global optimization if requested
                    if global_opt:

                        #update the global optima storage with the previous values - local min
                        optima_list = updateOptima(optima_list, optimum_data, num_optima)

                        #come up with a noise protocol for sgd
                        if len(optima_list) < num_optima:
                            print("Adding noise to current protocol to escape minima")
                            noise = np.abs(np.random.normal(0, 0.1, ((PARAM,T)))) * np.sign(g)
                            E = E + noise
                            E = enforceBounds(E, boundary, disc)
                        else:
                            print("Adding noise to best protocol so far to continue optimization")
                            noise = np.abs(np.random.normal(0, 0.1, ((PARAM,T))))
                            E = optima_list[0][2] + noise
                            E = enforceBounds(E, boundary, disc)

                    #no global optimization
                    else:
                        #return the current local minimum - do it in a list to be consistent
                        optima_list.append(optimum_data)

                        #return list of optima with just the first local max
                        return optima_list, p

                #line search finds step and updates
                else:

                    #update the protocol to the new one
                    E = newE
                    prob = newP
                    obj = newO

            #do a normal gd w/o line search
            else:

                #do a standard gradient descent step
                h = h0
                grad = g

                #do gd step
                if (regularize):
                    E[0] = solveHeatEq(E[0], grad[0,:], T, lam, h, tau)
                    E[1] = solveHeatEq(E[1], grad[1,:], T, lam, h, tau)

                else:
                    E = E + h * grad

                #enforce bounds on the protocol
                E = enforceBounds(E, boundary, disc)

            
            #convert the new parameter protocol to the sequence of transition matrices + derivs
            TMs, derivs = getMatsFromProtocol(E, nsa, nsg, T, cache_flag, Pcache, disc, 
                                              interp=interp)

            #solve the primal and adjoint equations
            p, f = solveKolmogorovEqs(TMs, T, initial, target, nsg)

            #get the new probability
            prob = p[T,target]
            if (not global_opt):
                obj = prob

            #check for the stopping criteria and print iteration information
            r = np.abs(prob-prev_prob)
            print("Iter {}, Prob {}, residual {}".format(i, prob, r))
            prev_prob = prob
            prev_obj  = obj

            #plot if desired
            if (plot_updates):
                plt.plot(t,E[0],t,E[1])
                plt.show()
    except KeyboardInterrupt:
        pass

    #return the optimal protocol and probability vector
    return optima_list, p


def main(T, tau, animation_period, interp='SQL'):
    #run optimization and return polyfit for the protocol

    #do the setup to use MSM and interpolated transition matrix
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = msmSetup(interp=interp, target_state=(12,31,27))

    #convenience time discretization
    t = np.linspace(0,1,T)

    #set regularizers
    lam = 100000
    #lam = 0
    mu = 0.9*t*t+0.01
    one = t*0 + 1
    
    #use the transition matrices for the optimization
    opt_list, prob_star = perform_optimization(tau, T, global_i, global_t, nsa, nsg, boundary, 
                                             cache_flag, Pcache, disc, spacing, 
                                             lam = lam, mu = None, interp=interp)
    E_star = opt_list[0][2]

    E_samples, prob_samples, pt = perform_optimization_c(E2, nsa, nsg, T, 
                                                         global_i, global_t, 
                                                         cache_flag, Pcache, disc,
                                                         interp=interp)

    max_index = prob_samples.argmax()
    #max_index = 3
    print("Constant max is {} at E={}".format(prob_samples[max_index], E_samples[max_index]))
    pt_const_max = pt[max_index]

    #make plots using the results
    # E_star=[[1.5]*256, [1.5]*256]
    # prob_star=np.zeros((257,nsg))
    analyzeProtocol.plotResults(tau, animation_period, T, E_star, prob_star, global_t, E2,
                E_samples, max_index, pt_const_max, prob_samples)

    #close the open cache file
    Pcache.close()
    print("Cache file closed")

    #pickle the protocol to use elsewhere
    with open("data/protocol_test", 'wb') as f:
        pickle.dump(E_star, f)

    return 0



if __name__ == '__main__':

    #set parameters
    animation_period = 25.0   #time between measurements for MSM
    tau              = 125     #lag time for MSM creation
    T                = 256     # 800000/(anim*tau)

    global NUM_PROCS
    NUM_PROCS = 4

    global PARAM
    PARAM = 2

    interp = 'matrix'
    interp = "SQL"

    main(T, tau, animation_period, interp=interp)

    #analyze_protocol(T, tau, animation_period)

