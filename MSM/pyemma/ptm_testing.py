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
import scipy
import scipy.io
import pyemma

import pickle
import time

import rbf
from rbf.interpolate import RBFInterpolant

from interpolatePYE import eval_matrix
from interpolatePYE import eval_matrix_sparse
from interpolatePYE import getParameterMap


def plotCommittor(committor, active_set, map):
    #plot the committor probabilities

    L = len(committor)

    probs = np.zeros((17,31))

    for i in range(L):
        prob = committor[i]
        state = active_set[i]
        pair = map[state]
        #print(pair, state)

        probs[pair[0], pair[1]] = prob

    #plot it 
    plt.imshow(probs)
    plt.colorbar()
    plt.xlabel("Number of Bonds")
    plt.ylabel("Number Attached Subunits")
    plt.show()

def computeCommittor(P, initial, target, num_states):
    #compute the committor probability  for eahc state

    R = np.array(P)
    b = np.zeros(num_states)

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

    x = np.linalg.solve(R,b)
    #print(x)

    return x

def computeTPTflux(P, eq, initial, target, num_states):
    #compute flux network for transition from initial to target

    #get the committor
    q = computeCommittor(P, initial, target, num_states)

    #init the flux matrix
    f = np.zeros((num_states, num_states),dtype=float)

    #compute fluxes from q, eq, and P
    for i in range(num_states):
        for j in range(num_states):
            f[i][j] = eq[i] * (1.0-q[i]) * P[i][j] * q[j] * 1e6
        f[i][i] = 0.0

    #get the net flux 
    fp = f - f.transpose()

    #zero out all negative entries
    fp[fp < 0] = 0

    return fp


def extractPath(parent, vertex, target, path):
    #get the path that is stored in parent

    # global parent
    if (vertex == 0):
        return

    #recursively get path
    extractPath(parent, parent[vertex], target, path)
    path.append(vertex)
    

def widest_path(Graph, src, target):
    # To keep track of widest distance
    widest = [-10**9]*(len(Graph))
 
    # To get the path at the end of the algorithm
    parent = [0]*len(Graph)
 
    # Use of Minimum Priority Queue to keep track minimum
    # widest distance vertex so far in the algorithm
    container = []
    container.append((0, src))
    widest[src] = 10**9
    container = sorted(container)
    while (len(container)>0):
        temp = container[-1]
        current_src = temp[1]
        del container[-1]
        for vertex in Graph[current_src]:
 
            # Finding the widest distance to the vertex
            # using current_source vertex's widest distance
            # and its widest distance so far
            distance = max(widest[vertex[1]],
                           min(widest[current_src], vertex[0]))
 
            # Relaxation of edge and adding into Priority Queue
            if (distance > widest[vertex[1]]):
 
                # Updating bottle-neck distance
                widest[vertex[1]] = distance
 
                # To keep track of parent
                parent[vertex[1]] = current_src
 
                # Adding the relaxed edge in the priority queue
                container.append((distance, vertex[1]))
                container = sorted(container)
    path = []
    extractPath(parent, target, target, path)

    return widest[target], path

def highestFluxPathways(f, initial, target, num_states, inv_map, a_set):
    #determine the highest flux pathways

    #begin my making a graph structure out of the flux matrix f
    #Graph[i] = [(f_{ij},j),...,]
    Graph = [[] for i in range(num_states+1)]

    for i in range(1,num_states+1):
        for j in range(1,num_states+1):
            if (f[i-1][j-1] > 0):
                Graph[i].append((f[i-1][j-1],j))

    #set iteration and output parameters
    num_paths = 10
    f_out = open("flux_out.txt",'w')

    #compute the desired number of highest flux paths
    for path_num in range(num_paths):

        #pass the graph to the widest path algo
        c, path = widest_path(Graph, initial+1, target+1)
        
        #print out the path and fluxes to a file
        f_out.write("Path {}. Capacity {}.\n".format(path_num+1, c))
        print("Path {}. Capacity {}.".format(path_num+1, c))
        for i in range(len(path)-1):
            state = path[i]-1
            full_state = a_set[state]
            config = inv_map[full_state]
            f_out.write("{} {}\n".format(config, f[state][path[i+1]-1]))
            print(config, f[state][path[i+1]-1])
            #subtract the capacity from all entries in the path
            f[state][path[i+1]-1] -= c
        i = len(path)-1
        state = path[i]-1
        full_state = a_set[state]
        config = inv_map[full_state]
        f_out.write("{}\n".format(config))
        f_out.write("\n")
        print(config,"\n")

        #construct a new graph of the modified flux network
        Graph = [[] for i in range(num_states+1)]

        for i in range(1,num_states+1):
            for j in range(1,num_states+1):
                if (f[i-1][j-1] > 0):
                    Graph[i].append((f[i-1][j-1],j))

    f_out.close()
    return


def computeTargetProbability(P, T, initial, target, num_states):
    #compute the probability of being in the target as a function of time using P

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
    #print(p[T,:])

    return t, p[:,target]




def computeSampledTargetProbability(folder, target_state):
    #compute p(t) estimated using sampled trajectories

    #count number of npy files
    trajs = fnmatch.filter(os.listdir(folder), '*.npy')

    #init storage
    samples = 0
    frames = 0

    for npy_file in trajs:

        #load the npy file
        try:
            #print(npy_file)
            path = np.load(folder+npy_file)
        except:
            continue

        if (frames == 0): #do the first time setup
            frames = len(path)
            p = np.zeros(frames+1)

        samples += 1

        count = 1
        for pair in path:
            if (pair[0] == target_state[0] and pair[1] == target_state[1] and pair[-1] == target_state[2]):
                p[count] += 1
                #break

            count += 1

    #get time discretization and normalize p
    t = np.linspace(0,500000, frames+1)
    p = p / float(samples)

    print(samples)

    return t, p





def dtram_test():

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get index of initial and target state
    initial = stateDict[(0,0)]
    target  = stateDict[(12,30)]
    print(initial, target)

    #get the dtram model from pickle
    with open("dtram4", 'rb') as f:
        dtram_obj = pickle.load(f)

    #extract the models from dtram
    models = dtram_obj.models
    L = len(models)

    #extract the data from each ensemble
    MSMs = []
    for i in range(L):
        MSMs.append(models[i])


    ensemble = 0
    x, t, p, a_set = analyzeMSM(MSMs[ensemble], initial, target)

    #set the folder containing trajectories
    folder = "../trajectories/E4.0S6.0/"
    t_s, p_s = computeSampledTargetProbability(folder)

    plotCommittor(x, a_set, inv_map)

    #plot
    plt.plot(t, p, t_s, p_s)
    plt.xlabel("Time")
    plt.ylabel("Fraction Complete")
    plt.legend(["Markov Model", "Simulation"])
    plt.show()






def analyzeMSM(msm, initial, target):
    #use given msm to compute an active set and behavior of target prob as fn of time

    #get the active set and the indices of the initial and target state
    a_set = msm.active_set
    #print(a_set)
    a_initial = np.where(a_set == initial)[0][0]
    a_target  = np.where(a_set == target )[0][0]
    print(a_initial, a_target)

    #get num states active
    num_states = len(a_set)

    #print the connections for the target state
    for i in range(num_states):
        if msm.P[a_target][i] > 0:
            print("State: {}, Prob: {}".format(i, msm.P[a_target][i]))

    #compute the committor
    x = computeCommittor(msm.P, a_initial, a_target, num_states)

    #get the target probability as a function of time
    t, p = computeTargetProbability(msm.P, 50, a_initial, a_target, num_states)

    #scale t by lag time tau
    animate_time = 10
    tau = 2
    t = t * tau * animate_time

    #return computed values
    return x, t, p, a_set





def msm_test():

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get index of initial and target state
    initial = stateDict[(0,0,0)]
    target_state = (12,30,30)
    target  = stateDict[target_state]
    print("Target State is {}. Index is {}".format(target_state,target))
    num_states = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get the msm model from pickle
    with open("msm_test_out", 'rb') as f:
      msm = pickle.load(f)

    print(msm.active_set)

    #analyze the MSM
    folder = "../trajectories/P1.7H1.2/"
    tau = 40
    animate_time = 25
    initial_active = np.where(msm.active_set == initial)[0][0]
    target_active  = np.where(msm.active_set == target)[0][0]
    print("Target active index is ", target_active)
    #P = manual_count_matrix(folder, stateDict, tau)
    t, p = computeTargetProbability(msm.P, int(500000 / (tau*animate_time)), initial_active,
                                    target_active, len(msm.active_set))

    #scale t by lag time tau
    t = t * tau * animate_time

    #get the equilibrium distribution
    eq = msm.stationary_distribution
    print(eq)

    #get the flux
    # nsa = len(msm.active_set)
    # f = computeTPTflux(msm.P, eq, initial_active, target_active, nsa)
    # highestFluxPathways(f, initial_active, target_active, nsa, inv_map, msm.active_set)

    #set folder with sampled trajs and compute MC estimate
    print("Computing fc estimate from sampling")
    t_s, p_s = computeSampledTargetProbability(folder, target_state)

    plotType = 1

    if (plotType == 0): #plot with interpolant as well
        #analyze the interpolant
        tI, pI = computeTargetProbability(I, 250, initial, target, num_states)
        tI *= 8.0
        plt.plot(t, p, tI, pI, t_s, p_s)
        plt.xlabel("Time")
        plt.ylabel("Fraction Complete")
        plt.legend(["Markov Model", "Interpolated Markov Model", "Simulation"])
        plt.show()
    elif (plotType == 1): #plot just MSM and sampling
        plt.plot(t, p, t_s, p_s)
        plt.xlabel("Time")
        plt.ylabel("Fraction Complete")
        plt.legend(["Markov Model", "Simulation"])
        plt.show()
    elif (plotType == 2): #plot just sampling and direct count matrix
        #get target prob from manual count matrix
        print("Computing count matrix directly")
        lag = 50
        m_factor = 2*lag
        fracpow = 1
        pc = manual_count_matrix(folder, stateDict, lag)
        # d= {}
        # d['mat'] = pc
        # scipy.io.savemat("mat50.mat", d)
        # print(np.linalg.eig(pc))
        # sys.exit()
        #take sqrt of pc
        pc = scipy.linalg.fractional_matrix_power(pc, fracpow)
        for i in range(num_states):
            for j in range(num_states):
                if (pc[i][j] < -1e-6):
                    print(i,j,pc[i][j])
                    pc[i][j] = 0
            print(np.sum(pc[i]))
            pc[i] /= np.sum(pc[i])
        m_factor *= fracpow
        tm, pm = computeTargetProbability(pc, int(2000/m_factor), initial, target, num_states)
        tm *= m_factor
        plt.plot(tm, pm, t_s, p_s)
        plt.xlabel("Time")
        plt.ylabel("Fraction Complete")
        plt.legend(["Direct MM", "Simulation"])
        plt.show()

    #analyze each MSM
    # xL , tL , pL , s_setL  = analyzeMSM(msmL,  initial, target)
    # xLS, tLS, pLS, a_setLS = analyzeMSM(msmLS, initial, target)
    # #plotCommittor(xLS, a_setLS, inv_map)

    # #plot from sampling data
    # t_s, p_s = computeSampledTargetProbability()

    # plt.plot(tL, pL, tLS, pLS, t_s, p_s)
    # plt.xlabel("Time")
    # plt.ylabel("Fraction Complete")
    # plt.legend(["Markov Model - Long Traj", "Markov Model - Long + Short", "Simulation"])
    # plt.show()


def manual_count_matrix(folder, stateDict, lag):
    #manually construct a count matrix using the trajectories in folder

    #get number of states
    num_states = len(stateDict)

    #get all npy files
    trajs = fnmatch.filter(os.listdir(folder), '*.npy')

    #init a matrix for the counts
    C = np.zeros((num_states, num_states))
    P = np.zeros((num_states, num_states))

    #loop over npy files, get counts
    for npy_file in trajs:

        #load the npy file
        try:
            #print(npy_file)
            path = np.load(folder+npy_file)
        except:
            continue

        
        #loop over all entries in the path, get transition for i+tau
        for i in range(len(path)-lag):
            #get initial and final state
            s = path[i]
            sl = path[i+lag]
            q = [s[0], s[1], s[4]]
            ql = [sl[0], sl[1], sl[4]]
            state1 = stateDict[tuple(q)]
            state2 = stateDict[tuple(ql)]

            #update the count matrix
            C[state1][state2] += 1
            print(state1, state2)

    #normalize rows of the count matrix to get probability matrix
    for i in range(num_states):
        row = C[i]
        S = np.sum(row)
        if (S > 1e-6):
            P[i] = row / S
        else:
            P[i][i] = 1.0

    #return probability matrix
    return P




    

def interpolant_test():
    #see how the transition matrix interpolants behave at integers and midpoints

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    with open("msmPYE/global_active", 'rb') as f:
        global_active = np.array(pickle.load(f))

    #load the matrix of interpolators
    with open("msmPYE/matrixI", 'rb') as f:
        Pfit = pickle.load(f)

    #get index of initial and target state
    initial = stateDict[(0,0,0)]
    target  = stateDict[(12,30,30)]
    tau = 2
    animate_time = 50
    num_states = len(stateDict)
    nsg        = len(global_active)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #create list to store probability functions
    msm_probs = []
    msmIndex  = []
    m         = []

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

        except: #could not unpickle msm, raise error
            print("Could not find {}".format(msm_file))
            raise()

        #get the active state indices
        initial_active = np.where(msm.active_set == initial)[0][0]
        target_active  = np.where(msm.active_set == target )[0][0]

        #get the target prob as fn of time
        t, p = computeTargetProbability(msm.P, int(400000 / (tau*animate_time)), 
                                        initial_active, target_active, 
                                        len(msm.active_set))

        print(p)

        #scale t by lag time tau
        t = t * tau * animate_time

        #append p to msm_probs
        msm_probs.append(p)
        msmIndex.append(int(msm_file.split('msm')[1]))
        m.append(msm.P)

        
    #get the number of ensembles found
    K = len(msm_probs)

    #get the energy parameters for each msm found, using the index to parameter map
    E2 = np.zeros(K, dtype=object)
    for k in range(K):
        index = msmIndex[k]
        p = P[index]
        h = H[index]
        E2[k] =[p,h]

    #specify midpoints or integers
    test = 0   # 0 -> integers, 1-> midpoints

    #get the same probabilities at the midpoints in energy
    if (test == 0):
        E = E2
    elif (test == 1):
        E = [4.5, 5.5, 6.5, 7.5, 8.5]

    #evaluate the interpolant matrix at the desired values
    i_probs = []
    for i in range(len(E)):

        #evaluate the matrix interpolant at these parameters
        print(E[i])
        I = eval_matrix_sparse(Pfit, E[i], nsg)

        # print(I - m[0])
        # sys.exit()


        #get the target indices in the global index
        global_i = np.where(global_active == initial)[0][0]
        global_t = np.where(global_active == target )[0][0]

        print(global_i, global_t)
        print(inv_map[global_active[global_t]])
        print(I.getrow(global_t))
        print(m[i].getrow(target_active))

        #evaluate the target probabilities
        tI, pI = computeTargetProbability(I, int(400000 / (tau*animate_time)), 
                                          global_i, global_t, nsg)
        print(pI)
        
        #append to i_probs
        i_probs.append(pI)

    #plot each set of two msm probabilities, with the midpoint for comparison
    fig = plt.figure(figsize=(12, 8))

    for i in range(len(E)):
        ax = fig.add_subplot(2,1,i+1)
        if (test == 1):
            ax.plot(t, msm_probs[i], label = "MSM, E = {}".format(4+i))
            ax.plot(t, msm_probs[i+1], label = "MSM, E = {}".format(4+i+1))
            ax.plot(t, i_probs[i], linestyle = '--', label = "Interpolant, E = {}".format(E[i]))
        elif (test == 0):
            ax.plot(t, msm_probs[i], label = "MSM, E = {}".format(E[i]))
            ax.plot(t, i_probs[i], linestyle = '--', label = "Interpolant, E = {}".format(E[i]))
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Fraction Complete")


    plt.show()



############################################################################################


# def view_database():
#     #view information on the global database in an ensemble

#     #set the name of the database
#     folder = "../trajectories/E4.0S6.0/"
#     db_name = folder + "stateDBg"

#     #open the database
#     with open(db_name, 'rb') as f:
#         db = pickle.load(f)

#     #loop over each state, print its pair and times seen
#     for state in db.states:
#         print(state.pair, state.times_seen)

#     obs = db.get_observed_distribution()
#     print(obs)
#     pdf = obs_to_distribution(obs)

#     #choose an initial state
#     for i in range(20):
#         ic_state = sample_discrete_rv(pdf)
#         print(db.states[ic_state].pair)

#     return










def plotPTM():

    #load the matrix
    with open("msmPYE/matrixI", 'rb') as f:
        Pfit = pickle.load(f)

    with open("msmPYE/global_active", 'rb') as f:
        global_active = pickle.load(f)

    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    print(stateDict[(12,26,12)])
    print(inv_map[93])
    print(inv_map[94])
    print(inv_map[101])
    print(inv_map[102])
    print(inv_map[26])
    print(inv_map[7])
    #sys.exit()

    #get index of initial and target state
    initial = stateDict[(0,0,0)]
    target  = stateDict[(12,30,30)]
    num_states = len(stateDict)
    print("The target state is {}".format(target))
    print("{} states have been visited among all simulations".format(num_states))

    #plot the transition probability between any two states
    x1, x2 = np.linspace(1, 2, 100), np.linspace(1, 2, 100)
    x_itp = np.reshape(np.meshgrid(x1, x2), (2, 100*100)).T
    i1 = target
    i2 = 40

    show_all = True

    for key in Pfit.keys():
        i1 = key[0]
        i2 = key[1]

        if (i1 == target or i2 == target or show_all == True):

            interp = Pfit[key]
            u_itp = interp(x_itp)
            z = interp.d
            if (np.all(z < 0.005)):
                continue
            if (np.any(z < 0.01)):
                continue
            x = interp.y + interp.center

            y = [x[i][1] for i in range(len(x))]
            x = [x[i][0] for i in range(len(x))]
            s = interp.sig
            s1 = global_active[i1]
            s2 = global_active[i2]
            print("Plotting P[{}][{}]. Transition {} -> {}".format(i1,i2,inv_map[s1],inv_map[s2]))
            # print(x)
            # print(y)
            # print(s)
            #plt.plot(x_itp, u_itp, x, y, 'o')
            plt.figure()
            vm = np.min(u_itp)
            vM = np.max(u_itp)
            plt.tripcolor(x_itp[:,0], x_itp[:,1], u_itp, vmin=0, vmax=vM, cmap='viridis', alpha=0.9)
            plt.scatter(x, y, s=50, c=z, vmin=0, vmax=vM,cmap='viridis', edgecolor='k', alpha=1)
            plt.colorbar()
            plt.show()

    '''
    List of very bad estimates to check on
    1 -> 4
    1 -> 27
    '''


def sparse_test():
    #perform matrix multiplications for forward solve using dense and sparse 
    #check if sparse results in better efficiency
    #get memory estimates - see if caching is viable


    #get the state mapping from pickle
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #load the matrix of interpolators
    with open("msm50/matrixI", 'rb') as f:
        Pfit = pickle.load(f)

    #get index of initial and target state
    initial = stateDict[(0,0)]
    target  = stateDict[(12,30)]
    num_states = len(stateDict)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #define a set of E to run for
    E = [4.1,4.7,6.5,7.6,4.2,4.1,6.9,8.9,6.6,5.2]
    M = len(E)

    #call once to do initialization so it doesnt skew the tests
    Id = eval_matrix(Pfit,7)
    Is = eval_matrix_sparse(Pfit,7)

    timeCreate = True
    if (timeCreate):

        #time sparse
        print("Timing the sparse creation. Average over {} execs.".format(M))
        t0 = time.time()
        for i in range(M):
            Is = eval_matrix_sparse(Pfit,E[i])
        t1 = time.time()
        print("Average time is {}".format((t1-t0)/float(M)))

        #time dense
        print("Timing the dense creation. Average over {} execs.".format(M))
        t0 = time.time()
        for i in range(M):
            Id = eval_matrix(Pfit,E[i])
        t1 = time.time()
        print("Average time is {}".format((t1-t0)/float(M)))

    #test the difference in memory usage
    size_s = Is.shape[0]
    size_d = num_states*num_states
    size_float = sys.getsizeof(Id[1][1])
    size_int = sys.getsizeof(1)
    print("Sparse Memory: {} bytes".format(size_s*(size_float+2*size_int)))
    print("Dense  Memory: {} bytes".format(size_d*size_float))


    timeMult = True
    if (timeMult):
        #test difference in matrix-vector product speed
        p0 = np.zeros(num_states).T
        p0[0] = 1.0

        #number of matrix multiplies
        T = 2000

        #time sparse
        p = p0
        print("Timing the sparse multiply. Time for {} multiplies.".format(T))
        t0 = time.time()
        for i in range(T):
            #p = p * Is
            p = np.dot(p, Is.toarray())
        t1 = time.time()
        print("Elapsed time is {}".format(t1-t0))

        #time dense
        p1 = p0
        print("Timing the dense multiply. Time for {} multiplies.".format(T))
        t0 = time.time()
        for i in range(T):
            p1 = p1.dot(Id)
        t1 = time.time()
        print("Elapsed time is {}".format(t1-t0))

        #check results are the same
        diff = np.abs(p1-p)
        diff_sum = np.linalg.norm(diff)
        print("Residual between methods is {}".format(diff_sum))


    timeCache = False
    if (timeCache):
        dx = 0.01
        mat_dict = {}

        for i in range(101):
            mat_dict[int(4/dx + i)] = None

        print(mat_dict)

        E = [np.random.random() + 4 for i in range(1000)]
        M = len(E)

        t0 = time.time()
        for i in range(M):
            e = E[i]
            er= int(np.around(e/dx,0))

            if (mat_dict[er] is not None):
                I = mat_dict[er]
                print("Extracted for E={}->{}".format(e,er))
            else:
                I = eval_matrix_sparse(Pfit, er*dx)
                mat_dict[er] = I
                print("COmputed for E={}->{}".format(e,er))
        t1 = time.time()
        print("Average cached time is {}".format((t1-t0)/float(M)))












if __name__ == '__main__':

    #get the msm model and plot fc and other quantities
    msm_test()
    #dtram_test()

    #check a database for visited states and frequency distribution
    #view_database()

    #plot the individual interpolants in the transition matrix
    #plotPTM()

    #compare interpolated transition matrix fc with base MSMs
    #interpolant_test()

    #compare performance with sparse and dense transition matrices
    #sparse_test()