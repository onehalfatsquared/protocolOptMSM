import matplotlib.pyplot as plt
import numpy as np
import pyemma
import sys
import os
import fnmatch
import glob

#import pickle
import dill as pickle


def exclude(pair):
    #define rules to exclude a trajectory

    return False


def manualMapToState(stateDict, traj, frames):
    #map the trajectory in (n_s, n_b) space, to non-negative state index

    #init an array for the path in state space
    state_path = np.zeros(frames, dtype=int)

    #loop over tuples, get state index
    for i in range(0,frames):
        s = [traj[i][0], traj[i][1], traj[i][4]]
        pair = tuple(s)
        if (exclude(pair)):
            return np.zeros(0)
        state = stateDict[pair]
        state_path[i] = state
        if (state == 1331):
            print(state_path[i-10:i-1])
            #sys.exit()

    return state_path


def extractTrajectories(dtrajs, folder, traj_files, stateDict):
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
        print(file)
        traj = manualMapToState(stateDict, data, frames)
        print(traj)

        #append to dtraj
        if (len(traj)):
            dtrajs.append(traj)

    return

def removeAbsorbing(dtrajs, max_lag):
    #add in an artificial transition to remove absorbing states

    #load the msm for E=4 - has full connectivity
    with open("data/msm5", 'rb') as f:
        baseMSM = pickle.load(f)

    #construct an MSM from dtrajs to access the count matrix
    preMSM = pyemma.msm.estimate_markov_model(dtrajs, lag=1, reversible=False)

    #get the count matrix and active set
    a_set  = preMSM.active_set
    counts = preMSM.count_matrix_full
    print(a_set)

    #loop over the rows, find ones with only the diagonal entry. add to absorbing
    num_states = len(counts)
    absorbing = []
    for i in range(num_states):
        if i not in a_set:
            active_states = 0
            for j in range(num_states):
                state_count = counts[j][i]
                if state_count > 0:
                    active_states += 1
                    #print("State {} interacts with {}".format(i,j))

            if active_states > 0:
                absorbing.append(i)
                print("State {} is part of the absorbing set".format(i))

    #for each absorbing state, determine the most likely transition from E=4
    addedPairs = []
    for state in absorbing:

        #get the index of this state in E=4 msm
        base_set = baseMSM.active_set
        try:
            base_index= np.where(base_set == state)[0][0]
        except:
            print("This state is also not in the reference ensemble. Skipping")
            continue

        #get the row for this state in PTM
        row = baseMSM.P[base_index]
        print("In the base ensemble, row {} has distribution:".format(base_index))
        for i in range(len(row)):
            if (row[i] > 0):
                print("{} {}".format(base_set[i],row[i]))

                #delete the entries corresponding to states not in the current active set
                if (base_set[i] in absorbing):
                    row[i] = 0
                    print("Removing state {} as an option. Absorbing.".format(base_set[i]))

        
        #get the location of the maximum in this list. convert to a state
        max_index = np.argmax(row)
        max_state = base_set[max_index]

        #introduce artificial transition between state and max_state
        a_traj = [state, *[max_state]*max_lag]
        print("Adding artificial trajectory {}".format(a_traj))
        dtrajs.append(a_traj)

        #keep record of the trasition that was added
        addedPairs.append((state, max_state))

    return addedPairs

def getQ(msm, initial, absorbing, tau):
    #get the q value for the given ensemble using fixed point iteration

    #fixed point parameters
    max_iter = 1000
    tol      = 1e-6
    alpha = 0.9

    #get the active set for th current msm
    a_set = msm.active_set
    init  = np.where(a_set == initial)[0][0]
    abso  = np.where(a_set == absorbing)[0][0]

    #set the forward rate
    a = msm.P[init][abso] / float(tau)

    #get the staying probability
    p = msm.P[abso][abso]
    if (p < tol): #if the state immediately disappears, skip it
        return -1

    #do fixed point iteration
    Q = 0.1
    for i in range(max_iter):
        Q_new = (1-alpha)*(-a + (a+Q*np.exp(-(a+Q)*float(tau))) / p) + alpha * Q
        dx = np.abs(Q-Q_new)
        if (dx < tol):
            break
        Q = Q_new
        #print("Iter: {}, Q: {}, tol: {}".format(i,Q,dx))

    if (i < max_iter-1):
        print("Q converged to {} in {} iterations".format(Q_new, i))
    else:
        print("Q did not converge to desired tolerance in {} allowed iterations".format(max_iter))

    return Q


def fixBackward(msm, E, addedPairs, stateDict, inv_map, tau):
    #set backward rates for artificial transition by an exponential rate assumption

    #first load in the reference MSMs, E=4 and E=5
    with open("data/msm4", 'rb') as f:
        ref1 = pickle.load(f)
        E1   = 4.0
    with open("data/msm5", 'rb') as f:
        ref2 = pickle.load(f)
        E2   = 5.0

    #set the common saccfold interation
    Es = 6.0

    #set active set
    a_set = msm.active_set

    #have an outer loop over the added interactions
    for pair in addedPairs:

        #do the probability computation, move to fn later
        #get the initial and absorbing state
        initial =   pair[1]
        absorbing = pair[0]

        init  = np.where(a_set == initial)[0][0]
        abso  = np.where(a_set == absorbing)[0][0]


        pair_i = inv_map[initial]
        pair_a = inv_map[absorbing]

        #get the q (rate) values for the two reference ensembles
        q1 = getQ(ref1, initial, absorbing, tau)
        q2 = getQ(ref2, initial, absorbing, tau)
        if (q1 == -1 or q2 == -1): # dont overwrite if the state is not absorbing
            print("State {} breaks instantly in lower ensembles. Will not overwrite.".format(absorbing))
            continue

        #use q values to determine fit parameters
        dE1 = (-pair_i[0]*Es - pair_i[1]*E1 + pair_a[0]*Es + pair_a[1]*E1)
        dE2 = (-pair_i[0]*Es - pair_i[1]*E2 + pair_a[0]*Es + pair_a[1]*E2)

        alpha = -np.log(q2 / q1) / (dE2 - dE1)
        k     = q1 * (q1/q2)**((dE1)/(dE2-dE1))

        #evaluate the staying probability in the new ensemble
        a  = msm.P[init][abso]
        dE = (-pair_i[0]*Es - pair_i[1]*E + pair_a[0]*Es + pair_a[1]*E)
        q  = k * np.exp(-alpha * dE)
        p  = (a+q*np.exp(-(a+q)*float(tau)))/(a+q)

        #erase the row to make room for current entries
        for i in range(len(a_set)):
            msm.P[abso][i] = 0.0
        #set the staying probability p and the leaving probability as 1-p
        msm.P[abso][init] = 1-p
        msm.P[abso][abso] = p
        #display output message on success
        print("Staying probability for state {} overwritten as {}".format(absorbing, p))
        print(msm.P[abso])



    return


def time_scale_analysis(dtraj, LAGS):
    #compute MSMs at various lag times, plot timescales, do markovity test

    #use discrete trajectories to compute MSMs and look at first few implied timescales
    its = pyemma.msm.its(dtraj, lags=LAGS, nits=5, reversible=True, 
                         errors='bayes', only_timescales=True)

    #plot the timescales
    pyemma.plots.plot_implied_timescales(its, ylog=False);
    plt.show()


    return




def testMSM(traj_folder):
    #set up and perform tests to determine MSM parameters

    #get the stateDict dictionary mapping
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get info about the state space and target
    num_states = len(stateDict)
    target_index = stateDict[(12,30,30)]
    print("Target state index is {}".format(target_index))
    print("There are {} states".format(num_states))

    #set the folders containing the trajectories. Include short folder for rachet trajs
    long_folder  = traj_folder
    # short_folder = long_folder + "short/"
    # short_folder = "../trajectories/E4.0S6.0/short/"
    # extra_folder = "../trajectories/E4.0S6.0/"

    #get all trajectory files
    long_trajs  = fnmatch.filter(os.listdir(long_folder), '*.npy')
    try:
        short_trajs = fnmatch.filter(os.listdir(short_folder), '*.npy')
    except:
        short_trajs = []
    try:
        extra_trajs = fnmatch.filter(os.listdir(extra_folder), '*.npy')
    except:
        extra_trajs = []

    #print how many were found
    print("Found {} long trajectories ".format(len(long_trajs)) +  
          "and {} short trajectories".format(len(short_trajs)) )


    #make list for all trajectories
    dtrajs = []

    #append trajectories from long and short sims to dtrajs 
    extractTrajectories(dtrajs, long_folder,  long_trajs,  stateDict)
    # extractTrajectories(dtrajs, short_folder, short_trajs, stateDict)
    # extractTrajectories(dtrajs, extra_folder, extra_trajs, stateDict)
    # for i in range(num_states):
    #     x = [i,62,62,62,62,62,62,62,62,62,62,62]
    #     y = [62, i,i,i,i,i,i,i,i,i,i,i,i]
    #     dtrajs.append(x)
    #     dtrajs.append(y)


    #set the lags to test. store the largest value for timescale testing purposes
    LAGS = [1,2,3,4,5,6,7,8,9,10]
    #LAGS = [25, 50, 75, 100, 125, 150, 175, 200]
    max_lag = LAGS[-1]

    #remove absorbing states
    #addedPairs = removeAbsorbing(dtrajs, max_lag)

    #print out the pairs of states with artficial transitions
    # for pair in addedPairs:
    #     print("Transition added between state {} and state {}".format(pair[0],pair[1]))

    #do time scale analysis to choose tau for MSM
    time_scale_analysis(dtrajs, LAGS)

    return


def buildMSM(traj_folder, lag_time):
    #construct an MSM using the trajectory data in the specfied folder

    #get the stateDict dictionary mapping
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

    #get the inverse mapping
    inv_map = {v: k for k, v in stateDict.items()}

    #get info about the state space and target
    num_states = len(stateDict)
    target_index = stateDict[(12,30,30)]
    print("Target state index is {}".format(target_index))
    print("There are {} states".format(num_states))

    #get the ensemble energy
    Ep = float(traj_folder.split('P')[1].split('H')[0])
    Eh = float(traj_folder.split('P')[1].split('H')[1].split('/')[0])

    #set the folders containing the trajectories. Include short folder for rachet trajs
    long_folder  = traj_folder
    #short_folder = long_folder + "short/"
    #short_folder = "../trajectories/E7.0S6.0/short/"
    #extra_folder = "../trajectories/E7.0S6.0/"

    #get all trajectory files
    long_trajs  = fnmatch.filter(os.listdir(long_folder), '*.npy')
    try:
        short_trajs = fnmatch.filter(os.listdir(short_folder), '*.npy')
    except:
        short_trajs = []
    try:
        extra_trajs = fnmatch.filter(os.listdir(extra_folder), '*.npy')
    except:
        extra_trajs = []

    #print how many were found
    print("Found {} long trajectories ".format(len(long_trajs)) +  
          "and {} short trajectories".format(len(short_trajs)) )


    #make list for all trajectories
    dtrajs = []

    #append trajectories from long and short sims to dtrajs 
    extractTrajectories(dtrajs, long_folder,  long_trajs,  stateDict)
    #dtrajs = dtrajs[100:500]
    # extractTrajectories(dtrajs, short_folder, short_trajs, stateDict)
    #extractTrajectories(dtrajs, extra_folder, extra_trajs, stateDict)
    # for i in range(num_states):
    #     x = [i,62,62,62,62,62,62,62,62,62,62,62]
    #     y = [62, i,i,i,i,i,i,i,i,i,i,i,i]
    #     dtrajs.append(x)
    #     dtrajs.append(y)
    #dtrajs.append([1334]*100 + [1332] + [1334] * 50)

    #remove absorbing states
    #addedPairs = removeAbsorbing(dtrajs, lag_time)

    #print out the pairs of states with artficial transitions
    # for pair in addedPairs:
    #     print("Transition added between state {} and state {}".format(pair[0],pair[1]))

    #generate MSM and test connectivity and thermodynamic properties
    msm = pyemma.msm.estimate_markov_model(dtrajs, lag=lag_time, reversible=False, 
                                           maxiter=5e5, maxerr=1e-10, count_mode='sliding',
                                           score_k=None, sparse=True, mincount_connectivity='1/n',
                                           weights='empirical')


    #save the msm
    with open("msm_test_out", 'wb') as f:
        pickle.dump(msm, f)

    #test loading the msm
    with open("msm_test_out", 'rb') as f:
        test = pickle.load(f)

    return







def count_tests():
    #perform tests with the count matrix from an msm

    #load in the test msm
    with open("msm7_fake", 'rb') as f:
        msm = pickle.load(f)

    #print some messages
    print("MSM loaded. The active set is:")
    print(msm.active_set)

    #get the count matrix
    counts = msm.count_matrix_full

    #loop over the rows, find ones with only the diagonal entry
    num_states = len(counts)
    for i in range(num_states):
        active_states = 0
        for j in range(num_states):
            state_count = counts[i][j]
            if state_count > 0:
                active_states += 1
                print("{} {}".format(i,j))

        if active_states <= 1:
            print("State {} interacts with {} states".format(i,active_states))


    #test trying to set a msm matrix entry - you can!
    print(msm.P[12][12])
    msm.P[12][12] = 0.8213
    print(msm.P[12][12])


    return

def manual_count_matrix(folder, lag):
    #manually construct a count matrix using the trajectories in folder

    #get the stateDict dictionary mapping
    with open("data/stateDict", 'rb') as f:
        stateDict = pickle.load(f)

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

    #normalize rows of the count matrix to get probability matrix
    for i in range(num_states):
        row = C[i]
        S = np.sum(row)
        if (S > 1e-6):
            P[i] = row / S
        else:
            P[i][i] = 1.0

    print(P[1642][1642])
    nz = 0
    for i in range(num_states):
        for j in range(num_states):
            if (P[i][j] > 1e-6):
                nz += 1

    print("{} nonzero entries out of {}, {} %".format(nz, num_states*num_states, float(nz)/float(num_states)))


    #return probability matrix
    return P







if __name__ == "__main__":

    folder = "../trajectories/P1.7H1.2/"
    #testMSM(folder)
    buildMSM(folder, 40)
    #manual_count_matrix(folder, 50)
    #count_tests()