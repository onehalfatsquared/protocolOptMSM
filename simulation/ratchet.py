'''
Sets up a Database and State class to be used for a ratcheting procedure.

For each state, State stores the number of times it has been visited during simulations
as well as a collection of snap shots from that state

This data is then used to sample new trajectories using the infrequent states as starting
points for the new trajectories

Databases can be local or global. Local databases contain only the states seen with the 
given parameter set, whereas the global database compiles states seen from every parameter 
set. 

Warning: Do NOT use global. There were issues regarding overwriting the same coordinates
for each example configuration that led to biased sampling. This may not be as much of a 
problem after introducing the ratchet equilibration step, but it is not guaranteed to 
work as intended. 
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import inspect

#set a path to the assembly folder to do relative importing
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir) 
assembly_dir = parent_dir + "/assembly/"
sys.path.insert(0, assembly_dir) 

from pathwaysPy2 import getPathway
from pathwaysPy2 import convert2npy
from batch_get_paths import npy2dict

#set a path to the MSM folder to do relative importing
msm_dir = parent_dir + "/MSM/"
sys.path.insert(0, msm_dir) 

from manualMSM import loadStateDict
from ratchetMSM import ratchetByEigUncertainty
from ratchetMSM import getReactiveStates
from ratchetMSM import ratchetByCriticalNucleus
import gsd.hoomd
import numpy as np
import string
import random
import fnmatch
import pickle
import time

from simulate import simulateFromSnap
from config_dict import ConfigDict

'''
We define classes for:

1) A State, which includes a coarse grained description called pair using
some subset of the coordinates (Np, Nh, Bpp, Bph, Ih) to describe the state. Also stores
some number of snapshots for each of these states, to init simulations from. Tracks the 
number of times each state has been seen in simulation data, to inform which states to 
sample better. 

2) A Database, which stores all the relevant states and contains info on where they were
observed. Can search through this database by pair or index to access a state to get a 
random snapshot to use in simulation. 
'''

class State:

    #define max snaps in a state
    max_snaps = 10 #small-ish due to memory concerns

    #create a state class with pair (n_s, n_b) and a snapshot
    def __init__(self, pair_, snap_):
        #init all values
        self.snaps = []
        self.times_seen = 1

        #edit and set all values
        self.pair = pair_
        self.snaps.append(snap_)

    #add a snapshot and increment times seen
    def add_snap(self, snap):
        self.times_seen += 1
        if (len(self.snaps) < self.max_snaps):
            self.snaps.append(snap)
        else:
            #delete a random snap and replace it
            self.snaps.pop(random.randrange(self.max_snaps))
            self.snaps.append(snap)

    #return a random snapshot
    def get_random_snap(self):
        s = self.snaps[random.randrange(len(self.snaps))]
        return s



class Database:

    #create a database class
    def __init__(self):
        self.num_states  = 0
        self.states      = []
        self.seen_traj   = []
        self.seen_short  = []

    #search the database for a state by pair and return its index
    def search_by_pair(self, pair):

        idx = [self.states[i].pair for i in range(self.num_states)].index(pair)
        return idx

    #search the DB for state by pair, update the state, add if not found
    def update(self, pair, snap, short):

        found = any(pair == self.states[i].pair for i in range(self.num_states))

        if (found):
            i = self.search_by_pair(pair)
            self.states[i].add_snap(snap)
        else:
            if (not short): #dont add new states found by short sims
                self.states.append(State(pair, snap))
                self.num_states += 1

    #search the DB for a state, update, add state if not found
    def update_by_state(self, state, short):

        found = any(state.pair == self.state[i].pair for i in range(self.num_states))

        if (found):
            i = self.search_by_pair(state.pair)
            self.states[i].add_snap(snap)
        else:
            if (not short): #dont add new states found by short sims
                self.states.append(state)
                self.num_states += 1



    #get the number of times each state has been seen
    def get_observed_distribution(self):

        obs = [self.states[i].times_seen for i in range(self.num_states)]

        return obs

    #check if the given file has already been used for the db
    def already_seen(self, traj, short):

        if (short):
            if (traj in self.seen_short):
                return True
            else: 
                self.seen_short.append(traj)
        else:
            if (traj in self.seen_traj):
                return True
            else:
                self.seen_traj.append(traj)

        return False


#####################################################################
############## Auxilliary functions for database management ######### 
#####################################################################

def add_traj_to_db(folder, file, db, short):
    #add observations from the given file to the database

    #set the gsd and npy files
    gsd_file = folder + file
    traj_id = file.split('.')[0]
    npy_file = folder + traj_id + '.npy'

    #get a trajectory number
    traj_num = traj_id.split("traj")[1]

    #check if the path has been previously added
    seen = db.already_seen(traj_num, short)
    if (seen):
        #print progress update
        #print("Already seen file {}".format(npy_file))
        return

    #print progress update
    print("Now analyzing new file {}".format(npy_file))

    #get the path from the npy file
    try:
        path = np.load(npy_file)
        snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
        #take min length of these two (weird bug where gsd file was shorter than npy...)
        frames = min(len(path), len(snaps))
    except:
        return

    #get the first state in the path and update db
    p0 = [path[0][0], path[0][1], path[0][-1]]
    #db.update(p0, snaps.read_frame(0))

    #loop over the rest of the path, logging states when the path hops to a new state
    for i in range(1, frames):
        pair = path[i]
        p = [pair[0], pair[1], pair[-1]]

        #if the state changes between frames, log new state
        if (not p == p0):
            snap = snaps.read_frame(i)
            db.update(p, snap, short)

        #set p0 to be current state
        p0 = p

    return


def update_database(folder, db, db_name):
    #build a databse using the trajectories in the specified folder

    #get a list of gsd files - long and short
    traj_files = fnmatch.filter(os.listdir(folder), 'traj*.gsd')
    traj_files.sort()
    try:
        short_files = fnmatch.filter(os.listdir(folder + "short/"), 'traj*.gsd')
        short_files.sort()
    except:
        short_files = []
    print(short_files)
    
    c = 0
    #loop over the gsd files
    for file in traj_files:

        add_traj_to_db(folder, file, db, short=False)
        c += 1
        #uncomment this for quick building, for debug runs
        # if c > 10:
        #     break

    #loop over the short gsd files
    for file in short_files:

        add_traj_to_db(folder + "short/", file, db, short=True)


    #pickle the database for long term storage - avoid concurrency issues
    #first, pickle to a randomly generated file
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(16))
    random_temp_name = folder + random_string
    with open(random_temp_name, 'wb') as f:
        pickle.dump(db, f)
    print("Temporary database {} created".format(random_temp_name))

    #now check if there are any read files active in a loop
    check_times = 200
    for i in range(check_times):
        #find read files
        read_files = fnmatch.filter(os.listdir(folder), 'read*')

        #if none, change the name
        if len(read_files) == 0:
            os.rename(random_temp_name, folder + db_name)
            print("Temp database {} renamed to {}".format(random_temp_name, folder + db_name))
            break
        else:
            #wait for 10 seconds and try again
            print("The following read files were found:")
            print(read_files)
            print("Attempt {} occuring in 10 seconds".format(i))
            time.sleep(10)

    #check if time elapsed
    if i == check_times-1:
        #remove the temp database and exit
        print("The database could not be overwritten, deleting temp file...")
        os.remove(random_temp_name)
        return


    return


def load_database(folder, local, pid, build_option = None):
    #checks for the database file in the given folder. If not found, creates it
    #also allows for remaking if desired (e.g. if new trajectories are added)

    #set the name of the database of interest. if global, append g
    db_name = "stateDB"
    if (not local):
        db_name += "g"

    #Build the database from scratch
    if (build_option == 'scratch'):

        #init new database and build it
        db = Database()
        update_database(folder, db, db_name)

    #if not building from scratch, database should be loaded
    else:
        #try to load the database - implement a locking mechanism
        try:
            #create a file representing the current process is about to read the db
            read_file = folder + "read{}".format(pid) + '.txt'
            with open(read_file, 'x') as f:
                f.write("This is a read file")

            #wait a few seconds in case a write is happening
            time.sleep(3)

            #now open the pickled file
            with open(folder + db_name, 'rb') as f:
                db = pickle.load(f)

            #close by deleting the read file and printing a success message
            time.sleep(1)
            os.remove(read_file)
            print("Successfully loaded database and removed read_file")

        #if it fails, exit
        except Exception as e:

            #remove the read file and print error message
            read_file = folder + "read{}".format(pid) + '.txt'
            os.remove(read_file)
            print("Error in loading database. Removing the read_file")

            #if looking for global db, exit. You messed up, it should exist already
            if (not local):
                sys.exit("Error: The global database should already exist." + 
                         "Exiting...")

            #print the exception and exit
            print(e)
            raise()


    #check if the database should be updated
    if (build_option == 'update'):

        #just update the database
        update_database(folder, db, db_name)


    return db

def buildDB():
    #need to load database in the trajectory folder
    #read input files
    try:
        config = ConfigDict(sys.argv[1])  # Read input parameter file
    except:
        print("Usage: %s <input_file> ..." % sys.argv[0])
        raise

    # Get simulation parameters
    HH_attraction       = float(config['HH Bond Strength'])
    HP_attraction       = float(config['HP Bond Strength'])

    #define the folder with parameters
    folder = "../trajectories/P{}H{}/".format(HP_attraction, HH_attraction)
    db = load_database(folder, True, 0, build_option = 'scratch')

    return


#####################################################################
############## Auxilliary functions for state selection ############# 
#####################################################################

def obs_to_distribution(obs):
    #get normalized probability distribution from observation counts
    #probability is proportional to 1/count

    L = len(obs)
    pdf = np.zeros(L)

    Z = 0
    for i in range(len(obs)):
        pdf[i] = 1.0 / float(obs[i])
        Z += pdf[i]

    #normalize
    pdf /= Z

    return pdf

def sample_discrete_rv(p):
    #sample from the discrete distribution in p

    return np.random.choice(range(len(p)),p=p)


#####################################################################
############## Functions for database based ratcheting  ############# 
#####################################################################


def getDistribution(folder, db, local):
    #get a distribution to sample states from

    #determine how to do ratcheting 
    if local == 4:

        #ratchet by count based approach on states further along than crit nucleus
        pastCritical = ratchetByCriticalNucleus(folder)
        state_list   = []
        obs          = []
        for state in pastCritical:
            try: #if a state hasnt been saved to db yet, this will give keyerror
                index =  db.search_by_pair(list(state))
                state_list.append(index)
                obs.append(db.states[index].times_seen)
                #print(state, obs[-1])
            except:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(state, exc_type, fname, exc_tb.tb_lineno)
                continue

        pdf = obs_to_distribution(obs)

    if local == 3:

        #ratchet by count based approach on reactive pathways
        reactiveStates = getReactiveStates(folder)
        state_list = []
        obs        = []
        for state in reactiveStates:
            try: #if a state hasnt been saved to db yet, this will give keyerror
                index =  db.search_by_pair(list(state))
                state_list.append(index)
                obs.append(db.states[index].times_seen)
                #print(state, obs[-1])
            except:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                continue

        pdf = obs_to_distribution(obs)


    if local == 2:

        #ratchet by MSM uncertainty
        state_list, pdf = ratchetByEigUncertainty(folder)

        #check for failure
        if (type(state_list) == int):
            print("Falling back to count based ratchet")
            local = 1

    #check if count based ratcheting should be used (requested or fall-back)
    if local == 1:
        #get the observations distribution
        obs = db.get_observed_distribution()

        #create a probability distribution to sample infrequent states (prop to 1/obs)
        state_list = range(len(obs))
        pdf = obs_to_distribution(obs)

    return state_list, pdf, local


def getState(folder, db, localCopy, inv_map):
    #get a state from the database to start ratcheting

    #set local to the desired value for each attempt at ratcheting
    local = localCopy

    #get a distribution to sample starting states from, based on local value
    state_list, pdf, local = getDistribution(folder, db, local)

    #sample the probability distribution to get a raw state
    raw_state  = sample_discrete_rv(pdf)
    print(raw_state)

    #convert raw state to msm state using the state_list
    true_state = state_list[raw_state]
    print(true_state)

    #if local is 2, the index needs to be converted to the database index
    if local == 2:

        #get the state discretization
        state_disc = inv_map[true_state]
        print(state_disc)

        #search the database for this state
        ic_state = db.search_by_pair(list(state_disc))
        print(ic_state)

        #print a message showing the chosen initial state
        print("Sampling from state: {}".format(state_disc))

    else: #the index is already a database index
        ic_state = true_state

        print("Sampling from state: {}".format(db.states[ic_state].pair))

    return ic_state

def perform_ratchet(folder, db, ic_state, local = True):
    #sample a short trajectory starting from a configuration not well sampled

    #get a random snapshot from this state
    snap = db.states[ic_state].get_random_snap()

    #get a random seed
    seed = random.randint(0,10000000)

    #simulate
    outfile = simulateFromSnap(snap, seed)

    #convert the outfile name to the form trajX.gsd
    gsdf = convertFilename(outfile, folder)

    #get path from the trajectory
    pathway = getPathway(gsdf, verbose=True)

    #convert to npy file
    convert2npy(gsdf, pathway)

    return


def baseRatchet(folder, repeats, local, pid):
    #the original basic ratcheting code. Uses a database to determine ratchet states
    #according to some criteria

    #get the stateDict dictionary mapping, and its inverse map
    stateDict = loadStateDict()
    inv_map = {v: k for k, v in stateDict.items()}

    #get the database from the folder
    db = load_database(folder, local, pid)

    #print out system variables
    print(repeats, local)

    #explicitly check the local condition
    if local == 0:
        localB = False
    else:
        localB = True

    #make a copy of the local variable to reset after each ratchet attempt
    localCopy = local

    #perform the desired number of ratcheting steps
    successes = 0
    attempts  = 0
    while successes < repeats:
        try:

            #increment attempts
            attempts += 1

            #set local to the desired value for each attempt at ratcheting
            local = localCopy

            #get a state to init with
            ic_state = getState(folder, db, localCopy, inv_map)

            #perform the ratchet from this state
            perform_ratchet(folder, db, ic_state, localB)

            #increment successes
            successes += 1

        except Exception as e:
            if type(e) == KeyError:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print("Error. A key was not found. Trying to recreate dictionary.")
                npy2dict(folder)
                stateDict = loadStateDict()
                inv_map = {v: k for k, v in stateDict.items()}
            else:
                print(e)
                print("An error occurred during ratchet {}. Moving to next trial.".format(attempts))
            
            continue

        #break out if it keeps failing
        if attempts > repeats * 2:
            break

    #update the database
    update_database(folder, db, 'stateDB')

    return

#####################################################################
############## Functions for snap based ratcheting ################## 
#####################################################################

def defineMetastable():
    #return a list of metastable states that can be used to init ratchet trajs

    metastable = []
    metastable.append((12,30,30))
    metastable.append((12,31,29))
    metastable.append((12,31,27))
    metastable.append((12,30,26))
    metastable.append((12,29,27))

    return metastable

def getSnap(local, pid):

    #if local is 5, just pick one of the metastable states defined
    if local == 5:

        #get list of metastable states
        metastable = defineMetastable()
        num_meta = len(metastable)
        if num_meta > 0:
            choice = random.randint(0, num_meta-1)
            state_disc = metastable[choice]
        else:
            print("Error: Metastable states not defined. Cannot proceed")
            raise()

        #display which state is being used to init
        print("Sampling from state: {}".format(state_disc))

        #load the snap from on disk
        a,b,c = state_disc[0], state_disc[1], state_disc[2]
        snap_file = "../simulation/snaps/state{}_{}_{}".format(a,b,c)
        with open(snap_file, 'rb') as f:
            snap = pickle.load(f)

    #if local is 6, we are sweeping across all states. use PID to determine which
    if local == 6:

        #load the database of snaps
        with open("../MSM/snapDict", 'rb') as f:
            snapDict = pickle.load(f)

        #load a state database
        stateDict, dummy = loadStateDict(refine=True)
        inv_map = {v: k for k, v in stateDict.items()}

        #get a list of keys
        all_keys = list(snapDict.keys())

        #get the key corresponding to PID and extract a snap
        this_key = all_keys[pid]
        snap = snapDict[this_key]
        print("Extracted snapshot for state {}".format(inv_map[this_key]))

    return snap

def perform_ratchet_snap(folder, snap):
    #sample a short trajectory starting from a configuration not well sampled

    #get a random seed
    seed = random.randint(0,10000000)

    #simulate
    print("Simulating for {} lags at lag time {}".format(num_lags, LAG))
    outfile = simulateFromSnap(snap, seed, LAG=LAG, num_lags=num_lags)

    #convert the outfile name to the form trajX.gsd
    gsdf = convertFilename(outfile, folder)

    #get path from the trajectory
    pathway = getPathway(gsdf, verbose=True)

    #convert to npy file
    convert2npy(gsdf, pathway)

    return

def newRatchet(folder, repeats, local, pid):
    #the newer ratcheting techniques. These get a snapshot through some means and 
    #init a simulation based on that

    #get a snap 
    snap = getSnap(local, pid)

    #perform the desired number of ratcheting steps
    successes = 0
    attempts  = 0
    while successes < repeats:
        try:

            #increment attempts
            attempts += 1

            #perform the sim
            perform_ratchet_snap(folder, snap)
            successes += 1

        except Exception as e:
            print(e)
            print("An error occurred during ratchet {}. Moving to next trial.".format(attempts))
            continue

        #break out if it keeps failing
        if attempts > repeats * 3:
            break

    return

#####################################################################
############## Helper functions and launcher scripts    ############# 
#####################################################################


def convertFilename(out0, folder):
    #converts the outfile name written by hoomd to a trajX file by availability

    #append a filename by looking at existing files, and incrementing by 1
    short_folder = folder + "short/"
    #check if the folder exists first
    if (not os.path.exists(short_folder)):
        os.makedirs(short_folder)
    traj_files = fnmatch.filter(os.listdir(short_folder), 'traj*')
    used = []
    #loop over all ratchet traj numbers
    for file in traj_files:
        traj_id = file.split('.')[0]
        num    = traj_id.split("traj")[1]
        used.append(int(num))

    #check if the used_seeds list is non-empty (i.e. first trajectory)
    if (len(used) > 0):
        M = max(used)
        num = M+1
    else:
        num = 0

    #change the name of the outfile to "traj{num}.gsd", same for log file
    file_path = out0.split('/')
    file_path.pop()
    file_path = '/'.join(file_path) + '/'
    gsdf = file_path + 'traj' + str(num) + '.gsd'
    logf = file_path + 'traj' + str(num) + '.log'
    os.rename(out0, gsdf)
    lpath = out0.split('.gsd')[0]
    os.rename(lpath + '.log', logf)

    #report the name change
    print("Changed file {} to {}".format(out0, gsdf))

    #return the gsd filename
    return gsdf


if __name__ == '__main__':

    #read input files
    try:
        config  = ConfigDict(sys.argv[1])  # Read input parameter file
        repeats = int(sys.argv[2])
        local   = int(sys.argv[3])
        pid     = int(sys.argv[4])
    except:
        print("Usage: {} <input_file> <num_repeats> <local> <pid>".format(sys.argv[0]))
        raise

    #if num_repeats == -1, simply make the database and exit
    if repeats == -1:
        buildDB()
        print("Database succesfully built. Exiting...")
        sys.exit()

    # Get simulation parameters
    HH_attraction       = float(config['HH Bond Strength'])
    HP_attraction       = float(config['HP Bond Strength'])

    #define the folder with parameters
    folder = "../trajectories/P{}H{}/".format(HP_attraction, HH_attraction)

    global LAG, num_lags
    LAG = 125
    num_lags = 3

    #call the correct ratcheting function 
    if local < 5:
        baseRatchet(folder, repeats, local, pid)
    else:
        newRatchet(folder, repeats, local, pid)

    


    
    