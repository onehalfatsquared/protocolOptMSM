'''
This script reads in a saved result of the protocol optimization and performs various 
operations on it. 

Main intent is for evaluating the probability dynamics for a given protocol (whether saved
and read or manually input), without having to recompute it from the optimization every 
time. 

Also includes functions to contruct different kinds of plots with the data. 
Plot probabilities as a function of time. Plot the protocols themselves. 
Plot the feasible domain, with the option of including a time dependent representation
of the given protocol on the feasible domain.
'''

import sys
import os
import fnmatch
import inspect

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse

import pandas as pd
import seaborn as sns

import pickle
import time

#import modules to do calculations on polygons
import shapely
import alphashape
from descartes import PolygonPatch

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

#import the MSM class object
from manualMSM import MSM
from manualMSM import computeSampledTargetProbability
from manualMSM import computeCommittor
from manualMSM import computeMFPT
from manualMSM import computeMFPTsampling

#import functions for loading MSMs and getting active set info
from interpolateMSM import loadMSMs
from interpolateMSM import getGlobalActive

#import shelve for reading the cache on hdd
import shelve   

#import functions from protocolOpt to init system and evaluate probabilites from MSM
import protocolOpt as opt

########################################################################
##################### Plotting Functions ###############################
########################################################################

def plot_colorline(x,y,c):
    #plot a curve in (x,y) with color corresponding to c

    #normalize values in c and assign colormap values
    c = plt.cm.cool((c-np.min(c))/(np.max(c)-np.min(c)))

    #plot each segement on the axes according to its color
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[len(x)-2-i])

    return


def plotProtocol(time, E_star, max_index=None, E_samples=None):
    #plot the protocol as a function of time

    #init as figure 1
    fig1 = plt.figure(1)
    ax1  = fig1.add_subplot(111)

    #check if optimal constants are desired in the plot
    if max_index is not None:

        #plot time-dep and constant protocols. Dashed for constants
        c0 = np.ones(len(time)) * E_samples[max_index][0]
        c1 = np.ones(len(time)) * E_samples[max_index][1]
        ax1.plot(time, E_star[0], 'b-', time, c0, 'b--', 
                 time, E_star[1], 'r-', time, c1, 'r--')
        # ax1.legend(["H-P Strength", "H-P Strength Const", "H-H Strength", "H-H Strength Const"],
        #         prop={'size':16})

    else:
        #just plot time dependent protocols
        ax1.plot(time, E_star[0], time, E_star[1])
        # ax1.legend(["H-P Strength", "H-H Strength"], prop={'size':16})

    #set axis labels and fontsizes
    # ax1.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    num_digits = 5
    ax1.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax1.set_ylabel("Subunit Attraction", fontsize = 20)

    #set number of ticks in each dimension
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    #reduce whitespace and plot
    plt.tight_layout()
    plt.show()

    return

def plotTargetDynamics(time, prob_star, target, pt_const_max=None):
    #plot probability as fn of time for the given states

    #init as figure 2
    fig2 = plt.figure(2)
    ax2  = fig2.add_subplot(111)

    #check if this is being plotted against an optimal constant
    if pt_const_max is not None:
        ax2.plot(time, prob_star[:, target], time, pt_const_max, linewidth=2.0)
        ax2.legend(["Optimal Protocol", "Optimal Constant"],prop={'size':16})
    else:
        ax2.plot(time, prob_star[:, target], linewidth=2.0)

    #set axis labels
    # ax2.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    num_digits = 5
    ax2.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax2.set_ylabel("Target Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()

def plotFeasibleDomain(time, parameters, E_samples=None, prob_samples=None, E_star=None):
    #plot the domain, nodes, interpolated MSM taret probability
    #include time-dep protocol if desired

    #init as figure 3
    fig3 = plt.figure(3)
    ax3  = fig3.add_subplot(111)

    #plot the target probability on the sampled values if provided
    if (E_samples is not None and prob_samples is not None):
        samples_x = np.array([entry[0] for entry in E_samples])
        samples_y = np.array([entry[1] for entry in E_samples])
        


        plt.tricontourf(samples_x, samples_y, prob_samples, levels = 50)
        cbar = plt.colorbar()
        cbar.ax.locator_params(nbins=5)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label("Target Probability, T={}e5".format(time[-1]),rotation=270, fontsize=18,labelpad=18)
        
        max_index = prob_samples.argmax()
        opt_const = [tuple(E_samples[max_index])]
        ax3.scatter(*zip(*opt_const), color='red')

    #create the feasible domain
    nodes = [tuple(entry) for entry in parameters]
    # boundary = alphashape.alphashape(nodes,0)
    ax3.scatter(*zip(*nodes), color='black')
    # ax3.add_patch(PolygonPatch(boundary, alpha=0.2, facecolor=None, edgecolor='black'))

    coords = np.array([[1.2,1.5], [1.2,1.4], [1.4,1.4], [1.6,1.2], [1.7,1.2], [1.8,1.3], \
                       [1.8,1.4], [1.7,1.5], [1.2,1.5]])
    polygon = Polygon(coords, True)
    p = PatchCollection([polygon], alpha=0.2, facecolor=None, edgecolor='black')
    # ax3.add_collection(p)
    # ax3.add_patch(PolygonPatch(boundary, alpha=0.2, facecolor=None, edgecolor='black'))

    #plot the time dependent protocols if desired 
    if (E_star is not None):
        plot_colorline(E_star[0], E_star[1], time)

    #format the plots
    ax3.set_xlim([1.29, 1.91])
    ax3.set_ylim([1.15, 1.55])
    ax3.set_xlabel("$E_{HP}$", fontsize = 18)
    ax3.set_ylabel("$E_{HH}$", fontsize = 18)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return


def plotResults(tau, animation_period, T, E_star, prob_star, global_t, parameters,
                E_samples=None, max_index=None, pt_const_max=None, prob_samples=None):

    #set real time
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]

    #append the final value - not used but for plotting purposes
    E0 = np.append(E_star[0], E_star[0][-1])
    E1 = np.append(E_star[1], E_star[1][-1])
    E_star = np.array([E0,E1])

    #plot 1 - Protocols vs time
    plotProtocol(time, E_star, max_index, E_samples)

    #plot 2 - fc dynamics for target state
    plotTargetDynamics(time, prob_star, global_t, pt_const_max)

    #plot 3 - Feasible domain, target prob, and optimal protocol
    # plotFeasibleDomain(time, parameters, E_samples, prob_samples) #no protocol
    plotFeasibleDomain(time, parameters, E_samples, prob_samples, E_star) #with protocol

    return

########################################################################
##################### Analysis Functions ###############################
########################################################################

def analyze_protocol(T, tau, animation_period):
    #analyze the protocol generated from the optimization by plotting the notable
    #probabilities as a fn of time for the protocol and the optimal constant

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    #load the protocol
    with open("data/protocolF", 'rb') as f:
        E_star = pickle.load(f)
        print("Protocol loaded.")

    #set the constant protocol
    E_c = [1.35, 1.475]
    protocol_c = [np.ones(T) * E_c[0], np.ones(T) * E_c[1]]

    #plot the protocol
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]
    plotProtocol(time[0:T], E_star, 0, [E_c])

    #get probability of all states as fn of time
    p = opt.getProbAllStates(E_star, T, nsa, nsg, global_i, cache_flag, Pcache, disc)
    p_c = opt.getProbAllStates(protocol_c, T, nsa, nsg, global_i, cache_flag, Pcache, disc)

    #get the notable states
    notable = []
    for i in range(nsg):
        if p[-1,i] > 0.02 or p_c[-1,i] > 0.02:
            state_full = globalActive[i]
            print("State: {}, Prob: {}, cProb: {}".format(inv_map[state_full], p[-1,i], p_c[-1,i]))
            if inv_map[state_full][0] > 11:
                notable.append(i)

    #make a plot with the time dependence of the probabilities
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    legend_list = []
    for state in notable:
        ax.plot(time, p[:,state])
        legend_list.append(str(inv_map[globalActive[state]]))

    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("State Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(legend_list)
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]
    legend_list = []
    for state in notable:
        ax2.plot(time, p_c[:,state])
        legend_list.append(str(inv_map[globalActive[state]]))

    num_digits = 5
    ax2.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax2.set_ylabel("State Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.legend(legend_list)
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return 0


def compareMSMtoSampling(folder):
    #compare trajectory data sampled w/ optimal protocol to MSM estimates

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    target_state = (12,30,30)

    #load the protocol
    with open("data/protocolF", 'rb') as f:
        E_star = pickle.load(f)
        print("Protocol loaded.")

    #get probability of all states as fn of time
    p = opt.getProbAllStates(E_star, T, nsa, nsg, global_i, cache_flag, Pcache, disc)
    target_prob = p[:,global_t]
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]

    #get the sampled probability
    time_S, target_prob_S, samples = computeSampledTargetProbability(folder, target_state, 
                                                            animation_period)
    time_S /= 1e5

    #plot them together
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    #plot data
    ax.plot(time, target_prob, time_S, target_prob_S, linewidth=2.0)
    ax.legend(["MSM Estimate", "Sampling Estimate"],prop={'size':16})

    #set axis labels
    # ax.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("Target Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()


def compareSampling(folder1, folder2):
    #compare yields via sampling for two parameter sets

    #define target 
    target_state = (12,30,30)

    #get the sampled probabilities
    time_1, target_prob_1, samples = computeSampledTargetProbability(folder1, target_state, 
                                                            animation_period)
    time_1 /= 1e5

    time_2, target_prob_2, samples = computeSampledTargetProbability(folder2, target_state,
                                                            animation_period)
    time_2 /= 1e5

    #plot them together
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    #plot data
    ax.plot(time_1, target_prob_1, time_2, target_prob_2, linewidth=2.0)
    ax.legend([folder1, folder2],prop={'size':16})

    #set axis labels
    # ax.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("Target Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()


def compareAll(opt_folder, comp_folder):
    #compare the MSM estimate and sampling, along with the best constant sampling

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    target_state = (12,30,30)

    #load the protocol
    with open("data/protocolF", 'rb') as f:
        E_star = pickle.load(f)
        print("Protocol loaded.")

    #set the constant protocol - actual is [1.35, 1.475]
    E_c = [1.35, 1.475]
    protocol_c = [np.ones(T) * E_c[0], np.ones(T) * E_c[1]]

    #get probability of all states as fn of time
    p   = opt.getProbAllStates(E_star, T, nsa, nsg, global_i, cache_flag, Pcache, disc)
    p_c = opt.getProbAllStates(protocol_c, T, nsa, nsg, global_i, cache_flag, Pcache, disc)

    target_prob   = p[:,global_t]
    target_prob_c = p_c[:,global_t]
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]

    #get the sampled probability
    time_S, target_prob_S, samples = computeSampledTargetProbability(opt_folder, target_state, 
                                                            animation_period)
    time_S /= 1e5

    time_C, target_prob_C, samples = computeSampledTargetProbability(comp_folder, target_state,
                                                            animation_period)
    time_C /= 1e5

    #de-noise the sampling data?
    #todo

    #plot them together
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    #plot data
    ax.plot(time, target_prob,   time_S, target_prob_S,  linewidth=2.0)
    ax.plot(time, target_prob_c, time_C, target_prob_C,  linewidth=2.0)

    #set axis labels
    # ax.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("Target Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()

def compareProtPhases():
    #make a plot to show how the protocol behaves in each phase

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    #load the protocol
    with open("data/protocol1", 'rb') as f:
        E_star = pickle.load(f)
        print("Protocol loaded.")

    #set phase 1 protocol
    E_1 = [1.5, 1.5]
    protocol_1 = [np.ones(T) * E_1[0], np.ones(T) * E_1[1]]

    #set phase 2 protocol
    E_2 = [1.34, 1.5]
    protocol_2 = [np.ones(T) * E_2[0], np.ones(T) * E_2[1]]

    #get probability of all states as fn of time
    p   = opt.getProbAllStates(E_star, T, nsa, nsg, global_i, cache_flag, Pcache, disc)
    p_1 = opt.getProbAllStates(protocol_1, T, nsa, nsg, global_i, cache_flag, Pcache, disc)
    p_2 = opt.getProbAllStates(protocol_2, T, nsa, nsg, global_i, cache_flag, Pcache, disc)

    target_prob   = p[:,global_t]
    target_prob_1 = p_1[:,global_t]
    target_prob_2 = p_2[:,global_t]
    print(target_prob_2[-1])
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]

    #plot them together
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    #plot data
    ax.plot(time, target_prob,   linewidth=2.0)
    ax.plot(time, target_prob_1,  linewidth=2.0)
    ax.plot(time, target_prob_2,  linewidth=2.0)

    #set axis labels
    # ax.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("Target Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()



def findTransitions(folder):
    #search through all npy files in a folder to determine when specified states break
    #or form

    #get all npy files in the given folder
    trajs = fnmatch.filter(os.listdir(folder), '*.npy')

    #if none are found, raise an error
    if (len(trajs) == 0):
        print("No npy files were found in {}".format(folder))
        raise()

    #place all target states of interest in a dictionary
    #lists are - first hit time, all reactive break times, all reactive form times 
    bfTimes = dict()
    bfTimes[(12,30,30)] = [[],[],[]]
    bfTimes[(12,30,26)] = [[],[],[]]
    bfTimes[(12,31,27)] = [[],[],[]]
    bfTimes[(12,29,27)] = [[],[],[]]
    bfTimes[(12,31,29)] = [[],[],[]]

    #get a list for these keys for quick checking
    all_states = list(bfTimes.keys())

    #add a dummy state for algorithmic convenience
    bfTimes[(0,0,0)]    = [[],[]]

    #scale times by lag
    ts = 1.0 / 125.0

    #define a minimum number of lags for a state to exist for stability
    stable = 4

    #loop over trajectories
    for npy_file in trajs:

        #load the npy file
        try:
            #print(npy_file)
            path = np.load(folder+npy_file)
        except:
            continue

        #init variables to track what state we were previously in
        lastState = (0,0,0)
        lastTime  = 0

        #loop over the path
        for i in range(len(path)):

            #get a state from the npy file 
            pair = path[i]
            state = (pair[0], pair[1], pair[-1])

            #check if this state is in our list
            hit = (state in all_states)
            if (hit):

                #check if the hit is stable by looking in the future
                skip = False
                for test in range(i+1, i+1+stable):
                    if test < len(path):
                        pairTest = path[test]
                        stateTest= (pairTest[0], pairTest[1], pairTest[-1])

                        if state != stateTest:
                            #unstable hit, set skip flag
                            skip = True
                            break

                #skip this hit if the flag is tripped
                if (skip):
                    continue

                #check if this is the first hit
                if (lastTime == 0):

                    #append this time to the first hit list
                    bfTimes[state][0].append(i * ts)

                    #update the lastState and lastTime
                    lastState = state
                    lastTime  = i

                #check if the current state is the same as the last
                if (state == lastState):

                    #update the last time to this time step
                    lastTime = i

                else:
                    #we are in different state -> transition occurred

                    # #update the break time of the lastState
                    # bfTimes[lastState][1].append((lastTime+stable) * ts)

                    # #update form time of the current state
                    # bfTimes[state][2].append(i * ts)

                    if state == (12,30,30):
                        print("{} -> {}, file:{}, breakT={}, formT={}".format( \
                             lastState, state, npy_file, i, lastTime))

                        #update the break time of the lastState
                        bfTimes[lastState][1].append((lastTime+stable) * ts)

                        #update form time of the current state
                        bfTimes[state][2].append(i * ts)

                    #set lastState to state. same for time
                    lastState = state
                    lastTime = i

    #pickle the result so it can be loaded later
    location = folder.split("/")[-2]
    with open("data/" + location + "transitions", 'wb') as f:
            pickle.dump(bfTimes, f)
            print("Transition data pickled.")

    return bfTimes

def analyzeTransitions(folder):
    #analyze the time distibution of transition data for simulations in a given folder

    #check if the distributions have already been computed
    location = folder.split("/")[-2]
    try:
        with open("data/" + location + "transitions", 'rb') as f:
            bfTimes = pickle.load(f)
            print("Transitions found and loaded.")
    except:
        print("Transition data not found. Computing from scratch")
        bfTimes = findTransitions(folder)

    all_transitions = []
    all_transitions.append(bfTimes[(12,30,26)][1])
    all_transitions.append(bfTimes[(12,29,27)][1])
    all_transitions.append(bfTimes[(12,31,29)][1])
    all_transitions.append(bfTimes[(12,31,27)][1])

    #make a histogram of one of the measured distributions
    fig, axs = plt.subplots(1, 1,
                            tight_layout = True)
    # counts, bins = np.histogram(bfTimes[(12,30,26)][1],100)
    #axs.hist(bins[:-1], bins, weights=1.29*counts/np.max(counts))

    #add hist data
    plt.hist(all_transitions, 50, stacked=True)

    axs.set_xlabel("Lagtimes", fontsize = 20)
    axs.set_ylabel("Transition Counts", fontsize = 20)

    #set num ticks
    # plt.locator_params(axis='y', nbins=5)
    # plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)

    axs.set_ylim(0,8)

    #make a second axis
    ax2 = axs.twinx()
    ax2.set_ylabel("Protocol Values", fontsize = 20)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylim(0,1.6)
    plt.locator_params(axis='y', nbins=6)

    #load the protocol
    with open("data/protocolF", 'rb') as f:
        E_star = pickle.load(f)
        print("Protocol loaded.")

    #plot protocol
    ax2.plot(range(len(E_star[0])), E_star[0], 'b')
    ax2.plot(range(len(E_star[1])), E_star[1], 'r')
     
    # Show plot
    plt.show()   

    return  


def getEqProb(e1, e2):
    #get the equilibrium probability of the target with the specified parameters

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    #set parameters
    E = [e1, e2]

    #get the transition matrix
    P = opt.get_matrix_cached(E, nsa, nsg, Pcache, disc)
    P = P[0]

    #get the left eigenvector, print to see if there is an eigenvalue of 1
    eigsL, vecsL = scipy.sparse.linalg.eigs(P.transpose(), k=5, which="LR")
    print(eigsL)

    #Gather the corresponding eigenvctor into an array
    print(len(vecsL), nsg, nsa)
    pi = np.zeros(nsg)
    for i in range(nsg):
        pi[i] = np.real(vecsL[i][0])

    #normalize it
    pi = pi / pi.sum()

    #get the large components
    for i in range(len(vecsL)):
        if pi[i] > 0.03:
            state_full = globalActive[i]
            state = inv_map[state_full]
            print("State {}, EqProb: {}".format(state, pi[i]))

    #define the states of interest
    important_states = []
    important_states.append((12,30,60,60,26))
    important_states.append((12,31,29))
    important_states.append((12,31,27))
    important_states.append((12,30,30))
    important_states.append((12,29,60,57,27))

    #convert them to global indices
    global_states = []
    for state in important_states:

        index = stateDict[state]
        global_index = np.where(globalActive == index)[0][0]
        global_states.append(global_index)


    #get probability of all states as fn of time #eq uses T=16000
    T = 16000
    tau = 125
    animation_period = 25
    # protocol = [np.ones(T) * E[0], np.ones(T) * E[1]]
    # p   = opt.getProbAllStates(protocol, T, nsa, nsg, global_i, cache_flag, Pcache, disc)

    #set the initial condition
    num_states = nsg
    p0 = np.zeros(num_states, dtype=float)
    p0[global_i] = 1.0

    #init storage for probabilities for all time and set ic
    probs = np.zeros((T+1, num_states), dtype=float)
    probs[0,:] = p0

    #iteratively multiply by transition matrix
    for i in range(T):
        probs[i+1,:] = probs[i,:] * P

    #loop over states on interest, print eq prob, plot dynamics
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]
    for i in range(len(important_states)):

        state_g = global_states[i]
        target_prob   = probs[:, state_g]
        plt.plot(time, target_prob)
        print("State {}, EQ prob: {}".format(important_states[i], target_prob[-1]))

    num_digits = 5
    ax.set_xlabel(r"t/$10^{}t_0$".format(num_digits), fontsize = 20)
    ax.set_ylabel("State Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()

    return


def protCompare():
    #compare the yield curves using two different protocols

    #load protocols
    with open("data/protocol1", 'rb') as f:
        E_1 = pickle.load(f)
        print("Protocol 1 loaded.")

    with open("data/protocol1_flatter", 'rb') as f:
        E_2 = pickle.load(f)
        print("Protocol 2 loaded.")

    testPlot = False
    if (testPlot):
        t1 = np.linspace(0,1,len(E_1[0]))
        t2 = np.linspace(0,1,len(E_2[0]))

        plt.plot(t1, E_1[0], t1, E_1[1])
        plt.plot(t2, E_2[0], t2, E_2[1])
        plt.show()

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    #set the additional states to look at
    defect_state  = (12,30,26)
    defect_index  = stateDict[defect_state]
    global_d = np.where(globalActive == defect_index )[0][0]


    #get probability of all states as fn of time
    p1 = opt.getProbAllStates(E_1, T, nsa, nsg, global_i, cache_flag, Pcache, disc)
    p2 = opt.getProbAllStates(E_2, T, nsa, nsg, global_i, cache_flag, Pcache, disc)

    #get yield curves
    target_prob1 = p1[:,global_t]
    target_prob2 = p2[:,global_t]
    defect_prob1 = p1[:,global_d]
    defect_prob2 = p2[:,global_d]
    time = [i * tau * animation_period / 1e5 for i in range(T+1)]

    #plot them together
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    #plot data
    ax.plot(time, target_prob1, linewidth=2.0)
    ax.plot(time, target_prob2, linewidth=2.0)
    ax.plot(time, defect_prob1, linewidth=2.0)
    ax.plot(time, defect_prob2, linewidth=2.0)

    #set axis labels
    ax.set_xlabel(r"Time / $10^5 t_0$", fontsize = 20)
    ax.set_ylabel("Target Probability", fontsize = 20)

    #set num ticks
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=4)

    #set tick label sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #remove whitespace and plot
    plt.tight_layout()
    plt.show()

    return 0


def tptCalcs(e1, e2):

    #get parameters needed to run MSM calcs
    E2, globalActive, nsg, nsa, stateDict, inv_map, boundary, disc, \
    spacing, cache_flag, Pcache, global_i, global_t \
    = opt.msmSetup()

    #set parameters
    E = [e1, e2]

    #get the transition matrix
    P = opt.get_matrix_cached(E, nsa, nsg, Pcache, disc)
    P = P[0]

    #define the initial and target states
    important_states = []
    important_states.append((12,30,60,60,26))
    important_states.append((12,31,29))
    important_states.append((12,31,27))
    important_states.append((12,30,30))
    important_states.append((12,29,60,57,27))

    #convert them to global indices
    global_states = []
    for state in important_states:

        index = stateDict[state]
        global_index = np.where(globalActive == index)[0][0]
        global_states.append(global_index)

    #do tpt to get a collection of states defining a crit nuc
    crit_states = []
    x = computeCommittor(P, global_i, global_states, nsg)
    for i in range(nsg):
        if x[i] > 0.6 and x[i] < 1:
            state = inv_map[globalActive[i]]
            print(i, state, x[i])
            crit_states.append(i)

    #solve mfpt problems
    t1 = computeMFPTsampling(P, global_i, crit_states, nsg, inv_map, globalActive)
    t2 = computeMFPTsampling(P, crit_states, global_states, nsg, inv_map, globalActive)
    print(t1, t2)


    return



if __name__ == '__main__':

    #set parameters
    animation_period = 25.0    #time between measurements for MSM
    tau              = 125     #lag time for MSM creation
    T                = 256     #jumps of 2500 with final time 500000 -> 200 jumps
                               # 800000k is 320 jumps

    global NUM_PROCS
    NUM_PROCS = 1

    global PARAM
    PARAM = 2

    #compare optimal protocol dynamics to optimal constant
    # analyze_protocol(T, tau, animation_period)

    #compare MSM estimate dynamics to simulation estimates
    # folder = "../trajectories/optF/"
    # compareMSMtoSampling(folder)

    #compare just the sampling results
    # compareSampling('../trajectories/P1.35H1.475/', '../trajectories/P1.3H1.5/')

    #make plot comparing both optimal protocols to sampling estimates
    # compareAll('../trajectories/optF/', '../trajectories/P1.35H1.475/')

    #make plot showing probs for the 2 intro phases of the optimal td prot
    #compareProtPhases()

    #get the equilibrium probabilities of important states
    # getEqProb(1.35, 1.475)

    #compare two protocols
    #protCompare()

    #get stats on break and form times of stable states
    # analyzeTransitions('../trajectories/P1.35H1.475/')
    analyzeTransitions('../trajectories/optF/')

    #do tpt calculations to study nucleation time
    # tptCalcs(1.35,1.475)
    # tptCalcs(1.5,1.5)
