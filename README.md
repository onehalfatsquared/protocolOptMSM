# protocolOptMSM

This repository contains all code for performing time-dependent protocol optimization via MSMs. The example system is the self-assembly of two types of conical subunits on the surface of a spherical nanoparticle. The tunable parameters are the interactions strengths between the subunit types. 

The simulation folder includes HOOMD scripts for performing various types of simulation of these subunits. Can perform long simulations, short simulations with starting states adaptively chosen, or disassembly trajectories started in specified states. 

The assembly folder includes functions for analyzing the HOOMD trajectories. We convert the gsd files with all coordinates to reaction coordinate trajectories. We then create a database of all unique reaction coordinates to index into discrete states. 

The MSM folder contains a custom MSM class for our trajectory data. It also contains files for constructing interpolants of the transition matrix entries, as well as evaluating those interpolants. Finally, it includes a driver function to perform the optimization over the global MSM as well as analyze the results. 

Note: optimization cannot be run directly from the entries in this repository. Trajectory data is several terabytes and is backed up on Brandeis hardware. The RBF interpolation uses a 3rd party library (https://rbf.readthedocs.io/en/latest/) that I have modified according to my needs here. This code is compiled locally on my machine and not included here, so the interpolation and optimization will not run. Will consider including this in future updates, or switching to scipy implementation. 