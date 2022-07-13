Information for files in this folder:

pyemma contains the pyemma code for constructing MSMs, and the resulting MSMs saved using the pyemma objects to create global transition matrices. This was the initial testing code, but the rest of the pipeline does not support this format, so use the custom class. 

manualMSM contains my custom class for constructing Markov State Models from trajectory info. 

ratchetMSM contains functions to determine the best starting states for a ratcheting algorithm to sample new trajectories. Some require only thermodynamic info, some require pathways, and some require the construction of an MSM. 

parameterMap.txt contains a list of all parameter sets that were sampled. It also includes identifiers for what kind fo sampling was performed. L denotes long simulations starting from an empty nanoparticle. S denotes short simulations, performed with the racheting procedure. D denotes disassembly simulations starting in each of the 5 common end states. N denotes null; either no sampling was done, or this data set is being excluded from the feasible set. 

interpolateMSM contains code for constructing RBF interpolant objects for each entry of a sparse transition matrix. There is an optimization over shape parameter that occurs here, simply by sampling. Each entry is fit in parallel and placed in a dict structure where the keys are the matrix component. 

interpolantEval takes a parameter value and re-constructs the transition matrix by evaluating all the interpolant objects. This is done in parallel as well. 

The two testing functions have some unit testing, as well as time/space testing for the interpolation procedure. 

protocolOpt performs the optimization algorithm, and analyzeProtocol has function to plot the results as well as compare to simulation data. 


Folders with data have different versions numbers corresponding to different runs. 
V0 was on the equally spaced grid. 
V1 introduced shelve caching and additional parameters in rapidly changing regions. 
V2 introduced use of SQL for caching, a refined state space that enables ratcheting, and additional parameters to the left of the nucleation regime. 
V3 added two new parameter sets. One in the disassmbly region, the middle of the other points. The other on the boundary of the nucleation region. Sampling used ratcheting on all "important" states. 



