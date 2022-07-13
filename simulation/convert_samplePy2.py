import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import inspect

#set a path to the assembly folder to do relative importing
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir) 
assemblydir = parentdir + "/assembly/"
sys.path.insert(0, assemblydir) 

#import the necessary functions
from pathwaysPy2 import getPathway
from pathwaysPy2 import convert2npy
from simulate import simulate


if __name__ == '__main__':
    #convert a given gsd file into an npy file
    #use that outfile to analyze the trajectory and convert to npy format

    #read input files
    try:
        outfile = sys.argv[1]  # get trajectory file
    except:
        print("Usage: %s <gsd_file>" % sys.argv[0])
        raise

    #get the pathway from the input gsd file
    path = getPathway(outfile, verbose=True)

    #convert to npy file
    convert2npy(outfile, path)
