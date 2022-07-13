

def writeTXT(i, H, P):
    #write out the txt file to specify the job to slurm

    #write filename
    filename = "input" + str(i) + ".txt"

    nc = 40 #number of columns until =

    #open file to write
    f = open(filename, 'w')

    #get the H and P values from the lists
    h = H[i]
    p = P[i]

    #set all the parameters and values
    parameters = ['Capsomers', 'Particle Bond Strength', 'HH Bond Strength', 'HP Bond Strength',
                  'Box Size', 'Sphere Radius', 'Cone Diameter', 'Time Step', 'Height', 'Alpha',
                  'Alpha Morse CC', 'Alpha Morse NC', 'Final Time', 'Animation Time', 'kT', 
                  'Size Ratio']

    values     = [300, 7, h, p, 120, 8.3, 7.7, 0.02, 4.0, 17.8, 14.0, 2.0, 800000, 25, 1.0,0.77]

    #write all the parameters in a loop
    for i in range(len(parameters)):
        num_spaces = nc - len(parameters[i])
        f.write(parameters[i] + ' '*num_spaces + '= ' + str(values[i]) + '\n')
    

    #close file
    f.close()

    return


def getParameterMap():
    #the rsync operation needs to know what folders to find trajectories in
    #can get this info from the parameter map file

    mfile_loc = "parameterMap.txt"
    m = open(mfile_loc, 'r')
    Lines = m.readlines()
    H = []
    P = []
    for line in Lines:
        line = line.split()
        H.append(line[1])
        P.append(line[2])
        
    m.close()

    return H, P




if __name__ == "__main__":

    #get map from file num to parameter values
    H, P = getParameterMap()
    num_files = len(H)


    #write the file for each input
    for i in range(num_files):
        writeTXT(i, H, P)
