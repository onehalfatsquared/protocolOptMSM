'''
This module will contain functions to evaluate the sparse interpolant matrices at 
a given value of the input parameters, using the parallel evaluation scheme. 

Will begin by taking in a matrix of interpolant objects and assigning them as 
jobs to be run for a particular row of the total transition matrix. This will then be passed 
to a collection of parallel workers to evaluate the matrix. 

Will also include functionality for caching the transition matrices for already computed
parameter values via the shelve module. 
'''

import sys
import shelve 
import multiprocessing
import time

import numpy as np
import scipy
from scipy import sparse

from collections import Counter

import pickle
import sqlite3

#import the radial basis function interpolation library for evaluation calls
import rbf
from rbf.interpolate import RBFInterpolant

#set a global to hold interpolants and their job assignment info - read only
global WORK_LIST

#set a global to get the location of the SQL database
global DB_OPEN

def setSQLdb(data_version='', refine=True):
    #set the file location of the SQL database and read options

    global DB_OPEN
    DB_OPEN = "file:msm" + data_version + "/interp"
    if refine:
        DB_OPEN += "R"
    DB_OPEN += ".db?mode=ro"

    return

def createWorkList(matrixI):
	#create a global memory work list using the contents of matrixI

    #get a list of all the keys in the dict, sorted
    all_keys = sorted(matrixI.keys())

    #get a list of the first index of each key
    id1 = [t[0] for t in all_keys]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    global WORK_LIST
    WORK_LIST = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        j_arrays = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key[1])
            j_arrays.append(matrixI[key])

        WORK_LIST[i] = [j_ids, j_arrays] 

    return nsa

def computeRowPar(row_idx, E):
    #create the 3 arrays that define the transition matrix row

    #make lists to store non-zero columns as well the probability in that column
    j_list = WORK_LIST[row_idx]
    thisRowData = []
    nonZeroCols = j_list[0]

    j_array = j_list[1]

    #loop over non-zero columns in the keys
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-12:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            # print(j, nonZeroCols)
            # print(nonZeroCols[j])
            del nonZeroCols[j]
            continue

    
    #perform a re-normalization to ensure rows sum to 1
    npRow = np.array(thisRowData, dtype=float)
    S     = np.sum(npRow)
    if (S > 1e-6):
        npRow /= S

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    return [r, nonZeroCols, npRow.tolist()]

def eval_matrix_sparsePar(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E) for i in range(nsa))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowPar, rowInput)

    #close pool
    p.close()
    p.join()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    
    #flatten all the lists
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    return imat

def computeRowDPar(row_idx, E):
    #create the 3 arrays that define the transition matrix row

    #make lists to store non-zero columns as well the probability in that column
    j_list = WORK_LIST[row_idx]
    thisRowData = []
    nonZeroCols = j_list[0]

    j_array = j_list[1]

    #make a storage array for the derivatives
    dims = len(E)
    d = [[] for i in range(dims)]

    #loop over non-zero columns in the keys - backwards in case of deletion
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-12:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            # print(j, nonZeroCols)
            # print(nonZeroCols[j])
            del nonZeroCols[j]
            continue

        #compute each derivative and append it
        for dim in range(dims):
            #get a tuple of the proper dimension to diff
            diff = np.zeros(dims, dtype=int)
            diff[dim] = 1

            deriv = I([E],diff)[0]
            d[dim].insert(0,deriv)

    
    #perform a re-normalization to ensure rows sum to 1
    npRow   = np.array(thisRowData, dtype=float)
    npDRows = np.array(d, dtype=float)
    S       = np.sum(npRow)
    D       = np.sum(npDRows,1)

    if (S > 1e-12):
        npRow /= S
        for i in range(dims):
            npDRows[i] = (npDRows[i] - npRow * D[i]) / (S)

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    #return in order row_i, col_i, mat_rows, deriv_rows
    return [r, nonZeroCols, npRow.tolist(), npDRows.tolist()]

def eval_matrix_deriv_sparsePar(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  
    #also evaluate the derivative

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E) for i in range(nsa))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowDPar, rowInput)

    #close pool
    p.close()
    p.join()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    der = [item[3] for item in results]
    
    #flatten all the lists except the derivatives list
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    #handle derivatives separately
    derivs = []
    for i in range(len(E)):
        deriv = [item for sublist in der for item in sublist[i]]
        derivs.append(scipy.sparse.coo_matrix((deriv, (row, col)), shape = (nsg,nsg)).tocsr())

    #return the transition matrix and its partial derivatives
    return imat, derivs






## Full global work list loaded
###############################################################################
## Use shelve cache to avoid loading everything in memory at once

def createWorkIndices(iCache):
    #create a global memory work list using the contents of matrixI

    #get a list of all the keys in the dict, sorted
    all_keys = list(iCache.keys())
    key_tups = [eval(key) for key in all_keys]
    all_keys = [x for _, x in sorted(zip(key_tups, all_keys))]
    key_tups = sorted(key_tups)

    #get a list of the first index of each key
    id1 = [t[0] for t in key_tups]

    #count how many nonzero entries each row index has, and the number of active states
    nz_counts = Counter(id1)
    nsa = len(nz_counts)

    #create a list of offsets to get to a particular state
    offsets = np.zeros(nsa+1,dtype=int)
    for i in range(1,nsa):
        offsets[i] = offsets[i-1] + nz_counts[i-1]

    #for each i, precompute all the work that needs to be done
    work_list = np.zeros(nsa, dtype=object)
    for i in range(nsa):
        num_non_zeros_i = nz_counts[i]
        j_ids    = []
        for j in range(num_non_zeros_i):
            key = all_keys[offsets[i]+j]
            j_ids.append(key)

        work_list[i] = j_ids 

    return nsa, work_list

def computeRowPar_cache(row_idx, E, work_list, iCache):
    #create the 3 arrays that define the transition matrix row

    #make lists to store non-zero columns as well the probability in that column
    #this needs to match up with how the arrays are created in the global work list fn**
    j_list = work_list[row_idx]
    thisRowData = []
    nonZeroCols = []
    j_array = []

    #make the j_array by using the cache
    for i in range(len(j_list)):

        #get the interpolant from cache using key in j_list
        key = j_list[i]
        j_array.append(iCache[key])

        #fill nonzero cols with the first component of the key
        key_tup = eval(key)
        nonZeroCols.append(key_tup[1])

    #loop over non-zero columns in the keys
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-12:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            # print(j, nonZeroCols)
            # print(nonZeroCols[j])
            del nonZeroCols[j]
            continue

    
    #perform a re-normalization to ensure rows sum to 1
    npRow = np.array(thisRowData, dtype=float)
    S     = np.sum(npRow)
    if (S > 1e-6):
        npRow /= S

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    return [r, nonZeroCols, npRow.tolist()]

def eval_matrix_sparsePar_cache(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  
    #read the interpolants from cache as necessary

    #load the interpolant cache
    shelve_name = "msm/interpCache"
    shelve_name = "/home/anthony/interpCache"
    iCache = shelve.open(shelve_name, flag='r')

    #make a work list applicable to the cache
    nsa, work_list = createWorkIndices(iCache)

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E, work_list, iCache) for i in range(nsa))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowPar_cache, rowInput)

    #close pool
    p.close()
    p.join()

    #close the cache
    iCache.close()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    
    #flatten all the lists
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    return imat

###############################################################################
## Use sql database to avoid loading everything in memory at once

def pruning(npRow, nonZeroCols, row_idx):
    #prune probabilities according to some criterion to make the MSM more accurate
    #after interpolation 

    #get the row sum
    S = np.sum(npRow)

    #check if the sum of non-normalized probabilities reaches a threshold
    prob_cut = 0.02
    if (S > prob_cut): 

        #get ratio of the max and min probability in the row
        M = np.max(npRow)
        m = np.min(npRow)
        r = M/m
        
        #define new lists to store only un-pruned entries
        npRowPruned = []
        nzcPruned   = []

        #keep track of all pruned probability
        prunedProb = 0

        #loop over the row, determining which entries to keep
        for entry in range(len(npRow)):

            #get the probability of this entry
            this_prob = npRow[entry]

            #keep probabilities greater than a cutoff, a constant factor times the largest
            if this_prob > m * r / 40000.0:
                npRowPruned.append(this_prob)
                nzcPruned.append(nonZeroCols[entry])

            #otherwise, remove the entry and keep track of removed probability
            else:
                # print("Removing val {}".format(this_prob))
                prunedProb += this_prob

        #add the pruned prob back to all except the diagonal
        num_cols = len(nzcPruned)
        if row_idx in nzcPruned:
            num_cols += -1

        if num_cols > 1:
            spread_prob = prunedProb / float(num_cols)

            for entry in range(len(npRowPruned)):
                if nzcPruned[entry] != row_idx:
                    npRowPruned[entry] += spread_prob

        #rename the arrays and renormalize them
        npRow = npRowPruned
        nonZeroCols = nzcPruned

        Snew = np.sum(npRow)
        npRow /= Snew

    #if the sum of row entries is too small, just put a 1 on the diagonal
    else:
        npRow = np.array([1])
        nonZeroCols = [row_idx]
        # print(npRow, nonZeroCols)

    #return the row of probabilities and list of nonzero columns
    return npRow, nonZeroCols

def pruningD(npRow, npDRows, nonZeroCols, dims, row_idx):
    #prune probabilities according to some criterion to make the MSM more accurate
    #after interpolation. Also construct the derivative matrix consistent with pruning

    #get the probability and derivative row sum
    S = np.sum(npRow)
    D = np.sum(npDRows, 1)
    # print(row_idx, S)

    #check if the sum of non-normalized probabilities reaches a threshold
    prob_cut = 0.02
    if (S > prob_cut): 

        #get ratio of the max and min probability in the row
        M = np.max(npRow)
        m = np.min(npRow)
        r = M/m
        
        #define lists to store un-pruned probabilities
        np_row_kept    = []
        nzc_kept       = []
        deriv_row_kept = [[] for _ in range(dims)]

        #define lists to store pruned probabilities
        np_row_pruned = []
        deriv_row_pruned = [[] for _ in range(dims)]

        #loop over the row, determining which entries to keep
        nonZeros = len(npRow)
        for entry in range(nonZeros):

            #get the probability and derivatives of this entry
            this_prob    = npRow[entry]
            these_derivs = [npDRows[dim][entry] for dim in range(dims)]

            #keep probabilities greater than a cutoff, a constant factor times the largest
            if this_prob > m * r / 40000.0:
                np_row_kept.append(this_prob)
                nzc_kept.append(nonZeroCols[entry])
                for dim in range(dims):
                    deriv_row_kept[dim].append(these_derivs[dim])

            #otherwise, remove the entry and keep track of removed probability and derivs
            else:
                np_row_pruned.append(this_prob)
                for dim in range(dims):
                    deriv_row_pruned[dim].append(these_derivs[dim])

        #determine number of kept states, not counting the diagonal entry
        kept_cols = len(nzc_kept)
        if row_idx in nzc_kept:
            kept_cols += -1

        #do the spreading as long as there are entries to spread to
        if len(np_row_pruned) > 0 and kept_cols > 0:

            #determine the probability being spread to each kept state
            Sp = np.sum(np_row_pruned)
            spread_prob = Sp / float(kept_cols)

            #determine the derivative being spread to each kept state
            Dp = np.sum(deriv_row_pruned, 1)

            #do the spreading, but not to any diagonal entries
            for entry in range(len(nzc_kept)):
                if nzc_kept[entry] != row_idx:

                    #spread the probability
                    np_row_kept[entry] += spread_prob

                    #spread the derivatives
                    for dim in range(dims):
                        deriv_row_kept[dim][entry] += (Dp[dim] / float(kept_cols))

        #turn the lists into arrays for arithmetic
        np_row_kept    = np.array(np_row_kept, dtype=float)
        deriv_row_kept = np.array(deriv_row_kept, dtype=float)

        #renormalize the arrays, using the proper quotient rule for derivatives
        np_row_kept_norm    = np_row_kept / S
        deriv_row_kept_norm = deriv_row_kept
        for dim in range(dims):
            quotient_rule = (deriv_row_kept[dim] - np_row_kept_norm * D[dim]) / (S)
            deriv_row_kept_norm[dim] = quotient_rule

    #if the sum of row entries is too small, just put a 1 on the diagonal
    else:
        np_row_kept_norm = np.array([1])
        nzc_kept = [row_idx]
        deriv_row_kept_norm = np.array([[0] for _ in range(dims)])

    #return the row of probabilities, list of nonzero columns, and derivative entries
    return np_row_kept_norm, nzc_kept, deriv_row_kept_norm



def computeRowPar_SQL(row_idx, E):
    #create the 3 arrays that define the transition matrix row

    # a = time.time()
    #load the interpolant database
    db = sqlite3.connect(DB_OPEN, uri=True)

    #init the arrays needed to make a row
    thisRowData = []
    nonZeroCols = []
    j_array = []

    #do a search query for the row
    cursor = db.execute("SELECT ROW,COL,FIT FROM INTERP WHERE ROW=?",(row_idx,))
    for item in cursor.fetchall():

        #convert the column from bytes to int
        col = int.from_bytes(item[1],sys.byteorder)

        #un-pickle the interpolant
        up = pickle.loads(item[2])

        #append to arrays
        j_array.append(up)
        nonZeroCols.append(col)

    #close the database
    db.close()

    # b = time.time()
    
    #loop over non-zero columns in the keys
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-7:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            # print(j, nonZeroCols)
            # print(nonZeroCols[j])
            del nonZeroCols[j]
            continue

    
    #perform a re-normalization to ensure rows sum to 1
    npRow = np.array(thisRowData, dtype=float)

    #do a pruning step to remove unneeded entries and normalize
    npRow, nonZeroCols = pruning(npRow, nonZeroCols, row_idx)

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    # c = time.time()
    # print("Row {} total took {} seconds. Read took {} seconds".format(row_idx,c-a, b-a))

    return [r, nonZeroCols, npRow.tolist()]

def computeRowDPar_SQL(row_idx, E):
    #create the 3 arrays that define the transition matrix row

    #load the interpolant database
    db = sqlite3.connect(DB_OPEN, uri=True)

    #init the arrays needed to make a row
    thisRowData = []
    nonZeroCols = []
    j_array = []

    #make a storage array for the derivatives
    dims = len(E)
    d = [[] for i in range(dims)]

    #do a search query for the row
    cursor = db.execute("SELECT ROW,COL,FIT FROM INTERP WHERE ROW=?",(row_idx,))
    for item in cursor.fetchall():

        #convert the column from bytes to int
        col = int.from_bytes(item[1],sys.byteorder)

        #un-pickle the interpolant
        up = pickle.loads(item[2])

        #append to arrays
        j_array.append(up)
        nonZeroCols.append(col)

    #close the database
    db.close()
    
    #loop over non-zero columns in the keys
    for j in range(len(j_array)-1,-1,-1):

        #get the interpolant
        I = j_array[j]

        #check if the interpolant is non-trivial
        y = I.d
        if (np.all(y < 1e-6)):
            del nonZeroCols[j]
            continue

        #if non-trivial, compute the interpolant, and append the probability
        prob = max(I([E])[0],0.0)
        if prob > 1e-8:
            #append it to the front of the list
            thisRowData.insert(0,prob)
        else:
            del nonZeroCols[j]
            continue

        #compute each derivative and append it
        for dim in range(dims):
            #get a tuple of the proper dimension to diff
            diff = np.zeros(dims, dtype=int)
            diff[dim] = 1

            deriv = I([E],diff)[0]
            d[dim].insert(0,deriv)

    #perform a re-normalization to ensure rows sum to 1
    npRow   = np.array(thisRowData, dtype=float)
    npDRows = np.array(d, dtype=float)

    #do a pruning and normalization of the row
    npRow, nonZeroCols, npDRows = pruningD(npRow, npDRows, nonZeroCols, dims, row_idx)

    #make a row entry that is i repeated len(npRow) times
    r = [row_idx] * len(npRow)

    return [r, nonZeroCols, npRow.tolist(), npDRows.tolist()]

def eval_matrix_sparsePar_SQL(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  
    #read the interpolants from cache as necessary

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E) for i in range(nsg))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowPar_SQL, rowInput)

    #close pool
    p.close()
    p.join()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    
    #flatten all the lists
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    return imat

def eval_matrix_deriv_sparsePar_SQL(E, nsa, nsg, num_procs):
    #do the matrix evaluation using sparse storage - parallelized  
    #also evaluate the derivative

    #create the pool
    p = multiprocessing.Pool(num_procs)

    #mak generator for input for each row
    rowInput = ((i, E) for i in range(nsa))

    #perform the computation for each list in j_list
    results = p.starmap(computeRowDPar_SQL, rowInput)

    #close pool
    p.close()
    p.join()

    #combine all the lists
    row = [item[0] for item in results]
    col = [item[1] for item in results]
    dat = [item[2] for item in results]
    der = [item[3] for item in results]
    
    #flatten all the lists except the derivatives list
    row = [item for sublist in row for item in sublist]
    col = [item for sublist in col for item in sublist]
    dat = [item for sublist in dat for item in sublist]

    #make the sparse array
    imat = scipy.sparse.coo_matrix((dat, (row, col)), shape = (nsg,nsg))

    #covert to csr
    imat = imat.tocsr()

    #handle derivatives separately
    derivs = []
    for i in range(len(E)):
        deriv = [item for sublist in der for item in sublist[i]]
        derivs.append(scipy.sparse.coo_matrix((deriv, (row, col)), shape = (nsg,nsg)).tocsr())

    #return the transition matrix and its partial derivatives
    return imat, derivs
