'''
This module's purpose is to convert between data formats used to store data. 

Interpolant objects are stored for each entry of a transition matrix, indexed by two states,
i and j. 

Storage options are:

1) Pickled sparse array with keys (i,j). 
Pros: Fast access and evaluations (10 second evals)
Cons: Large memory usage. May not fit in RAM and cause frequent swapfile access. 

2) Shelve database
Pros: Low RAM usage, saved to HDD. 
Cons: Terribly slow access

3) SQL database
Pros: Low RAM usage, saved to HDD.
Cons: Slow access until evaluated several times. Not sure why but probably a caching effect.
After cached, evaluation times on par with loading all data (around 80 seconds)
'''

#system imports
import sys
import os
import fnmatch
import inspect

#array imports
import numpy as np
import scipy
from scipy import sparse

#database imports
import shelve
import pickle
import sqlite3

#be able to recognize rbf interpolant objects
import rbf
from rbf.interpolate import RBFInterpolant


def pickleToShelve(pickledFitLoc):
    #convert a pickled array of interpolants to a shelve cache
    #assume the pickle can be loaded in RAM

    with open(pickledFitLoc, 'rb') as f:
        pickledFit = pickle.load(f)
        print("Interpolant dict loaded")

    #create the shelve file
    shelve_name = "interpCache"
    f = shelve.open(shelve_name)

    #loop over keys and add it to f
    all_keys = list(pickledFit.keys())
    num_keys = len(all_keys)
    for i in range(num_keys):
        key = all_keys[i]
        f[repr(key)] = pickledFit[key]

        #print progress message
        if (i % 1000) == 0:
            print("Copying key {} of {}".format(i,num_keys))


    #test that it worked 
    test_key = all_keys[342]
    print(f[repr(test_key)], pickledFit[test_key])

    #close the shelve module 
    f.close()

def pickleToSQL(pickledFitLoc):
    #convert a pickled array of interpolants to a sqlite3 database
    #assume pickle can be loaded in RAM

    with open(pickledFitLoc, 'rb') as f:
        pickledFit = pickle.load(f)
        print("Interpolant dict loaded")

    #create the database
    conn = sqlite3.connect("interp.db")

    #create a table for interpolants
    conn.execute('''CREATE TABLE IF NOT EXISTS INTERP
         (ROW    INT    NOT NULL,
          COL    INT    NOT NULL,
          FIT    BLOB   NOT NULL,
          UNIQUE(ROW, COL) ON CONFLICT REPLACE);''')

    #loop over the keys and add it to the database
    all_keys = list(pickledFit.keys())
    num_keys = len(all_keys)
    for i in range(num_keys):
        key = all_keys[i]
        R = key[0]
        C = key[1]
        I = pickledFit[key]
        pI = pickle.dumps(I, pickle.HIGHEST_PROTOCOL)
        
        conn.execute('''INSERT INTO INTERP (ROW, COL, FIT) \
            VALUES (?, ?, ?)''',(R,C,sqlite3.Binary(pI)))

        #print progress message
        if (i % 1000) == 0:
            print("Copying key {} of {}".format(i,num_keys))

    #save the entries
    conn.commit()

    #test selection
    cursor = conn.execute("SELECT ROW,COL FROM INTERP WHERE ROW=121")
    for item in cursor.fetchall():
        print(item[0], int.from_bytes(item[1],sys.byteorder))

    #close the db
    conn.close()

    return

def readDBitem(item):
    #read the bytestream database items and convert to useful variables

    #get the row normally
    row = item[0]

    #convert the column from bytes to int
    col = int.from_bytes(item[1],sys.byteorder)

    #un-pickle the interpolant
    up = pickle.loads(item[2])

    #return all three
    return row, col, up

def SQLtoPickle(sqlDBloc):
    #convert a SQL database to a pickled array

    #set the open options to be read only
    db_open = "file:" + sqlDBloc + "?mode=ro"

    #define dict to store objects
    Pdict = dict()

    #open the database
    conn = sqlite3.connect(db_open, uri=True)

    #get cursor with all rows
    cursor = conn.execute('SELECT * FROM INTERP') 

    #loop over rows
    for item in cursor:
        
        #extract the data from the item
        row, col, up = readDBitem(item)
        print(row,col)

        #add to dict
        Pdict[tuple((row,col))] = up

    #pickle the dict
    Iname = sqlDBloc.split('.')[0] + "dict"
    with open(Iname, 'wb') as f:
        pickle.dump(Pdict, f)

    return








if __name__ == "__main__":


    #set pickled file loc
    # PfitLoc = "msm/matrixI"

    #test the conversion of pickle to shelve
    #pickleToShelve(PfitLoc)

    #test conversion of pickle to SQL
    #pickleToSQL(PfitLoc)

    #test conversion of SQL to pickle
    dbLoc = "msm/interpR.db"
    SQLtoPickle(dbLoc)

