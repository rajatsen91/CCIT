'''Author: Rajat Sen (rajat.sen@utexas.edu), Karthikeyan Shanmugam, Ananada Theertha Suresh
#Generation of Synthetic data-sets for CI testing simulations 
	Data-Sets are of three kinds: 
	1. X Independent of Y -----    I 
	2. X Independent of Y given Z ------ CI
	3. X not Independent of Y given Z ------ NI 

'''


#General Headers####################
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool
import copy
#####################################

#sklearn headers##################################
from sklearn.metrics import zero_one_loss
import xgboost as xgb
from sklearn import metrics   
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
#####################################################




def generate_samples_cos(size = 1000,sType = 'CI',dx = 1,dy = 1,dz = 20,nstd = 0.5,freq = 1.0):
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian 
    2. X = cos(<a,Z> + b + noise) and Y = cos(<c,Z> + d + noise) in case of CI
    Arguments:    
        size : number of samples
        sType: CI,I, or NI
        dx: Dimension of X 
        dy: Dimension of Y 
        dz: Dimension of Z 
        nstd: noise standard deviation
        freq: Freq of cosine function
    
    Output:
    	allsamples --> complete data-set
    Note that: 	
    [X = first dx coordinates of allsamples each row is an i.i.d samples]
    [Y = [dx:dx + dy] coordinates of allsamples]
    [Z = [dx+dy:dx+dy+dz] coordinates of all samples]
    '''
    
    num = size
    cov = np.eye(dz)
    mu = np.ones(dz)
    Z = np.random.multivariate_normal(mu,cov,num)
    Z = np.matrix(Z)
    Ax = np.random.rand(dz,dx)
    for i in range(dx):
        Ax[:,i] = Ax[:,i]/np.linalg.norm(Ax[:,i])
    Ax = np.matrix(Ax)
    Ay = np.random.rand(dz,dy)
    for i in range(dy):
        Ay[:,i] = Ay[:,i]/np.linalg.norm(Ay[:,i])
    Ay = np.matrix(Ay)
    
    Axy = np.random.rand(dx,dy)
    for i in range(dy):
        Axy[:,i] = Axy[:,i]/np.linalg.norm(Axy[:,i])
    Axy = np.matrix(Axy)
    
    if sType == 'CI':
        X = np.cos(freq*(Z*Ax + nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),num)))
        Y = np.cos(freq*(Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)))
    elif sType == 'I':
        X = np.cos(freq*(nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),num)))
        Y = np.cos(freq*(nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)))
    else:
        X = np.cos(freq*(np.random.multivariate_normal(np.zeros(dx),np.eye(dx),num)))
        Y = np.cos(freq*(2*X*Axy + Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)))
        
    allsamples = np.hstack([X,Y,Z])
    
    return allsamples


def cos_helper(sims):
	'''
	Helper Function for parallel processing of generate_samples_cos
	'''
	np.random.seed()
	random.seed()
	L = generate_samples_cos(size=sims[0],sType=sims[1],dx=sims[2],dy=sims[3],dz=sims[4],nstd=sims[5],freq=sims[6])
	s = sims[7] + str(sims[8])+'_'+ str(sims[4]) + '.csv'
	L = pd.DataFrame(L,columns = None)
	L.to_csv(s)
	return 1

def parallel_cos_sample_gen(nsamples = 1000,dx = 1,dy = 1,dz = 20,nstd = 0.5,freq = 1,filetype = '../data/dim20/datafile',num_data = 50, num_proc = 4):
	''' 
	Function to create several many data-sets of post-nonlinear cos transform half of which are CI and half of which are NI, 
	along wtih the correct labels. The data-sets are stored under a given folder path. 

	############## The path should exist#####################
	For example create a folder ../data/dim20 first. 


	Arguments:
	nsamples: Number of i.i.d samples in each data-set
	dx, dy, dz : Dimension of X, Y, Z
	nstd: Noise Standard Deviation 
	freq: Freq. of cos function 
	filetype: Path to filenames. if filetype = '../data/dim20/datafile', then the files are stored as '.npy' format in folder './dim20' 
	and the files are named datafile0_20.npy .....datafile50_20.npy
	num_data: number of data files 
	num_proc: number of processes to run in parallel 
	
	Output:
	num_data number of datafiles stored in the given folder. 
	datafile.npy files that constains an array that has the correct label. If the first label is '1' then  'datafile20_0.npy' constains a 'CI' dataset. '''
	inputs = []
	stypes = []
	for i in range(num_data):
		x = np.random.binomial(1,0.5)
		if x > 0:
			sType = 'CI'
		else:
			sType = 'NI'
		inputs = inputs + [(nsamples,sType,dx,dy,dz,nstd,freq,filetype,i)]
		stypes = stypes + [x]
	
	np.save(filetype+'.npy',stypes)
	pool = Pool(processes=num_proc)
	result = pool.map(cos_helper,inputs)
	cleaned = [x for x in result if not x is None]
	pool.close()






