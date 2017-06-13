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




def CI_sampler_conditional_kNN(X_in,Y_in,Z_in,train_len = -1, k = 1):
    '''Generate Test and Train set for converting CI testing into Binary Classification
    Arguments:
    	X_in: Samples of r.v. X (np.array)
    	Y_in: Samples of r.v. Y (np.array)
    	Z_in: Samples of r.v. Z (np.array)
    	train_len: length of training set, must be less than number of samples 
    	k: k-nearest neighbor to be used: Always set k = 1. 

    	Xtrain: Features for training the classifier
    	Ytrain: Train Labels
    	Xtest: Features for test set
    	Ytest: Test Labels
    	CI_data: Developer Use only

    '''
    assert (type(X_in) == np.ndarray),"Not an array"
    assert (type(Y_in) == np.ndarray),"Not an array"
    assert (type(Z_in) == np.ndarray),"Not an array"
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"

    samples = np.hstack([X_in,Y_in,Z_in])

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)
    
    if train_len == -1:
    	train_len = 2*len(X_in)/3

    assert (train_len < nx), "Training length cannot be larger than total length"

    train = samples[0:train_len,:]
    train_2 = copy.deepcopy(train)
    X = train_2[:,Xset]
    Y = train_2[:,Yset]
    Z = train_2[:,Zset]
    Yprime = copy.deepcopy(Y)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree',metric = 'l2').fit(Z)
    distances, indices = nbrs.kneighbors(Z)
    for i in range(len(train_2)):
        index = indices[i,k]
        Yprime[i,:] = Y[index,:]
    train1 = train_2
    train2 = np.hstack([X,Yprime,Z])
    y1 = np.ones([len(train1),1])
    y2 = np.zeros([len(train2),1])
    all_train1 = np.hstack([train1,y1])
    all_train2 = np.hstack([train2,y2])
    all_train = np.vstack([all_train1,all_train2])
    shuffle = np.random.permutation(len(all_train))
    train = all_train[shuffle,:]
    l,m = train.shape
    Xtrain = train[:,0:m-1]
    Ytrain = train[:,m-1]
    
    test = samples[train_len::,:]
    test_2 = copy.deepcopy(test)
    X = test_2[:,Xset]
    Y = test_2[:,Yset]
    Z = test_2[:,Zset]
    Yprime = copy.deepcopy(Y)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree',metric = 'l2').fit(Z)
    distances, indices = nbrs.kneighbors(Z)
    for i in range(len(test_2)):
        index = indices[i,k]
        Yprime[i,:] = Y[index,:]
    test1 = test_2
    test2 = np.hstack([X,Yprime,Z])
    y1 = np.ones([len(test1),1])
    y2 = np.zeros([len(test2),1])
    all_test1 = np.hstack([test1,y1])
    all_test2 = np.hstack([test2,y2])
    all_test = np.vstack([all_test1,all_test2])
    shuffle = np.random.permutation(len(all_test))
    test = all_test[shuffle,:]
    l,m = test.shape
    Xtest = test[:,0:m-1]
    Ytest = test[:,m-1]
    
    CI_data = np.vstack([train2,test2])
    
    
    return Xtrain,Ytrain,Xtest,Ytest,CI_data
