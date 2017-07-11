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

from math import erfc
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



def XGB_crossvalidated_model(max_depths, n_estimators, colsample_bytrees,Xtrain,Ytrain,nfold,feature_selection = 0,nthread = 8):
    '''Function returns a cross-validated hyper parameter tuned model for the training data 
    Arguments:
    	max_depths: options for maximum depth eg: input [6,10,13], this will choose the best max_depth among the three
    	n_estimators: best number of estimators to be chosen from this. eg: [200,150,100]
    	colsample_bytrees: eg. input [0.4,0.8]
    	nfold: Number of folds for cross-validated
    	Xtrain, Ytrain: Training features and labels
    	feature_selection : 0 means feature_selection diabled and 1 otherswise. If 1 then a second output is returned which consists of the selected features

    Output:
    	model: Trained model with good hyper-parameters
    	features : Coordinates of selected features, if feature_selection = 0
    	bp: Dictionary of tuned parameters 

    This procedure is CPU intensive. So, it is advised to not provide too many choices of hyper-parameters
    '''
    classifiers = {}
    model =  xgb.XGBClassifier( nthread=nthread, learning_rate =0.02, n_estimators=100, max_depth=6,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
    model.fit(Xtrain,Ytrain)
    m,n = Xtrain.shape
    features = range(n)
    imp = model.feature_importances_
    if feature_selection == 1:
        features = np.where(imp == 0)[0]
        Xtrain = Xtrain[:,features]
    
    bp = {'max_depth':[0],'n_estimator':[0], 'colsample_bytree' : [0] }
    classifiers['model'] = xgb.XGBClassifier( nthread = nthread, learning_rate =0.02, n_estimators=100, max_depth=6,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.9,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
    classifiers['train_X'] = Xtrain
    classifiers['train_y'] = Ytrain
    maxi = 0
    pos = 0
    for r in max_depths:
        classifiers['model'] = xgb.XGBClassifier( nthread=nthread,learning_rate =0.02, n_estimators=100, max_depth=r,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
        score = cross_validate(classifiers,nfold)
        if maxi < score:
            maxi = score
            pos = r
    bp['max_depth'] = pos
    #print pos
    
    maxi = 0
    pos = 0
    for r in n_estimators:
        classifiers['model'] = xgb.XGBClassifier( nthread=nthread,learning_rate =0.02, n_estimators=r, max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
        score = cross_validate(classifiers,nfold)
        if maxi < score:
            maxi = score
            pos = r
    
    bp['n_estimator'] = pos
    #print pos
    
    maxi = 0
    pos = 0
    for r in colsample_bytrees:
        classifiers['model'] = xgb.XGBClassifier( nthread=nthread, learning_rate =0.02, n_estimators=bp['n_estimator'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=r,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
        score = cross_validate(classifiers,nfold)
        if maxi < score:
            maxi = score
            pos = r
            
    bp['colsample_bytree'] = pos
    model = xgb.XGBClassifier( nthread=nthread,learning_rate =0.02, n_estimators=bp['n_estimator'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(Xtrain,Ytrain)
    
    return model,features,bp


def cross_validate(classifier, n_folds = 5):
    '''Custom cross-validation module I always use '''
    train_X = classifier['train_X']
    train_y = classifier['train_y']
    model = classifier['model']
    score = 0.0
    
    skf = KFold(n_splits = n_folds)
    for train_index, test_index in skf.split(train_X):
        X_train, X_test = train_X[train_index], train_X[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        clf = model.fit(X_train,y_train)
        pred = clf.predict_proba(X_test)[:,1]
        #print 'cross', roc_auc_score(y_test,pred)
        score = score + roc_auc_score(y_test,pred)

    return score/n_folds


def XGBOUT2(bp, all_samples,train_samp,Xcoords, Ycoords, Zcoords,k,threshold,nthread,bootstrap = True):
    '''Function that takes a CI test data-set and returns classification accuracy after Nearest-Neighbor  Bootstrap'''
    
    np.random.seed()
    random.seed()
    num_samp = len(all_samples)
    if bootstrap:
        I = np.random.choice(num_samp,size = num_samp, replace = True)
        samples = all_samples[I,:]
    else:
        samples = all_samples
    Xtrain,Ytrain,Xtest,Ytest,CI_data = CI_sampler_conditional_kNN(all_samples[:,Xcoords],all_samples[:,Ycoords], all_samples[:,Zcoords],train_samp,k)
    model = xgb.XGBClassifier(nthread=nthread,learning_rate =0.02, n_estimators=bp['n_estimator'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11)
    gbm = model.fit(Xtrain,Ytrain)
    pred = gbm.predict_proba(Xtest)
    pred_exact = gbm.predict(Xtest)
    acc1 = accuracy_score(Ytest, pred_exact)
    AUC1 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    gbm = model.fit(Xtrain[:,len(Xcoords)::],Ytrain)
    pred = gbm.predict_proba(Xtest[:,len(Xcoords)::])
    pred_exact = gbm.predict(Xtest[:,len(Xcoords)::])
    acc2 = accuracy_score(Ytest, pred_exact)
    AUC2 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    if AUC1 > AUC2 + threshold:
        return [0.0, AUC1 - AUC2 , AUC2 - 0.5, acc1 - acc2, acc2 - 0.5]
    else:
        return [1.0, AUC1 - AUC2, AUC2 - 0.5, acc1 - acc2, acc2 - 0.5]

def pvalue(x,sigma):

    return 0.5*erfc(x/(sigma*np.sqrt(2)))



def bootstrap_XGB2(max_depths, n_estimators, colsample_bytrees,nfold,feature_selection,all_samples,train_samp,Xcoords, Ycoords, Zcoords,k,threshold,num_iter,nthread, bootstrap = False):
    Xtrain,Ytrain,Xtest,Ytest,CI_data = CI_sampler_conditional_kNN(all_samples[:,Xcoords],all_samples[:,Ycoords], all_samples[:,Zcoords],train_samp,k)
    model,features,bp = XGB_crossvalidated_model(max_depths, n_estimators, colsample_bytrees,Xtrain,Ytrain,nfold,feature_selection = 0,nthread = nthread)
    ntot,dtot = all_samples.shape
    del model
    cleaned = []
    if bootstrap:
        assert (num_iter >= 20),"Number of bootstrap iteration should be atleast 20."
    if bootstrap == False:
        num_iter = 1
    for i in range(num_iter):
        cleaned = cleaned + [XGBOUT2(bp, all_samples,train_samp,Xcoords, Ycoords, Zcoords,k,threshold,nthread,bootstrap)]
    cleaned = np.array(cleaned)
    R = np.mean(cleaned,axis = 0)
    S = np.std(cleaned,axis = 0)
    #print S
    s = S[2]
    s2 = S[4]
    new_t = s
    new_t2 = s2
    #print new_t
    a = np.where(cleaned[:,1] < new_t)
    a2 = np.where(cleaned[:,3] < new_t2)
    R = list(R)
    R = R + [float(len(a[0]))/num_iter]
    R = R + [float(len(a2[0]))/num_iter]
    #pval = pd.Series(cleaned[:,3]).apply(lambda g: pvalue(g,s2))
    #pval = pd.Series(cleaned[:,1]).apply(lambda g: pvalue(g,s))
    p = np.mean(cleaned[:,3])
    if bootstrap:
        pval = pvalue(p,s2)
    else:
        pval = pvalue(p,1/np.sqrt(ntot))
    R = R + [pval]
    dic = {}
    dic['tr_auc_CI'] = R[0]
    dic['auc_difference'] = R[1]
    dic['auc2_deviation'] = R[2]
    dic['acc_difference'] = R[3]
    dic['acc_deviation'] = R[4]
    dic['autotr_auc_CI'] = R[5]
    dic['autotr_acc_CI'] = R[6]
    dic['pval'] = R[7]
    return dic

def CCIT(X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,feature_selection = 0,train_samp = -1,k = 1,threshold = 0.03,num_iter = 20,nthread = 8,bootstrap = False):
    '''Main function to generate pval of the CI test. If pval is low CI is rejected if its high we fail to reject CI.
        X: Input X table
        Y: Input Y table
        Z: Input Z table
        Optional Arguments:
        max_depths : eg. [6,10,13] list of parameters for depth of tree in xgb for tuning
        n_estimators: eg. [100,200,300] list of parameters for number of estimators for xgboost for tuning
        colsample_bytrees: eg. recommended [0.8] list of parameters for colsample_bytree for xgboost for tuning
        nfold: n-fold cross validation 
        feature_selection : default 0 recommended
        train_samp: -1 recommended. Number of examples out of total to be used for training. 
        threshold: defualt recommended
        num_iter: Number of Bootstrap Iterations. Default 10. Recommended 30. 
        nthread: Number of parallel thread for running XGB. Recommended number of cores in the CPU. Default 8. 

        Output: 
        pvalue of the test. 
     '''


    assert (type(X) == np.ndarray),"Not an array"
    assert (type(Y) == np.ndarray),"Not an array"
    assert (type(Z) == np.ndarray),"Not an array"
    
    nx,dx = X.shape
    ny,dy = Y.shape
    nz,dz = Z.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"

    assert (num_iter > 1), "Please provide num_iter > 1."

    all_samples = np.hstack([X,Y,Z])
    #print all_samples.shape

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)

    if train_samp == -1:
        train_len = (2*nx)/3

    #print train_len

    dic = bootstrap_XGB2(max_depths = max_depths, n_estimators=n_estimators, colsample_bytrees=colsample_bytrees,nfold=nfold,feature_selection=0,all_samples=all_samples,train_samp = train_len,Xcoords = Xset, Ycoords = Yset, Zcoords = Zset ,k = k,threshold = threshold,num_iter = num_iter,nthread = nthread,bootstrap = bootstrap)

    return dic['pval']
    
