# CCIT
__Classifier Conditional Independence Test: A CI test that uses a binary classifier (XGBoost) for CI testing__

__This is an implementation of the paper: https://arxiv.org/abs/1709.06138__

_Please cite the above paper if this package is used in any publication._ 

__Usage for pip install__

1. ```pip install CCIT``` or ```sudo -H pip install CCIT```.

2. Now in your python script:

```python
from CCIT import *

pvalue = CCIT(X,Y,Z)    #without bootstrap

pvalue = CCIT(X,Y,Z,num_iter = 30, bootstrap = True, nthread = 20)  #with 30 bootstrap iterations and 20 threads in parallel. 


```

Note that when ```Z is None ```, it produces a pvalue for independence test between X and Y.  

__Usage for pip install from github repo__

1. clone the repo. 

2. ```cd CCIT ```

3. ```pip install . ```

4. (Optional) from the root directory of the package, run the command 

```
nosetests
```

This is a comprehensive test and may take some time to run. 

5. Now in your python script:

```
from CCIT import *

pvalue = CCIT(X,Y,Z)

```
There may be some trouble in installing the xgboost dependency. In that case it is recommended to follow the steps in https://github.com/dmlc/xgboost/blob/master/python-package/build_trouble_shooting.md for install xgboost first. Then install CCIT from pip. 

__CI Tester__

_Functions:_

1. __CCIT()__

```
Main function to generate pval of the CI test. If pval is low CI is rejected if its high we fail to reject CI.
        X: Input X table
        Y: Input Y table
        Z: Input Z table. If None then it reverts back to Independence test between X and Y. 
        Optional Arguments:
        max_depths : eg. [6,10,13] list of parameters for depth of tree in xgb for tuning
        n_estimators: eg. [100,200,300] list of parameters for number of estimators for xgboost for tuning
        colsample_bytrees: eg. recommended [0.8] list of parameters for colsample_bytree for xgboost for tuning
        nfold: n-fold cross validation 
        feature_selection : default 0 recommended
        train_samp: -1 recommended. Number of examples out of total to be used for training. 
        threshold: defualt recommended
        num_iter: Number of Bootstrap Iterations. Default 20. Recommended 30. 
        nthread: Number of parallel thread for running XGB. Recommended number of cores in the CPU. Default 8. 
	bootstrap : True or False. If False, then num_iter is set to 1. One deterministic pval is outputted without averaging. If True, results are averaged over num_iter bootstraps and can have randomness. num_iter in this case has to be >= 20.   
        Output: 
        pvalue of the test. 
     
```

__tl;dr version__

```
If the dimensions of X, Y, and Z are 1,1,2 respectively and if the first three i.i.d samples are as follows:

|  X  | Y   |    Z    |
| 1.0 | 1.0 | 1.5 2.5 |
| 0.5 | 1.2 | 0.5 0.6 |
| 0.1 | 4.5 | 1.2 3.6 |

then the input is: 
```
```python
X = np.array([[1.0],[0.5],[0.1]])
Y = np.array([[1.0],[1.2],[4.5]])
Z = np.array([[1.5,2.5],[0.5,0.6],[1.2,3.6]])
pval = CCIT(X,Y,Z)
```

2. __CI_sampler_conditional_kNN()__

```
Generate Test and Train set for converting CI testing into Binary Classification
    Arguments:
    	X_in: Samples of r.v. X (np.array)
    	Y_in: Samples of r.v. Y (np.array)
    	Z_in: Samples of r.v. Z (np.array)
    	train_len: length of training set, must be less than number of samples 
    	k: k-nearest neighbor to be used: Always set k = 1. 
    Output:
    	Xtrain: Features for training the classifier
    	Ytrain: Train Labels
    	Xtest: Features for test set
    	Ytest: Test Labels
    	CI_data: Developer Use only

```


__DataGen Module__

_Functions:_

1. __generate_samples_cos()__

```
Generate CI,I or NI post-nonlinear samples:
    
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
```   
2. __parallel_cos_sample_gen()__

```
Function to create several many data-sets of post-nonlinear cos transform half of which are CI and half of which are NI, along with the correct labels. The data-sets are stored under a given folder path:

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
	datafile.npy files that constains an array that has the correct label. If the first label is '1' then  'datafile20_0.npy' constains a 'CI' dataset. 
```



