# CCIT
__Classifier Conditional Independence Test: A CI test that uses a binary classifier (XGBoost) for CI testing__

__DataGen Module__

_Functions:_

1. __generate_samples_cos()__

'''
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
'''   
2. __parallel_cos_sample_gen()__

'''
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
'''



