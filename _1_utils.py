#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:06:49 2017

@author: george

A set of utility functions to facilitate the experiments
Some are costum made and others are taken and adjusted from the sources mentioned 
"""

from __future__ import division, print_function, absolute_import
from copy import deepcopy
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

def parse_activations(loc_file,M = 179500,N=100,partial=False):
    """
	Parse the activations time series from files 
    """
    print("\nReading:",loc_file)
    neuron_time_series = np.zeros((M, N))#
    for timestep, line in enumerate( open(loc_file) ):
        neuron_time_series[timestep,:] = map(float,line.strip().split(","))
    if(partial):
        return neuron_time_series[:10000,]
    else:
        return neuron_time_series
    

def parse_neuron_connections(loc_file_N, N = 100):
    print("\nReading:",loc_file_N)
    neuron_connections = np.zeros([N,N])
    for e, line in enumerate( open(loc_file_N) ):
        row = line.strip().split(",")
        weight = int(row[2])
        if weight == -1:
             weight=0
        neuron_connections[int(row[0]) - 1,int(row[1]) - 1] = weight
    return neuron_connections


def parse_neuron_positions(loc_file_P, N = 100):
    print("\nReading:", loc_file_P)
    neuron_positions = np.zeros([N,2])
    for neuron_id, line in enumerate( open(loc_file_P) ):
        row = line.strip().split(",")
        x = int(1000*float(row[0])/2)
        y = int(1000*float(row[1])/2)
        neuron_positions[neuron_id,0] = x
        neuron_positions[neuron_id,1] = y
    return neuron_positions



"""
Unscatter the effect of light in fluorescence series, 
based on https://github.com/spoonsso/TFconnect
"""
def unscatter(fluor, positions):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and network position information 
        (spatial position), returns a 2-D numpy array of fluorescence time series with light-scattering effects removed

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
        positions: 2-D numpy array of spatial positions. Rows are cells, columns are x/y-coordinates
    outputs---
        2-D numpy array of fluorescence time series, scattering effects removed. rows are cells, columns are time points / frames
    """
    
    lsc = 0.025
    Nn = fluor.shape[0]
    D = np.zeros((Nn,Nn))
    for i in range(Nn):
        for j in range(Nn):
            if i==j:
                D[i,i] = 1
            else:
                D[i,j]=((positions[i,0]-positions[j,0])**2+(positions[i,1]-positions[j,1])**2)**0.5;
                D[i,j]=0.15*np.exp(-(D[i,j]/lsc)**2)

    Dinv = np.linalg.inv(D)

    Xfluor = fluor.copy()

    #Mtest here is time points x neurons
    for j in range(fluor.shape[1]):
        Xfluor[:,j] = np.dot(Dinv,fluor[:,j])

    return Xfluor.T



"""         Preprocess ofChalearn original code
Based on mlwave implementation https://github.com/bisakha/Connectomics
"""
def chalearn_discretization(F):        
    
    relativeBins = False;                           
    conditioningLevel = 0.12
    debug = True;                                    
    epsilon = 1e-3; # To avoid weird effects at bin edges                           
                                
    #Get the conditioning level                         
    avgF = np.mean(F, axis = 1, dtype=np.float64)                           
    if(conditioningLevel == 0):                         
        hits = np.histogram(avgF,  100);                            
        idx = np.argmax(hits[0])                            
        pos = hits[1]                           
        CL =  pos[idx]+0.05;                            
        print('Best guess for conditioning found at: % .2f \n'%(CL));                           
    else:                           
        CL = conditioningLevel;                         
                                
    #print avgF                           
    ###Apply the conditioning                           
    G = []  
    for i in range(len(avgF)):                          
       if avgF[i] >= CL :                           
           G.append(2)                          
       else:                            
           G.append(1)                          
                            
    hist, bins = np.histogram(F, bins = 100)    
                         
    ###Apply the high pass filter                           
    F_min = np.amin(F, axis=0)                          
    F_max = np.amax(F, axis=0)                          
    diff_F = []                         
    diff_F = np.diff(F.T)                         
                                
    G = G[1:len(G)-1]                           
    binEdges = []                           
                                
    ### Discretize the signal
    F = deepcopy(diff_F)
    D = deepcopy(F)                         
    D[:] = np.NAN                           
    if(len(bins) > 1):                          
       relativeBins = False; # Just in case                         
                                
    F_min = np.amin(diff_F, axis=0)                         
    F_max = np.amax(diff_F, axis=0)                                    
    hits = []                           
    #bins = [3]
    bins = [-10,0.12,10] #default for GTE = [-10,0.12,10], default original: 3
    
    if(relativeBins):                           
                                  
       for i in range(0,F_min.shape[0]-1):                          
            bins = bins + 1                         
            binEdges = np.linspace(F_min[i]- epsilon, F_max[i]+ epsilon, 4)                         
            #binEdges = np.linspace(F_min[i]-epsilon, F_max[i]+epsilon, bins+1)                         
            for j in range( 0,(len(binEdges)-2)):                           
                for k in range(len(F)):                         
                    if diff_F[k,i] >= binEdges[j] and diff_F[k,i] < binEdges[j+1]:                          
                        hits.append(1)                          
                    else:                           
                        hits.append(0)                          
            for k in range(len(diff_F)):                                    
               D[hits[k], i] = j
    else:                           
        if(len(bins) == 1):             
            binEdges = np.linspace(F.min()- epsilon, F.max() + epsilon, 4)                            
                                        
        else:                           
            binEdges = bins;                            
        hits = []                           
       
        for j in range(len(binEdges)-1):                          
            for i in range(F.shape[0]):                         
                for k in range(F.shape[1]):

                    if (F[i,k] >= binEdges[j] and F[i,k] < binEdges[j+1]):
                            D[i,k] = j
        if(debug):                          
            print('Global bin edges set at: ');                            
            for j in range(len(binEdges)):                            
                print("%.2f" %( binEdges[j]))                            
    
    return D.T    


"""         Preprocess of  Winning solution
Based on https://github.com/asutera/kaggle-connectomics
"""

def f1(X):
    return X + np.roll(X, -1, axis=0) + np.roll(X, 1, axis=0)


def f2(X):
    return (X + np.roll(X, 1, axis=0) + 0.8 * np.roll(X, 2, axis=0)
            + 0.4 * np.roll(X, 3, axis=0))


def g(X):
    return np.diff(X, axis=0)


def h(X, threshold=0.11):
    threshold1 = X < threshold * 1
    threshold2 = X >= threshold * 1
    X_new = np.zeros_like(X)
    X_new[threshold1] = 0
    X_new[threshold2] = 1#X[threshold2]
    return X_new


def w(X):
    X_new = X
    Sum_X_new = np.sum(X_new, axis=1)
    Sum4 = Sum_X_new
    for i in range(X_new.shape[0]):
        if Sum4[i] != 0:
            X_new[i, :] = ((X_new[i, :] + 1) ** (1 + (1. / Sum4[i])))
        else:
            X_new[i, :] = 3
    return X_new


def winning_discretization(X, LP='f1', weights=True):
    
    if LP == 'f1':
        X = f1(X)
    elif LP == 'f2':
        X = f2(X)
    else:
        raise ValueError("Unknown filter, got %s." % LP)
    X = g(X)
    X = h(X)
    if weights:
        X = w(X)
    X[X<=1] = 0
    X[X>1] = 1
    return X



def downsample(fluor):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a downsampled version, following the
        filtering method published by Romaszko (threshold summed time-diff global network activity at 0.02)

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
    outputs---
        fluor_ds: downsampled numpy array of fluorescence time series.
        
    FROM  https://github.com/spoonsso/TFconnect
    """
    thresh1 = 0.02
    thresh2 = 30000000
    fluordiff = np.diff(fluor, axis=1)
    totF = np.mean(fluordiff, axis=0)
    fluor = fluor[:,np.hstack((False,np.logical_and(totF>thresh1,totF<thresh2)))]
    
    return fluor


def standardize_rows(np_arr):
    """
    Standardizes data row-wise by subtracted the row mean and dividng by the row standard deviation

    inputs---
        np_arr: 2-D numpy array
    outputs---
        standardized 2-D numpy array
    
    FROM  https://github.com/spoonsso/TFconnect
    """

    np_arr = np_arr - np.mean(np_arr,axis=1)[:,None]
    np_arr = np_arr/np.std(np_arr,axis=1)[:,None]
    return np_arr


"""
Compute evaluation metrics 
based on Scott Linderman's Code https://github.com/slinderman/pyhawkes
"""
def compute_auc_roc(A_true,Connectivity):
    A_flat = A_true.ravel()
    
    aucs = roc_auc_score(A_flat,Connectivity.ravel())

    return aucs


def compute_auc_prc(A_true, Connectivity, average="macro"):
    A_flat = A_true.ravel()
    
    prcs = average_precision_score(A_flat, Connectivity.ravel(), average=average)
    
    return prcs

