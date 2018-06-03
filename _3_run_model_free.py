#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george

Run model free based inference in all 6 networks for the different discretization techniques
Infer a network given activation time series using
	Correlation
	Partial Correlation with PCA
	Partial Correlation with Graphical Lasso
"""
import os
os.chdir("Path/To/Code")

import pandas as pd
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import GraphLassoCV


def infer_correlation(neuron_activations, timeLag = 1):
#------------------------ Infer connections based on cross correlation (based on Scott Linderman's code)
 
    (T,N) = neuron_activations.shape
    #----- Neuron 1, neuron 2, list of cross correlations for different lags
    H = np.zeros((N,N,timeLag))

    #----- Compute cross correlation at each time offset
    for dt in np.arange(timeLag):
        print(" In lag :"+str(dt))
        # Compute correlation in sections to conserve memory
        chunksz = 16
        for n1 in np.arange(N):#, step=chunksz
            for n2 in np.arange(N):
                n1c = min(n1 + chunksz, N)
                n2c = min(n2 + chunksz, N)
                # Corr coef is a bit funky. We want the upper right quadrant
                # of this matrix. The result is ((n1c-n1)+(n2c-n2)) x ((n1c-n1)+(n2c-n2))
                H[n1:n1c, n2:n2c, dt] = np.corrcoef(neuron_activations[:T-dt, n1:n1c].T,
                                                    neuron_activations[dt:,  n2:n2c].T)[:(n1c-n1),(n1c-n1):]
        
        # Set diagonal to zero at zero offset (obviously perfectly correlated)
        if dt == 0:
            H[:,:,0] = H[:,:,0]-np.diag(np.diag(H[:,:,0]))
    H = np.mean(np.abs(H), axis=2)
    # Self connections are not possible
    np.fill_diagonal(H,0)
    return H

def infer_precision_pca(neuron_activations):
    # Based on winning solution of the competition
    pca = PCA(whiten=True, n_components=int(0.8 * neuron_activations.shape[1])).fit(neuron_activations)
    try:
        H = -pca.get_precision()
        np.fill_diagonal(H,0)
        return H
    except:
        return np.zeros([neuron_activations.shape[1],neuron_activations.shape[1]])


def infer_precision_glasso(neuron_activations):
    graph_lasso = GraphLassoCV()
    graph_lasso.fit(neuron_activations)
    H = -graph_lasso.precision_ 
    np.fill_diagonal(H,0)
    return H


def compute_joint_prob(X,Y):
   X_1 = X[1:,] 
   X = X[:(len(X)-1),] 
   Y = Y[:(len(Y)-1),] 
   
   joint = {}
   for c in set(X_1):
       for c_x in set(X):
           for c_y in set(Y):
              j = (np.array(X_1 == c)+0)+(np.array(X == c_x)+0)+(np.array(Y == c_y)+0)
              joint[(c,c_x,c_y)] = np.mean(j==3)
                 
   return joint


def transfer_entropy(X,Y):
    joint = compute_joint_prob(X,Y)
    s = 0
    for i in range(0,len(X)-1):
        #p(x_n+1|x_n,y_n)  =
        numerator = joint[(X[i+1],X[i],Y[i])]/( joint[(0,X[i],Y[i])]+joint[(1,X[i],Y[i])] )
        #p(x_n+1|x_n)
        denom = ( joint[(X[i+1],X[i],0)]+joint[(X[i+1],X[i],1.0)] ) / (np.sum([joint[(i,X[i],0)] for i in [0,1]])+np.sum([joint[(i,X[i],1)] for i in [0,1]]))
            
        s+= joint[(X[i+1],X[i],Y[i])]*np.log2(numerator/denom)
    return s  


def infer_transfer_entropy(neuron_activations):
    n = neuron_activations.shape[1]
    connectivity_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if(i!=j):
                X = neuron_activations[:,i]
                Y = neuron_activations[:,j]
                start_time = time.time()
                connectivity_mat[i,j] = transfer_entropy(X,Y)
                t = time.time() - start_time
                print(t)
                
    return connectivity_mat
            



log = file("../Data/results/time_model_free.txt","w")
disc_meth = "oasis"
for i in range(1,7) :

    print("Now in :"+str(i)+"-"+disc_meth)
   
    activations_loc = "../Data/small/discretized_"+disc_meth+"_"+str(i)+".csv" 
    neuron_activations = pd.read_csv(activations_loc)
    neuron_activations = neuron_activations.values
    
    #---------- Run all the metrics and store the connectivity results and computational time
    start_time = time.time()

    correlation_mat = infer_correlation( neuron_activations )
    pd.DataFrame(correlation_mat).to_csv("../Data/results/correlation_"+disc_meth+"_"+str(i)+".csv")

    t = time.time() - start_time
    log.write("correlation time "+str(t)+" "+str(i)+" "+disc_meth)
    log.write("\n")
    
    
    start_time = time.time()

    precision_mat1 = infer_precision_pca( neuron_activations )
    pd.DataFrame(precision_mat1).to_csv("../Data/results/pca_"+disc_meth+"_"+str(i)+".csv")

    t = time.time() - start_time
    log.write("precision pca time "+str(t)+" "+str(i)+" "+disc_meth)
    log.write("\n")
    
    
    start_time = time.time()

    precision_mat2 = infer_precision_glasso( neuron_activations )
    pd.DataFrame(precision_mat2).to_csv("../Data/results/glasso_"+disc_meth+"_"+str(i)+".csv")

    t = time.time() - start_time
    log.write("precision glasso time "+str(t)+" "+str(i)+" "+disc_meth)
    log.write("\n")
    
log.close()



