#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george

Evaluate the predictions of each algorithm for each discretization method
Store the results for each fold in a csv and compute the mean cross validation metrics
"""

import numpy as np
import pandas as pd
import os

os.chdir("/home/george/Desktop/Network-Inference-From-Neural-Activations/Code")

from _1_utils import parse_neuron_connections,compute_auc_roc, compute_auc_prc


log = file("../Data/results/metrics.csv","w")
disc_meth = "oasis"
#---------- Parse different datasets
for i in range(1,7) :
    
    network_loc = "../Data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
    neuron_connections = parse_neuron_connections(network_loc)
    
    print("pair "+str(i))
 
    #----------- hawkes
    hawkes = np.array(pd.read_csv("../Data/results/hawkes_"+str(i)+"_oasis.csv"))
    if(hawkes.shape[1]==101):
        hawkes = np.delete(hawkes,0,1)
    
    
    aucs = compute_auc_roc(neuron_connections, hawkes)
    prcs = compute_auc_prc(neuron_connections, hawkes)
    log.write(",".join([str(aucs),str(prcs),str(i),disc_meth,"hawkes"]))
    log.write("\n")
    
    #----------- pca
    pca = np.array(pd.read_csv("../Data/results/pca_"+disc_meth+"_oasis.csv"))
    if(pca.shape[1]==101):
        pca = np.delete(pca,0,1)
    
    pca = pca/np.max(pca)
    aucs = compute_auc_roc(neuron_connections, pca)
    prcs = compute_auc_prc(neuron_connections, pca)
    log.write(",".join([str(aucs),str(prcs),str(i),disc_meth,"pca"]))
    log.write("\n")
    
    #----------- cross correlation
    corr = np.array(pd.read_csv("../Data/results/correlation_"+disc_meth+"_oasis.csv"))
    if(corr.shape[1]==101):
        corr = np.delete(corr,0,1)
    
    aucs = compute_auc_roc(neuron_connections, corr)
    prcs = compute_auc_prc(neuron_connections, corr)
    log.write(",".join([str(aucs),str(prcs),str(i),disc_meth,"corr"]))
    log.write("\n")
    
    
    #----------- graphical lasso
    glasso = np.array(pd.read_csv("../Data/results/glasso_"+disc_meth+"_oasis.csv"))
    if(glasso.shape[1]==101):
        glasso = np.delete(glasso,0,1)
     
    glasso = glasso+np.max(glasso)
    glasso = glasso/np.max(glasso)
    np.fill_diagonal(glasso,0)  
    
    aucs = compute_auc_roc(neuron_connections, glasso)
    prcs = compute_auc_prc(neuron_connections, glasso)
    log.write(",".join([str(aucs),str(prcs),str(i),disc_meth,"glasso"]))
    log.write("\n")
    
log.close()


#--------- Read the results and print the average metrics to add in the report
metrics = pd.read_csv("../Data/results/metrics.csv",header=None)

metrics = metrics.loc[metrics[3]!='oasis']

metrics.groupby([3,4])[1].mean().to_csv("../Data/results/metrics_prc.csv")
metrics.groupby([3,4])[0].mean().to_csv("../Data/results/metrics_arc.csv")

#--------- Read the results and print the average time for model free methods
time = pd.read_csv("../Data/results/time_model_free.txt",sep=" ",header=None)
time = time.drop([1,4],axis=1)
time.groupby(0)[2].mean()


