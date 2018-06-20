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
os.chdir("/Path/To/Code")

from _1_utils import parse_neuron_connections,compute_auc_roc, compute_auc_prc

log = file("../Data/results/metrics.csv","w")


#---------- Parse the different networks
for i in range(1,7) :
    
    network_loc = "../Data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
    neuron_connections = parse_neuron_connections(network_loc)
    
    print("pair "+str(i))
 
    #----------- influence
    influence = np.array(pd.read_csv("../Data/results/influence_"+str(i)+".csv",header=None))
    if(influence.shape[1]==101):
        influence = np.delete(influence,0,1)
    
    np.fill_diagonal(influence,0)     
    aucs = compute_auc_roc(neuron_connections, influence)
    prcs = compute_auc_prc(neuron_connections, influence)
    log.write(",".join([str(aucs),str(prcs),str(i),"oasis","influence"]))
    log.write("\n")
    
    
    #----------- rcnn
    rcnn = np.array(pd.read_csv("../Data/results/rcnn_"+str(i)+".csv",header=None))
    if(rcnn.shape[1]==101):
        rcnn = np.delete(rcnn,0,1)
    
    np.fill_diagonal(rcnn,0)  #---------------------------------------------------------????????????????????
    aucs = compute_auc_roc(neuron_connections, rcnn)
    prcs = compute_auc_prc(neuron_connections, rcnn)
    log.write(",".join([str(aucs),str(prcs),str(i),"oasis","rcnn"]))
    log.write("\n")

     
    #----------- hawkes
    hawkes = np.array(pd.read_csv("../Data/results/hawkes_"+str(i)+".csv"))
    if(hawkes.shape[1]==101):
        hawkes = np.delete(hawkes,0,1)
    
    aucs = compute_auc_roc(neuron_connections, hawkes)
    prcs = compute_auc_prc(neuron_connections, hawkes)
    log.write(",".join([str(aucs),str(prcs),str(i),"oasis","hawkes"]))
    log.write("\n")
    
    #----------- pca
    pca = np.array(pd.read_csv("../Data/results/pca_"+str(i)+".csv"))
    if(pca.shape[1]==101):
        pca = np.delete(pca,0,1)
    
    pca = pca/np.max(pca)
    aucs = compute_auc_roc(neuron_connections, pca)
    prcs = compute_auc_prc(neuron_connections, pca)
    log.write(",".join([str(aucs),str(prcs),str(i),"oasis","pca"]))
    log.write("\n")
    
    #----------- cross correlation
    corr = np.array(pd.read_csv("../Data/results/correlation_"+str(i)+".csv"))
    if(corr.shape[1]==101):
        corr = np.delete(corr,0,1)
    
    aucs = compute_auc_roc(neuron_connections, corr)
    prcs = compute_auc_prc(neuron_connections, corr)
    log.write(",".join([str(aucs),str(prcs),str(i),"oasis","corr"]))
    log.write("\n")
    
    
    #----------- graphical lasso
    glasso = np.array(pd.read_csv("../Data/results/glasso_"+str(i)+".csv"))
    if(glasso.shape[1]==101):
        glasso = np.delete(glasso,0,1)
    glasso = glasso+np.max(glasso)
    glasso = glasso/np.max(glasso)
    np.fill_diagonal(glasso,0)  
    
    aucs = compute_auc_roc(neuron_connections, glasso)
    prcs = compute_auc_prc(neuron_connections, glasso)
    log.write(",".join([str(aucs),str(prcs),str(i),"oasis","glasso"]))
    log.write("\n")
    
log.close()


#--------- Read the results and print the average metrics to add in the report
metrics = pd.read_csv("../Data/results/metrics.csv",header=None)

metrics.groupby([3,4])[1].mean().to_csv("../Data/results/metrics_prc.csv")
metrics.groupby([3,4])[0].mean().to_csv("../Data/results/metrics_arc.csv")


#--------- Read the results and print the average time for model free methods
model_time = pd.read_csv("../Data/results/time_model_free.txt",sep=" ",header=None)
model_time = model_time.drop([1,4],axis=1)
hawkes_time = pd.read_csv("../Data/results/time_hawkes.txt",sep=" ",header=None)
rcnn_time = pd.read_csv("../Data/results/time_rcnn.txt",sep=" ",header=None)
influence_time = pd.read_csv("../Data/results/time_influence.txt",sep=" ",header=None)

results_time = model_time.groupby(0)[2].mean().to_frame()
results_time.reset_index(inplace=True)

results_time = results_time.append(pd.DataFrame({0:"hawkes",2:[float(np.mean(hawkes_time))]}))
results_time = results_time.append(pd.DataFrame({0:"influence",2:[influence_time.groupby([2])[1].sum().mean()+30]}))
results_time[2]+=30 #------ Adding processing time
results_time = results_time.append(pd.DataFrame({0:"rcnn",2:[float(np.mean(rcnn_time))]}))

results_time.to_csv("../Data/results/time_overall.csv")