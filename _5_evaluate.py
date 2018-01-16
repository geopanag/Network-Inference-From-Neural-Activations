#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:06:49 2017

@author: george

Evaluate the predictions of each algorithm for each discretization method
Store the results for each fold in a csv and compute the mean cross validation metrics
"""

import numpy as np
import pandas as pd

from _1_utils import parse_neuron_connections,compute_auc_roc, compute_auc_prc


log = file("../results/metrics.csv","w")


#---------- Parse different datasets
for i in range(1,7) :
    
    network_loc = "../data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
    neuron_connections = parse_neuron_connections(network_loc)
    
    for j in ["winning","chalearn"]: #,"oopsi"]
        print("pair "+str(i)+"_"+j)
        network_loc = "../data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
        neuron_connections = parse_neuron_connections(network_loc)
 
    
        #----------- transfer entropy
        #te = np.array(pd.read_csv("../results/transfer_entropy_"+j+"_"+str(i)+".csv"))
        #te = np.delete(te,0,1)
        #te = np.nan_to_num(te)
        #te = te/np.max(te)
            
        #aucs, _, _ = compute_auc_roc(neuron_connections, te)
        #prcs, _, _ = compute_auc_prc(neuron_connections, te)
        #log.write(",".join([str(aucs),str(prcs),str(i),j,"te"]))
        #log.write("\n")
            
        #----------- rcnn
        rcnn = np.array(pd.read_csv("../results/rcnn_"+str(i-1)+".csv",header=None))
        rcnn = np.reshape(rcnn, (100, 100))
        rcnn = rcnn/np.max(rcnn)
        np.fill_diagonal(rcnn,0)   
        
        aucs, _, _ = compute_auc_roc(neuron_connections, rcnn)
        prcs, _, _ = compute_auc_prc(neuron_connections, rcnn)
        log.write(",".join([str(aucs),str(prcs),str(i),j,"rcnn"]))
        log.write("\n")
        
        #----------- hawkes
        hawkes = np.array(pd.read_csv("../results/hawkes_"+str(i)+"_"+j+".csv"))
        hawkes = np.delete(hawkes,0,1)
        
        aucs, _, _ = compute_auc_roc(neuron_connections, hawkes)
        prcs, _, _ = compute_auc_prc(neuron_connections, hawkes)
        log.write(",".join([str(aucs),str(prcs),str(i),j,"hawkes"]))
        log.write("\n")
        
        #----------- pca
        pca = np.array(pd.read_csv("../results/pca_"+j+"_"+str(i)+".csv"))
        pca = np.delete(pca,0,1)
        pca = pca/np.max(pca)
        
        aucs, _, _ = compute_auc_roc(neuron_connections, pca)
        prcs, _, _ = compute_auc_prc(neuron_connections, pca)
        log.write(",".join([str(aucs),str(prcs),str(i),j,"pca"]))
        log.write("\n")
        
        
        #----------- cross correlation
        corr = np.array(pd.read_csv("../results/correlation_"+j+"_"+str(i)+".csv"))
        corr = np.delete(corr,0,1)
        
        aucs, _, _ = compute_auc_roc(neuron_connections, corr)
        prcs, _, _ = compute_auc_prc(neuron_connections, corr)
        log.write(",".join([str(aucs),str(prcs),str(i),j,"corr"]))
        log.write("\n")
        
        
        #----------- graphical lasso
        glasso = np.array(pd.read_csv("../results/glasso_"+j+"_"+str(i)+".csv"))
        glasso = np.delete(glasso,0,1)
        glasso = glasso+np.max(glasso)
        glasso = glasso/np.max(glasso)
        np.fill_diagonal(glasso,0)  
        
        aucs, _, _ = compute_auc_roc(neuron_connections, glasso)
        prcs, _, _ = compute_auc_prc(neuron_connections, glasso)
        log.write(",".join([str(aucs),str(prcs),str(i),j,"glasso"]))
        log.write("\n")
        
log.close()


#--------- Read the results and print the average metrics to add in the report
metrics = pd.read_csv("../results/metrics.csv",header=None)
print(metrics.groupby([3,4])[1].mean())
print(metrics.groupby([3,4])[0].mean())

