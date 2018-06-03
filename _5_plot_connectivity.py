#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

from _1_utils import parse_neuron_connections


disc_meth = "oasis"

fig = plt.figure(1)
    
fig.set_size_inches(19, 11)
for i in range(1,7):
    #------------ Load ground truth
    network_loc = "../Data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
    neuron_connections = parse_neuron_connections(network_loc)
    
    #------------ Load precision
    glasso = np.array(pd.read_csv("../results/glasso_"+disc_meth+"_"+str(i)+".csv"))
    glasso = np.delete(glasso,0,1)
    glasso = -glasso+np.max(glasso)
    glasso = glasso/np.max(glasso)
    np.fill_diagonal(glasso,0)  
    
    #------------ Load hawkes
    hawkes = np.array(pd.read_csv("../results/hawkes_"+str(i)+"_"+disc_meth+".csv"))
    hawkes = np.delete(hawkes,0,1)
    hawkes = hawkes/np.max(hawkes)
    
    #------------ Load hawkes
    rcnn = np.array(pd.read_csv("../results/rcnn_"+str(i-1)+".csv",header=None))
    rcnn = np.reshape(rcnn, (100, 100))
    rcnn = rcnn/np.max(rcnn)
    np.fill_diagonal(rcnn,0)   
    
    net = "Network "+str(i)
    
    idx = (i-1)*4+1
    plt.subplot(6,4,idx)
    plt.imshow(glasso, cmap='hot', interpolation='nearest')
    if(i==1):
        plt.title("Graphical Lasso",fontsize=18)
    plt.ylabel(net,fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    plt.subplot(6,4,idx+1)
    plt.imshow(hawkes, cmap='hot', interpolation='nearest')
    if(i==1):
        plt.title("Hawkes",fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    plt.subplot(6,4,idx+2)
    plt.imshow(rcnn, cmap='hot', interpolation='nearest')
    if(i==1):
        plt.title("RCNN",fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    plt.subplot(6,4,idx+3)
    plt.imshow(neuron_connections, cmap='hot', interpolation='nearest')
    if(i==1):
        plt.title("True Connectivity",fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
fig.savefig('../figures/connectivity_heatmaps.png')


