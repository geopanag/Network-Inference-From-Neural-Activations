#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george
"""

import os
os.chdir("/Path/To/Code")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _1_utils import parse_neuron_connections

def to_binary(x):
    x[x>=0.5]=1
    x[x<0.5]=0
    return x
    

fig, axes = plt.subplots(6, 5)
    
fig.set_size_inches(19, 11)

for i in range(1,7):
    #------------ Load ground truth
    network_loc = "../Data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
    neuron_connections = parse_neuron_connections(network_loc)
    
    #------------ influence
    influence = np.array(pd.read_csv("../Data/results/influence_"+str(i)+".csv",header=None))
    if(influence.shape[1]==101):
        influence = np.delete(influence,0,1)
    np.fill_diagonal(influence,0)  
    
    
    #------------  precision
    glasso = np.array(pd.read_csv("../Data/results/glasso_"+str(i)+".csv"))
    glasso = np.delete(glasso,0,1)
    np.fill_diagonal(glasso,0)  
    
    #------------  hawkes
    hawkes = np.array(pd.read_csv("../Data/results/hawkes_"+str(i)+".csv"))
    hawkes = np.delete(hawkes,0,1)
    #------------  rcnn
    rcnn = np.array(pd.read_csv("../Data/results/rcnn_"+str(i)+".csv",header=None))
    np.fill_diagonal(rcnn,0)   
    
    net = "Network "+str(i)
    
   
    axes[i-1,0].imshow(influence, cmap='hot', interpolation='nearest')
    #plt.
    if(i==1):
        axes[i-1,0].set_title("CIRUSIM",fontsize=18)
    axes[i-1,0].set_xticks([])
    axes[i-1,0].set_yticks([])
    axes[i-1,0].set_ylabel(net,fontsize=18)

    axes[i-1,1].imshow(hawkes, cmap='hot', interpolation='nearest')
    if(i==1):
        axes[i-1,1].set_title("Hawkes",fontsize=18)
    axes[i-1,1].set_xticks([])
    axes[i-1,1].set_yticks([])

    axes[i-1,2].imshow(rcnn, cmap='hot', interpolation='nearest')
    if(i==1):
        axes[i-1,2].set_title("RCNN",fontsize=18)
    axes[i-1,2].set_xticks([])
    axes[i-1,2].set_yticks([])
  
    axes[i-1,3].imshow(glasso, cmap='hot', interpolation='nearest')
    if(i==1):
            axes[i-1,3].set_title("Glasso",fontsize=18)
    axes[i-1,3].set_xticks([])
    axes[i-1,3].set_yticks([])

    axes[i-1,4].imshow(neuron_connections, cmap='hot', interpolation='nearest')
    if(i==1):
        axes[i-1,4].set_title("Truth",fontsize=18)
    axes[i-1,4].set_xticks([])
    axes[i-1,4].set_yticks([])
    

plt.tight_layout()
fig.savefig('../Figures/connectivity_heatmaps.png')


