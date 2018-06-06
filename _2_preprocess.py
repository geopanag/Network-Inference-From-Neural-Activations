#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george

Preprocess the 6 datasets and store the discretized versions
"""
import os
os.chdir("/home/george/Desktop/Network-Inference-From-Neural-Activations/Code")


import time
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from _1_utils import parse_activations, parse_neuron_positions,unscatter
from oasis.functions import deconvolve

def discretize(x):
    x[x>=0.12]=1
    x[x<0.12]=0
    return x

log = file("../Data/results/time_preprocess.txt","a")

n = []
n_dis = []
for f in range(1,7):
    print(f)
    
    #----------- Read the data
    activations_loc = "../Data/small/fluorescence_iNet1_Size100_CC0"+str(f)+"inh.txt" 
    positions_loc = "../Data/small/networkPositions_iNet1_Size100_CC0"+str(f)+"inh.txt"
    
    neural_activations = parse_activations(activations_loc,partial=False)
    positions = parse_neuron_positions(positions_loc)
    

    print("removing light scattering")
    start_time = time.time()
    #---------_ Remove light scattering
    neural_activations = unscatter(neural_activations.T,positions)
    n.append(neural_activations[1:3000,65])    
    #---------- Discretize with the threshold-based method proposed in chalearn competition
    print("discretizing using oasis")    
    
    neural_dis = np.apply_along_axis(lambda x: deconvolve(x, penalty=1)[1],0,neural_activations)
    neural_dis = pd.DataFrame(neural_dis).apply(discretize,0)
    n_dis.append(neural_dis.iloc[1:3000,65])
    
    #neural_dis.to_csv("../Data/small/discretized_oasis_"+str(f)+".csv",index=False)
    
    t = time.time() - start_time
    print(t)
    log.write("oasis "+str(t)+" "+str(f))
    log.write("\n")
    
log.close()



#---------- Plot for the poster an example of discretization for the poster

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) 

#for i in range(0,99):
#    plt.figure(i)
#    plt.plot(neural_activations[1:3000,i])
#    plt.savefig("../Figures/value" + str(i) + ".png")


plt.figure(figsize=(10.0, 5.0))    
f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(n[1])
axarr[1].plot(n[5])
axarr[2].plot(n[3])
axarr[0].set_ylabel('Cell 1', fontsize=16)
axarr[0].get_xaxis().set_visible(False)
axarr[1].set_ylabel('Cell 2', fontsize=16)
axarr[1].get_xaxis().set_visible(False)
axarr[2].set_ylabel('Cell 3', fontsize=16)
axarr[2].get_xaxis().set_visible(False)
f.savefig("../Figures/activations.png")


f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(n_dis[1])
axarr[0].get_xaxis().set_visible(False)
axarr[1].plot(n_dis[5])
axarr[1].get_xaxis().set_visible(False)
axarr[2].plot(n_dis[3])
axarr[2].get_xaxis().set_visible(False)
f.savefig("../Figures/spikes.png")

