#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:06:49 2017

@author: george

Preprocess the 6 datasets and store the two different discretized versions of the dataset
"""

import time
import pandas as pd
from _1_utils import parse_activations, parse_neuron_positions,unscatter, winning_discretization, chalearn_discretization,extract_spike_oopsi


log = file("../results/time_preprocess2.txt","w")
for i in range(1,7):
    activations_loc = "../data/small/fluorescence_iNet1_Size100_CC0"+str(i)+"inh.txt" 
    positions_loc = "../data/small/networkPositions_iNet1_Size100_CC0"+str(i)+"inh.txt"
    
    print(i)
    neural_activations = parse_activations(activations_loc,partial=False)
    positions = parse_neuron_positions(positions_loc)

    print("removing light scattering")
    #---------_ Remove light scattering
    neural_activations = unscatter(neural_activations.T,positions)
    
    start_time = time.time()
    print("discretizing using chalearn")
    #---------_ Discretization using the ChaLearn approach
    neural_dics2 = chalearn_discretization(neural_activations)
    neural_dics2 = neural_dics2[:(neural_dics2.shape[0]-3),:]
    pd.DataFrame(neural_dics2).to_csv("../data/small/discretized_chalearn_"+str(i)+".csv")
    t = time.time() - start_time
    print(t)
    log.write("chalearn "+str(t)+" "+str(i))
    log.write("\n")
    
    start_time = time.time()
    print("discretizing using winning") 
    #---------_ Discretization of the winning solution
    neural_disc1 = winning_discretization(neural_activations)
    neural_disc1 = neural_disc1[:(neural_disc1.shape[0]-3),:]
    pd.DataFrame(neural_disc1).to_csv("../data/small/discretized_winning_"+str(i)+".csv")
    t = time.time() - start_time
    print(t)
    log.write("winning "+str(t)+" "+str(i))
    log.write("\n")

log.close()


