#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:07:06 2017

@author: george

Run model free based inference in all 6 networks for the different discretization techniques
"""

import pandas as pd
from _3_model_free import infer_correlation, infer_precision_pca, infer_precision_glasso
import time

log = file("../results/time_metrics.txt","w")

for i in range(1,7) :
    
    for disc_meth in ["winning","chalearn"]:
        
        print("Now in :"+str(i)+"-"+disc_meth)
       
        activations_loc = "../data/small/discretized_"+disc_meth+"_"+str(i)+".csv" 
        neuron_activations = pd.read_csv(activations_loc)
        #---------- Delete row names and keep the numpy array matric
        neuron_activations = neuron_activations.drop(neuron_activations.columns[0], 1)
        neuron_activations = neuron_activations.values
        
        
        #---------- Run all the metrics and store the connectivity results and computational time
        start_time = time.time()

        correlation_mat = infer_correlation( neuron_activations )
        pd.DataFrame(correlation_mat).to_csv("../results/correlation_"+disc_meth+"_"+str(i)+".csv")

        t = time.time() - start_time
        log.write("correlation time "+str(t)+" "+str(i)+" "+disc_meth)
        log.write("\n")
        
        
        start_time = time.time()

        precision_mat1 = infer_precision_pca( neuron_activations )
        pd.DataFrame(precision_mat1).to_csv("../results/pca_"+disc_meth+"_"+str(i)+".csv")

        t = time.time() - start_time
        log.write("precision pca time "+str(t)+" "+str(i)+" "+disc_meth)
        log.write("\n")
        
        
        start_time = time.time()

        precision_mat2 = infer_precision_glasso( neuron_activations )
        pd.DataFrame(precision_mat2).to_csv("../results/glasso_"+disc_meth+"_"+str(i)+".csv")

        t = time.time() - start_time
        log.write("precision glasso time "+str(t)+" "+str(i)+" "+disc_meth)
        log.write("\n")
        
log.close()



