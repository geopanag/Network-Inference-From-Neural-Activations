#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george
"""
import time
import pickle
import os

os.chdir("Path/To/Code")
import gzip
import numpy as np
import pandas as pd

from pyhawkes.utils.basis import IdentityBasis
from pyhawkes.models import DiscreteTimeNetworkHawkesModelGammaMixtureSBM


def fit_network_hawkes_svi(S, K, C, dt, dt_max,
                           output_path,
                           standard_model=None,
                           N_iters=100,
                           true_network=None):
    """
    From Scott Linderman's experiments in https://github.com/slinderman/pyhawkes/tree/master/experiments
    """
    # Check for existing Gibbs results
    if os.path.exists(output_path + ".svi.pkl.gz"):
        with gzip.open(output_path + ".svi.pkl.gz", 'r') as f:
            print("Loading SVI results from ", (output_path + ".svi.pkl.gz"))
            (samples, timestamps) = pickle.load(f)
    elif os.path.exists(output_path + ".svi.itr%04d.pkl" % (N_iters-1)):
        with open(output_path + ".svi.itr%04d.pkl" % (N_iters-1), 'r') as f:
            print("Loading SVI results from ", (output_path + ".svi.itr%04d.pkl" % (N_iters-1)))
            sample = pickle.load(f)
            samples = [sample]
            timestamps = None
            # (samples, timestamps) = cPickle.load(f)

    else:
        print("Fitting the data with a network Hawkes model using SVI")

        #------------- Make a new model for inference
        test_basis = IdentityBasis(dt, dt_max, allow_instantaneous=True)
        E_W = 0.01
        kappa = 10.
        E_v = kappa / E_W
        alpha = 10.
        beta = alpha / E_v
        network_hypers = {'C': 2,
                          'kappa': kappa, 'alpha': alpha, 'beta': beta,
                          'p': 0.8,
                          'allow_self_connections': False}
        test_model = DiscreteTimeNetworkHawkesModelGammaMixtureSBM(K=K, dt=dt, dt_max=dt_max,
                                                                basis=test_basis,
                                                                network_hypers=network_hypers)
        #------------- Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)
        minibatchsize = 3000
        test_model.add_data(S)

        #------------- Stochastic variational inference learning with default algorithm hyperparameters
        samples = []
        delay = 10.0
        forgetting_rate = 0.5
        stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
        timestamps = []
        for itr in range(N_iters):

            print("SVI Iter: ", itr, "\tStepsize: ", stepsize[itr])
            test_model.sgd_step(minibatchsize=minibatchsize, stepsize=stepsize[itr])
            test_model.resample_from_mf()
            samples.append(test_model.copy_sample())
            timestamps.append(time.clock())
        
            with open(output_path + ".svi.itr%04d.pkl" % itr, 'w') as f:
                pickle.dump(samples[-1], f, protocol=-1)

        with gzip.open(output_path + ".svi.pkl.gz", 'w') as f:
            print("Saving SVI samples to ", (output_path + ".svi.pkl.gz"))
            pickle.dump((samples, timestamps), f, protocol=-1)

    return samples, timestamps


#--------- Default model hyperparamaters 
C      = 1
dt     = 0.02
dt_max = 0.08

np.random.seed(2017)

log = file("../Data/results/time_hawkes.txt","a")
for i in range(1,7):
        
    print("Now in :"+str(i))
    
    neuron_activations = pd.read_csv("../Data/small/discretized_oasis2_"+str(i)+".csv" )
    neuron_activations = neuron_activations.values
    neuron_activations = neuron_activations.astype(int)
    
	    #------- Run the hawkes model
    out = "../Data/hawkes/gibbs"+str(i)+"_"
    start_time = time.time()
    svi_models, timestamps = fit_network_hawkes_svi(neuron_activations, neuron_activations.shape[1], C, dt, dt_max, output_path=out)

    #------- Store the model and the connectivity matrix        
    W_svi = svi_models[-1].weight_model.expected_W()
    pd.DataFrame(W_svi).to_csv("../Data/results/hawkes_"+str(i)+".csv",index=False)
    pickle.dump( svi_models, open( "../Data/results/hawkes_"+str(i)+".p", "wb" ) )
    t = time.time() - start_time
    print(t)
    log.write("time "+str(t)+" "+str(i))
    log.write("\n")
    
log.close()
        
