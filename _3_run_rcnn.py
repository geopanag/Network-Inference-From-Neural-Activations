"""
@author: george

RCNN based on https://github.com/spoonsso/tfomics/blob/master/example/Connectomics.ipynb
"""
import os
os.chdir("/Path/To/Code")

import time
import numpy as np
from _1_utils import downsample,standardize_rows, parse_activations, parse_neuron_positions,parse_neuron_connections,unscatter
import sys

sys.path.append('../Tensor/tfomics')
from tfomics import neuralnetwork as nn
from tfomics import learn
from model_zoo import residual_connectomics_model4
import conutils

np.random.seed(2017)

#-------------------------------- Load the data
#---------- Keep the connectivity, activation, and partial correlations of each network in a position of the lists below
connectivities = []
activations = []
partial_corr = []

for i in range(1,7):
    print(i)
    #--------- Load activations
    activations_loc = "../Data/small/fluorescence_iNet1_Size100_CC0"+str(i)+"inh.txt" 
    neural_activations = parse_activations(activations_loc,partial=False)
    
    #---------  Compute partial correlations
    pcorr = conutils.get_partial_corr_scores(neural_activations.T)
    #--------- Standardize partial correlation coefficients
    pcorr[pcorr==0] = np.min(pcorr[pcorr!=0])
    pcorr = pcorr - np.mean(pcorr)
    pcorr = pcorr/np.std(pcorr)
    partial_corr.append(pcorr)
    
    positions_loc = "../Data/small/networkPositions_iNet1_Size100_CC0"+str(i)+"inh.txt"
    positions = parse_neuron_positions(positions_loc)
    
    #--------- Unscatter, downsample, standardize
    neural_activations = standardize_rows(downsample(unscatter(neural_activations.T,positions).T).T)
    
    activations.append(neural_activations.T)
    
    #--------- Connectivity matrices
    network_loc = "../Data/small/network_iNet1_Size100_CC0"+str(i)+"inh.txt"
    neuron_connections = parse_neuron_connections(network_loc)
    connectivities.append(neuron_connections)
    

#-------------------------------- Corss Validation (1 network test, 5 train)
log = file("../Data/results/time_rcnn.txt","a")


for i in range(0,6):
    print("dataset "+str(i))
    
    start = time.time()
    
    #-------------- Define which network is test and which train
    test_act = activations[i]
    train_act = activations[:i] + activations[(i + 1):]
    
    test_conn = connectivities[i]
    train_conn = connectivities[:i] + connectivities[(i + 1):]
    
    test_pcor = partial_corr[i]
    train_pcor = partial_corr[:i]+partial_corr[(i+1):]
    
    #-------------- Create the training dataset out of the 5 time series datasets
    dtf, ltf = conutils.pairwise_prep_tuple_partialcorr(tuple(train_act),tuple(train_conn),tuple(train_pcor),num_samples = 320)
    
    #-------------- Randomly permute the samples
    inds = np.random.choice(dtf.shape[0],replace=False,size=dtf.shape[0])
    X_train = dtf[inds,:,:,:]
    y_train = ltf[inds]
    
    num_data, height, width, dim = X_train.shape
    input_shape=[None, height, width, dim]
    num_labels = y_train.shape[1]  
    
    #--------- Load model
    net, placeholders, optimization = residual_connectomics_model4.model(input_shape, num_labels)

    #--------- Build neural network class and compile it
    nnmodel = nn.NeuralNet(net, placeholders)
    #nnmodel.inspect_layers()

    nntrainer = nn.NeuralTrainer(nnmodel, optimization, save='best', filepath="../Data/results/residual_"+str(i)+".txt")
    
    #---------  Train the NN
    train = {'inputs': X_train, 'targets': y_train, 'keep_prob_conv': 0.8, 'keep_prob_dense': 0.5, 'is_training': True}
    data = {'train': train}
<<<<<<< HEAD

    learn.train_minibatch(nntrainer, data, batch_size=100, num_epochs=100, 
=======
    learn.train_minibatch(nntrainer, data, batch_size=100, num_epochs=200, 
>>>>>>> 48330085e47c63f92a0f2ee211e8889b8e93bb10
                          patience=20, verbose=2, shuffle=True)

    #---------  Predict the connectivity of the test network
    pred_lbl =  conutils.valid_eval_tfomics_partialcorr(nntrainer,test_act,test_pcor,fragLen=320)
    pred_lbl = np.reshape(pred_lbl, (100, 100))
    
    #---------  Store the predictions
    np.savetxt("../Data/results/rcnn_"+str(i+1)+".csv", pred_lbl, delimiter=",")
    
    t = time.time()-start
    print("Time:"+str(t))
    log.write(str(t))
    log.write("\n")
    
log.close()
