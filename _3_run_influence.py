#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: george
"""

import os
os.chdir("/Path/To/Code")

import pandas as pd
import time
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from _1_utils import parse_neuron_connections


def find_next_impulse(x,s):
    if(len(x)>0):
        return x[0]-s
    else:
        return 0
    
#---- An  impulse  can  be eattributed to a spike before spike s, with a span  of 50 samples (1 sec)
#---- This is an effort to capture triangle effects  (cause->spike at t1, cause->impulse at t2,  spike!->impulse at t3)
false_causation = 50 
def find_previous_impulse(impulse,causes):
    for c in causes:
        if((c>(impulse-false_causation)).any()):
            return False
    return True
    
log = file("../Data/results/time_influence.txt","a")

#---- Over half of the network needs to burst at the same time to be considered invalid
total_net_activity_threshold = 70
def remove_network_bursts(x):
    #----- Remove network bursts or total silent samples
    spike_times = x[0:99].value_counts()
    if(spike_times.index[0]!=0):
        if(spike_times.iloc[0]>total_net_activity_threshold):
            return False
    else:
        if(spike_times.iloc[0]>=99):
            return False
    return True

#---------------- Compute the impulses
for f in range(1,7) :
    start_time = time.time()   
    print("Now in file:"+str(f))
    #------------ Compute impulses
    neuron_activations = pd.read_csv("../Data/small/discretized_oasis_"+str(f)+".csv" )    
    neuron_activations = neuron_activations.values
    N = neuron_activations.shape[1]
    
    #--- Transform spike trains into list of spike positions arrays
    spikes_idx = pd.DataFrame(neuron_activations).apply(lambda x: np.where(x>0),0)
    spikes_idx = [np.array(s)[0,:] for s in spikes_idx]
    
    #--------- Store the times of impulses of all neurons after each spike of n1
    impulses_times =  []
    impulses_times_triangles =  []
    
    
    #------------- For each spike of a neuron
    for n1 in range(0,len(spikes_idx)):
        print("In neuron:"+str(n1))
        
        spikes = spikes_idx[n1]
        
        #--------- List of all spikes of all neurons
        spikes_for_n1 = spikes_idx
        possible_causes = spikes_idx
        
        #------------- Find the next spike of the other neurons (possibly caused by it)
        for s in spikes:
        
            #--------------- Reduce spikes_for_n1 every time, to keep only the spikes after s
            spikes_for_n1 = map(lambda x: x[np.where(x>s)], spikes_for_n1)
            
            #--------------- Define the impulse response as the first spike after s for each neuron
            possible_impulses = np.array([find_next_impulse(sn1,s) for sn1 in spikes_for_n1]) 
            
            #------ Refrain from triangles
            #--------------- Keep the exact preceding spikes, as possible causes of the impulse responses
            possible_causes = map(lambda x: x[np.where(x>=(s-false_causation))], possible_causes)
            
            #--------------- Subset to the possible causes that are before spike s
            possible_causes_for_s_impulses = map(lambda x: x[np.where(x<s)], possible_causes)
            
            impulses_times.append(np.append(possible_impulses,n1))
            
            #-------------- If the impulse is less then DT from the previous spike, do not assign them as impulses to s
            possible_impulses[[find_previous_impulse(s+spike,possible_causes_for_s_impulses) for spike in possible_impulses]] = 0
            impulses_times_triangles.append(np.append(possible_impulses,n1))
    
    impulses_times = pd.DataFrame(np.asarray(impulses_times))
    impulses_times_triangles = pd.DataFrame(np.asarray(impulses_times_triangles))
    
    #------------ Remove cases where the whole network bursts
    impulses_times = impulses_times.loc[impulses_times.apply(remove_network_bursts, axis=1),:]
    impulses_times_triangles = impulses_times_triangles.loc[impulses_times_triangles.apply(remove_network_bursts, axis=1),:]
    
    impulses_times_triangles_scores = impulses_times_triangles.iloc[:,0:100].apply(lambda x:np.exp(1.0/(x))-1,axis=0)
    impulses_times_triangles_scores['label'] = impulses_times_triangles.iloc[:,100]
    impulses_times_triangles_scores[np.isinf(impulses_times_triangles_scores)] = 0
     
    impulses_times_triangles_scores = pd.DataFrame(np.asarray(impulses_times_triangles_scores))
    impulses_times_triangles_scores.to_csv("../Data/small/impulse_times_triangles_scores_"+str(f)+".csv",index=False)

    t = time.time() - start_time
    log.write("impulse "+str(f)+" " +str(t))
    log.write("\n")    


#---------------- Create the dataset
print("computed the impulses, now creating the dataset")


for f in range(1,7) :
    start_time = time.time()  
    print("Doing Dataset:"+str(f))
    final = []
    impulses_times = pd.read_csv("../Data/small/impulse_times_triangles_scores_"+str(f)+".csv")
    
    #------------ Form the dataset with feature extraction
    connections = parse_neuron_connections("../Data/small/network_iNet1_Size100_CC0"+str(f)+"inh.txt")
    
    for n1 in range(0,100):
        print("In neuron :"+str(n1))
        edges =  impulses_times.loc[impulses_times.iloc[:,100]==n1, impulses_times.columns[np.where(connections[n1,:]==1)[0]]]
        absent = impulses_times.loc[impulses_times.iloc[:,100]==n1, impulses_times.columns[np.where(connections[n1,:]==0)[0]]]
        
        for n2 in edges.columns:
            dt = edges.loc[:,n2]
            
            dt = dt[dt!=0]
           
            #------ Number of times N2 copied N1 relative to the total number of N1's spikes
            spike_impulse_percentage = len(dt)/float(edges.shape[0])
            
            #------ Simple Features
            mean_copy_time = np.mean(dt)
            variance_copy_time = np.var(dt)
            
            final.append(('n_'+str(n1)+'_'+str(n2), mean_copy_time, variance_copy_time,spike_impulse_percentage,dt.quantile(.95), 1))                   
    
        for n2 in absent.columns:
            dt = absent.loc[:,n2]
            
            dt = dt[dt!=0]
            
            #------ Number of time N2 copied N1 relative to the total number of N1's spikes
            spike_impulse_percentage = len(dt)/float(absent.shape[0])
            #------ Simple Features
            mean_copy_time = np.mean(dt)
            variance_copy_time = np.var(dt)
            
            final.append(('n_'+str(n1)+'_'+str(n2), mean_copy_time, variance_copy_time,spike_impulse_percentage,dt.quantile(.95),0))                   

    final = pd.DataFrame(np.asarray(final))
    
    final.to_csv("../Data/small/influence_"+str(f)+".csv",index=False)
    
    t = time.time() - start_time
    log.write("scores "+str(f)+" " +str(t))
    log.write("\n")    



#--------------- Machine learning with leave one network out validation
print("running cross validation")

aucs = []
prcs = []
for f in range(1,7) :
    
    print("Test dataset now:"+str(f))
    
    start_time = time.time()
    #------------- Leave one network out cross validation
    test = pd.read_csv("../Data/small/influence_"+str(f)+".csv")
    
    train = pd.DataFrame()
    for j in range(1,7):
        if(f!=j):
            train = train.append(pd.read_csv("../Data/small/influence_"+str(j)+".csv"))
    
    #------------- Keep the pairs to reconstruct the weighted matrix of predictions
    test_pairs = test[test.columns[0]]
    
    train = train.drop(train.columns[0], axis=1)
    test = test.drop(test.columns[0], axis=1)
     
    test = test.apply(pd.to_numeric)
    train = train.apply(pd.to_numeric)
    
    #----------- Balanced SVM
    clf = SVC(class_weight='balanced',probability=True)#
    clf.fit(train.as_matrix(train.columns[0:4]),train.iloc[:,4])
    X_test = test.as_matrix(test.columns[0:4])
  
    #------------ Prediction
    predictions = [p[1] for p in clf.predict_proba(X_test)]
  
    adj = np.zeros([100,100])
    for pair in range(0,len(test_pairs)):
        adj[int(test_pairs[pair].split("_")[1]),int(test_pairs[pair].split("_")[2])] =  predictions[pair]
    
    #---------  Store the predictions
    np.savetxt("../Data/results/influence_"+str(f)+".csv",adj, delimiter=",")
    
    t = time.time() - start_time
    log.write("classification " +str(f)+" "+str(t))
    log.write("\n")

    aucs.append(roc_auc_score(test.iloc[:,4],predictions))
    prcs.append(average_precision_score(test.iloc[:,4],predictions))

results = pd.DataFrame({'auc':aucs,'prc':prcs})
results.to_csv("../Data/results/influence_results_validation.csv",index=False)

#--------------- Machine learning with traditional cross validation
train = pd.DataFrame()
for f in range(1,7) :
    print("Now in dataset:"+str(f))
    train = train.append(pd.read_csv("../Data/small/influence_"+str(f)+".csv"))   #"times"+c+"_"+str(f)+".csv"))
    
train = train.drop(train.columns[0], axis=1)
train = train.apply(pd.to_numeric)

scoring = ['accuracy','roc_auc','precision', 'recall']
clf = SVC(class_weight='balanced',probability=True)#
scores = cross_validate(clf, train.as_matrix(train.columns[0:4]),train.iloc[:,4], cv=5,scoring = scoring,return_train_score=False)

with open('../Data/results/influence_fold_validation.txt', 'w') as f:
    f.write(str(np.mean(scores['test_precision'])))
    f.write("\n")
    f.write(str(np.mean(scores['test_roc_auc'])))
