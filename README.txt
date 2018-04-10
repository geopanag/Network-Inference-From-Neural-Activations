
The scripts follow the order of the number in the title. If two scripts have the same number, they can be run simultaneously.
The code was developed using python 2.7, in an ubuntu mate environment and tested in a laptop with Intel i7 CPU@2.4 GHz, 8 Gb RAM

Python packages used: pandas, sklearn, numpy, tensorflow, pickle, pyhawkes
RCNN is accompanied with folders:
	-conutils from https://github.com/spoonsso/TFconnect
	-tfomics and model_zoo from https://github.com/spoonsso/tfomics


The folder structure of the project is :

code->	  The content of this folder
data->	  Download the "small" dataset from https://www.kaggle.com/c/connectomics/data and extract it here. 
	  Each network has three .txt files: the activations, locations and ground truth connections of neurons
results-> Will be filled with estimated connectivity matrices, evaluation metrics and time logs


_1_utils contains functions (either costum or found online, with appropriate reference) for data preprocessing and evaluation of the algorithms.  
_2_preprocess runs the two discretization algorithms and produces discritized versions of the activation series in data/small
_3_model_free contains functions used by _4_run_model_free, which runs model free approaches to all networks in data/small and stores the results
_3_transfer_entropy runs in R without the need of any extra library, but takes arround 20 hours for one network. I could not find something more efficient in python. I have added my own transfer entropy implementation in _3_model_free, but it runs even slower.
_4_run_hawkes runsthe hawkes process based on pyhawkes
_4_run_rcnn runs residual convolutional neural network with the aid of conutiles, tfomics and model_zoo
_5_evaluate uses the predicted connectivity matrices stored in results and the ground truth in data/small to calculate evaluation metrics
_6_plot_connectivity plots the predicted connectivity matrices of all networks, for Hawkes, RCNN and Glasso.


