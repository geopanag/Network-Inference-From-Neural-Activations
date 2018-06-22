
[A Review of Network Inference Techniques for Neural Activation Time Series](https://arxiv.org/abs/1806.08212)  <br />

The scripts follow the order of the number in the title. If two scripts have the same number, they can be run simultaneously.  <br />
The code was developed using python 2.7, in an ubuntu mate environment and tested in a laptop with Intel i7 CPU@2.4 GHz, 8 Gb RAM  <br />

Python packages used: pandas, sklearn, numpy, tensorflow, pickle, pyhawkes <br />
OASIS  for spike inference from https://github.com/j-friedrich/OASIS<br />
RCNN is accompanied with folders: <br />
	-conutils from https://github.com/spoonsso/TFconnect <br />
	-tfomics and model_zoo from https://github.com/spoonsso/tfomics <br />


The folder structure of the project is : <br />
Code-> The content of this folder <br />
Data-> Download the "small" dataset from https://www.kaggle.com/c/connectomics/data and extract it here. Each network has three .txt files: the neuron's activations, positions and ground truth connectivity <br />
Results-> Will be filled with estimated connectivity matrices, evaluation metrics and time logs <br />


_1_utils contains functions (either costum or found online, with appropriate reference) for data preprocessing and evaluation of the algorithms   <br />

_2_preprocess runs the discretization algorithms and produces discritized versions of the activation series <br />

_3_run_model_free runs correlation-based approaches <br />

_3_run_hawkes runs hawkes process model using on pyhawkes <br />

_3_run_rcnn runs residual convolutional neural network with the aid of conutiles, tfomics and model_zoo <br />

_3_run_influence runs the model that is based on influence estimation in social networks <br />

_4_evaluate uses the predicted connectivity matrices stored in results and the ground truth to calculate evaluation metrics <br />

_5_plot_connectivity plots the predicted connectivity matrices and the ground truth<br />


