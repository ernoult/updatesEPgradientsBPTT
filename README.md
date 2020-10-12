# Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input

This repository contains the code producing the results of [the paper](http://papers.nips.cc/paper/8930-updates-of-equilibrium-prop-match-gradients-of-backprop-through-time-in-an-rnn-with-static-input) "Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input", published as an oral contribution at NeurIPS 2019.

The following document provides details about the code provided, alongside the commands to be run to reproduce
the results appearing in the draft.

The project contains the following files:

  + `main.py`: executes the code, with arguments specified in a parser.

  + `netClasses.py`: contains the network classes.

  + `netFunctions.py`: contains the functions to run on the networks.

  + `plotFunctions.py`: contains the functions to plot the results. 


## Package requirements

Run the following command lines to set the environment using conda:
```
conda create --name EP python=3.6
conda activate EP
conda install -c conda-forge matplotlib
conda install pytorch torchvision -c pytorch
```


## Details about `main.py`


  `main.py` proceeds in the following way:

  + It first parses arguments typed in the terminal to build a network and get optimization parameters
  (i.e. learning rates, mini-batch size, network topology, etc.)

  + It loads the MNIST data set with torchvision.

  + It builds the nets using netClasses.py.

  + It takes one of the three actions that can be fed into the parser.


  The parser takes the following arguments:

  + Optimization arguments:
  
  |Arguments|Description|Examples|
|-------|------|------|
|`batch-size`|Training batch size.|`--batch-size 128`|
|`test-batch-size`|Test batch size used to compute the test error.|`--test-batch-size 128`|
|`epochs`|Number of epochs.| `--epochs 50`|
|`lr_tab` |Learning rates tab, to be provided from the output layer towards the first layer.|`--lr_tab 0.01 0.04` will apply a learning rate of 0.01 to W_{01} and 0.04 to W_{12}|
|`training-method`|Training method used, either Equilibrium Propagation or Backpropagation Through Time|`--training-method EP`,`--training-method BPTT`|
|`benchmark`|Trains two exact same models (i.e. with the same weights initially), one under 'EP' AND another one under 'BPTT', with the exact same hyperparameters.|`--benchmark`|
 

 + Network arguments: 
 
   |Arguments|Description|Examples|
   |-------|------|------|
   |`size_tab`|Specifies the topology of the network, backward from the output layer. It is also used alongside `--C_tab` to define the fully connected part of a
                convolutional architecture (see below) |`--size_tab 10 512 784`|
   |`discrete`|Specifies if we are in the prototypical (discrete = True) or energy-based (discrete = False) setting |`--discrete` |
   |`dt`| Time increment in the energy-based setting (denoted \epsilon in the draft)|`--dt 0.1`|
   |`T`|Number of steps in the first phase.|`--T 30`|
   |`Kmax`|Number of steps in the second phase.|`--Kmax 10`|
   |`beta`|Value of the nudging parameter.|`--beta 0.1`|
   |`activation-function`|Selects the activation function used: either 'tanh', 'sigm' (for sigmoid) or 'hardsigm' (for hard-sigmoid)|`--activation-function 'sigm'`|
   |`no-clamp`|Specifies whether we clamp the updates (no-clamp = False) or not when training in the energy-based setting.|`--no-clamp`|
   |`toymodel`|Specifies whether we work on the toymodel (toymodel = True) or not. |`--toymodel`|
   |`C_tab`|Channels tab, going backward from the classifier.|`--C_tab 64 32 1`|
   |`padding`|Specifies whether padding is applied (padding = 1) to keep the image size invariant after a convolution.|`--padding`|
   |`Fconv`|Convolution filter size. |`--Fconv 3`|
   
  + Others:

   |Arguments|Description|Examples|
   |-------|------|------|
   |`action`|Specifies the action to take in main.py (see next bullet). | `--action train`|
   |`device-label`|Selects the cuda device to run the simulation on (default: -1, selecting CPU). | `--device-label 1`|


 main.py can take three different actions:

  + `train`: the network is trained with the arguments provided in the parser. It is trained by default with EP and/or
      with BPTT depending on the arguments provided in the parser. Results are automatically saved in a folder sorted by
      date, GPU ID and trial number, along with a .txt file containing all hyperparameters. 

  + `plotcurves`: we demonstrate the GDU property on the network with the arguments provided in the parser. Results are 
       automatically saved in a folder sorted by trial number, along with a .txt file containing all hyperparameters. 

  + `receipe`: computes the proportion of synapses that satisfy the GDU property in sign, averaged over 20 sample mini-batches. It helps to tune the recurrent hyperparameters T, K and beta. 


IV-  Details about netClasses.py


* There are four network classes:

  i) EPcont: builds fully connected layered architectures in the energy-based setting.

  ii) EPdisc: builds fully connected layered architectures in the prototypical setting. 

  iii) toyEPcont: builds the toy model in the energy based setting. 

  iv) convEP: builds the convolutional architecture in the prototypical setting. 

* Each neural network class contains the following features and methods:

  i) Each class is a subclass of torch.nn.Module.

  ii) stepper: runs the network dynamics between two consecutive steps.

  iii) forward: runs the network dynamics over T steps, with many options depending on the context
      forward is being used. 

  iv) initHidden: initializes hidden units to zero.

  v) computeGradients: compute gradients parameters given the state of the neural network. 

  vi) updateWeights: updates the parameters given the gradient parameters.


V - Details about netFunctions.py

* netFunctions.py contains the following functions:
  
  i) train: trains the model on the training set. 

  ii) evaluate: evaluates the model on the test set. 

  iii) compute_nSdSDT: computes \nabla^{\rm BPTT}_{\rm s}(nS), \Delta^{EP}_{s} (dS) and \sum_{t}\Delta^{EP}_{\theta}(t)(DT). 

  iv) compute_NT: computes \sum(\nabla^{BPTT}_{\theta})(NT). \sum(\nabla^{BPTT}_{\theta})(NT)(t) is the cumulated
                sum of the \nabla^{BPTT}_{\theta} up to t. 

  v) compute_nTdT: computes \nabla^{BPTT}_{\theta} and \delta^{EP}_{\theta} from \sum(\nabla^{BPTT}_{\theta}) and 
                  \sum(\delta^{EP}_{\theta}).

  vi) receipe: computes the proportion of synapses that satisfy the GDU property in sign, averaged over 20 sample mini-batches.

  vii) createPath: creates a path to a directory depending on the date, the gpu device used, the model simulated and on the trial number
                where the results will be saved. 

  viii) createHyperparameterfile: creates a .txt file saved along with the results with all the hyperparameters. 


VI - Details about plotFunctions.py

* plotFunctions.py contains the following functions:

  i) plot_T: plots \nabla^{BPTT}_{\theta} and \Delta^{EP}_{\theta} processes.

  ii) plot_S: plots \nabla^{BPTT}_{s} and \Delta^{EP}_{s} processes.

  iii) compute_nTdT: same function as in netFunctions.py. 

  iv) compute_Hist: computes the Relative Mean Squared error (RMSE) between \Delta^{EP} and \nabla^{BPTT} processes.

  v) plot_results: plots the test and train accuracy as a function of epochs. 


******************************************************************************

VII - Commands to be run in the terminal to reproduce the results of the paper

******************************************************************************

* Everytime a simulation is run, a result folder is created and contains plotFunctions.py. To visualize the results,
  plotFunctions.py has to be run within the result folder. 

* Section 4

  i) Subsection 4.2, Fig. 2 (GDU property in the energy-based setting, toy model):

    python main.py --action 'plotcurves' --toymodel --no-clamp --batch-size 1 --size_tab 10 50 5 --activation-function 'tanh' --dt 0.08 --beta 0.01 --T 5000 --Kmax 80

  ii) Subsection 4.3, Fig. 3 (RMSE analysis on the fully connected layered architectures, in the canonical and prototypical settings):

    - Energy-based setting, 1 hidden layer: 

      python main.py --action 'plotcurves' --no-clamp --batch-size 20 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.001 --T 800 --Kmax 80

    - Energy-based setting, 2 hidden layers: 
      
      python main.py --action 'plotcurves' --no-clamp --batch-size 20 --size_tab 10 512 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.01 --T 5000 --Kmax 150
      
    - Energy-based setting, 3 hidden layers: 

      python main.py --action 'plotcurves' --no-clamp --batch-size 20 --size_tab 10 512 512 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.02 --T 30000 --Kmax 200  

    - Prototypical setting, 1 hidden layer: 

      python main.py --action 'plotcurves' --discrete --batch-size 20 --size_tab 10 512 784 --activation-function 'tanh' --beta 0.01 --T 150 --Kmax 10  

    - Prototypical setting, 2 hidden layers:

      python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 512 784 --activation-function 'tanh' --beta 0.01 --T 1500 --Kmax 40
        
    - Prototypical setting, 3 hidden layers:

      python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 512 512 784 --activation-function 'tanh' --beta 0.015 --T 5000 --Kmax 40

  iii) Subsection 4.4, Fig. 4 (GDU property on the convolutional architecture):

    python main.py --action 'plotcurves' --batch-size 1 --size_tab 10 --C_tab 64 32 1 --activation-function 'hardsigm' --beta 0.02 --T 5000 --Kmax 10

  iv) Table 1 (training simulation results):

    - Energy-based setting, 1 hidden layer (EB-1h):
    
      python main.py --action 'train' --size_tab 10 512 784 --lr_tab 0.05 0.1 --epochs 30 --T 100 --Kmax 12 --beta 0.5 --dt 0.2 --benchmark

    - Energy-based setting, 2 hidden layers (EB-2h):

      python main.py --action 'train' --size_tab 10 512 512 784 --lr_tab 0.01 0.1 0.4 --epochs 50 --T 400 --Kmax 40 --beta 0.5 --dt 0.2 --benchmark

    - Prototypical setting, 1 hidden layer (P-1h):

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 40 --Kmax 15 --beta 0.1 --benchmark

    - Prototypical setting, 2 hidden layers (P-2h):

      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5 --benchmark

    - Prototypical setting, 3 hidden layers (P-3h):

      python main.py --action 'train' --discrete --size_tab 10 512 512 512 784 --lr_tab 0.002 0.01 0.05 0.2 --epochs 100 --T 180 --Kmax 20 --beta 0.5 --benchmark

    - Prototypical setting, convolutional architecture (P-conv): 

      python main.py --action 'train' --activation-function 'hardsigm' --C_tab 64 32 1 --size_tab 10 --lr_tab 0.015 0.035 0.15 --epochs 40 --T 200 --Kmax 10 --beta 0.4 --benchmark

* Appendix C:

  i) Appendix C.2.1, Fig. 8 (GDU property in the energy-based setting, fully connected layer architecture, 1 hidden layer):

    python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.001 --T 800 --Kmax 80


  ii) Appendix C.2.1, Fig. 9 (GDU property in the energy-based setting, fully connected layer architecture, 2 hidden layers):

    python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.01 --T 5000 --Kmax 150 
  

  iii) Appendix C.2.1, Fig. 10 (GDU property in the energy-based setting, fully connected layer architecture, 3 hidden layers):

    python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 512 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.02 --T 30000 --Kmax 200     


  iv) Appendix C.2.2, Fig. 11 (GDU property in the prototypical setting, fully connected layer architecture, 1 hidden layer):

    python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --beta 0.01 --T 150 --Kmax 10


  iv) Appendix C.2.2, Fig. 12 (GDU property in the prototypical setting, fully connected layer architecture, 2 hidden layers):

    python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 512 784 --activation-function 'tanh' --beta 0.01 --T 1500 --Kmax 40


  v) Appendix C.2.2, Fig. 13 (GDU property in the prototypical setting, fully connected layer architecture, 3 hidden layers):

    python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 512 512 784 --activation-function 'tanh' --beta 0.015 --T 5000 --Kmax 40

  vi) Appendix D, Fig. 16 (RMSE analysis on the convolutional architecture):
  
    python main.py --action 'plotcurves' --batch-size 20 --size_tab 10 --C_tab 64 32 1 --activation-function 'hardsigm' --beta 0.02 --T 5000 --Kmax 10
 
