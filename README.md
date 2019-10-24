# Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input
(https://arxiv.org/abs/1905.13633)

The following document provides details about the code provided, alongside the commands to be run to reproduce
the results appearing in the draft. $$x=\sqrt{2}$$


I - Package requirements

* Our code is compatible with Python 2.7 or 3.

* Our virtual environment contains the following packages (after executing a pip freeze command):

    #Basic packages:
    absl-py==0.7.1
    astor==0.7.1
    backports.functools-lru-cache==1.5
    backports.weakref==1.0.post1
    cycler==0.10.0
    enum34==1.1.6
    funcsigs==1.0.2
    futures==3.2.0
    gast==0.2.2
    grpcio==1.19.0
    h5py==2.9.0
    kiwisolver==1.0.1
    Markdown==3.1
    matplotlib==2.2.4
    mock==2.0.0
    numpy==1.16.2
    pbr==5.1.3
    Pillow==6.0.0
    protobuf==3.7.1
    pyparsing==2.3.1
    python-dateutil==2.8.0
    pytz==2018.9
    PyYAML==5.1
    scipy==1.2.1
    six==1.12.0
    subprocess32==3.5.3
    termcolor==1.1.0
    Werkzeug==0.15.2

    #Relevant packages for our project:
    Keras==2.2.4
    Keras-Applications==1.0.6
    Keras-Preprocessing==1.0.5
    torch==1.0.1.post2
    torchvision==0.2.2.post3
    tensorboard==1.12.2
    tensorflow==1.12.0

* To create an environment to run our code:

  i) Install Python 2.7 or 3.
  ii) Install pip.
  iii) Run pip install virtualenv.
  ii) Run mkdir myproject.
  iii) Run cd myproject.
  iv) Run virtualenv myenv.
  v) Create a requirements.txt file containing the package requirements of the previous bullet.
  vi) source myenv/bin/activate.
  vii) Run pip install -r requirements.txt.

II - Files

* The project contains the following files:

  i) main.py: executes the code, with arguments specified in a parser.

  ii) netClasses.py: contains the network classes.

  iii) netFunctions: contains the functions to run on the networks.

  iv) plotFunctions: contains the functions to plot the results. 

III - Details about main.py

* main.py proceeds in the following way:

  i) It first parses arguments typed in the terminal to build a network and get optimization parameters
  (i.e. learning rates, mini-batch size, network topology, etc.)

  ii) It loads the MNIST data set with torchvision.

  iii) It builds the nets using netClasses.py.

  iv) It takes one of the three actions that can be fed into the parser.


* The parser takes the following arguments:

  i) Optimization arguments:  

    --batch-size: training batch size.

    --test-batch-size: test batch size used to compute the test error.

    --epochs: number of epochs.

    --lr_tab: learning rates tab, to be provided from the output layer towards the first layer.
              Example: --lr_tab 0.01 0.04 will apply a learning rate of 0.01 to W_{01} and 0.04 to W_{12}.
 
    --training-method: specify either 'EP' or 'BPTT'.

    --benchmark: trains two exact same models (i.e. with the same weights initially), one under 'EP' AND another one under 'BPTT', with the exact same hyperparameters.


  ii) Network arguments: 

    --size_tab: specify the topology of the network, backward from the output layer.
                Example: --size_tab 10 512 784.
                It is also used alongside --C_tab to define the fully connected part of a
                convolutional architecture -- see below. 

    --discrete: specifies if we are in the prototypical (discrete = True) or energy-based (discrete = False) setting. 

    --dt: time increment in the energy-based setting (denoted \epsilon in the draft). 

    --T: number of steps in the first phase. 

    --Kmax: number of steps in the second phase. 

    --beta: nudging parameter. 
 
    --activation-function: selects the activation function used: either 'tanh', 'sigm' (for sigmoid) or 'hardsigm' (for hard-sigmoid).

    --no-clamp: specifies whether we clamp the updates (no-clamp = False) or not when training in the energy-based setting.

    --toymodel: speficy whether we work on the toymodel (toymodel = True) or not. 
                                                 
    --C_tab: channels tab, going backward from the classifier. 
             Example: --C_tab 64 32 1

    --padding: specifies whether padding is applied (padding = 1) to keep the image size invariant after a convolution.

    --Fconv: specifies the convolution filter size. 

  iii) Others:

    --action: specifies the action to take in main.py (see next bullet).

    --device-label: selects the cuda device to run the simulation on. 


* main.py can take three different actions:

  i) 'train': the network is trained with the arguments provided in the parser. It is trained by default with EP and/or
      with BPTT depending on the arguments provided in the parser. Results are automatically saved in a folder sorted by
      date, GPU ID and trial number, along with a .txt file containing all hyperparameters. 

  ii) 'plotcurves': we demonstrate the GDU property on the network with the arguments provided in the parser. Results are 
       automatically saved in a folder sorted by trial number, along with a .txt file containing all hyperparameters. 

  iii) 'receipe': computes the proportion of synapses that satisfy the GDU property in sign, averaged over 20 sample mini-batches. It helps to tune the recurrent hyperparameters T, K and beta. 


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
 
