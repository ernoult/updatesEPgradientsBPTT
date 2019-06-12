from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from main import rho, rhop, rhop2

#*****************************EP, energy based *********************************#
class EPcont(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab, T, Kmax, beta, dt = 1, no_clamp = False):
        super(EPcont, self).__init__()
        
        self.T = T
        self.Kmax = Kmax
        self.dt = dt
        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False    
        self.device = device
        self.no_clamp = no_clamp
        self.beta = beta
                    
        w = nn.ModuleList([])           
        for i in range(self.ns - 1):
            w.append(nn.Linear(size_tab[i + 1], size_tab[i], bias = True))
            w.append(None)

        w.append(nn.Linear(size_tab[-1], size_tab[-2]))                                         
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + self.w[0](rho(s[1])))     
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + torch.mul(rhop(s[i]), self.w[2*i](rho(s[i + 1])) + torch.mm(rho(s[i - 1]), self.w[2*(i-1)].weight)))

        dsdt.append(-s[-1] + torch.mul(rhop(s[-1]), self.w[-1](rho(data)) + torch.mm(rho(s[-2]), self.w[-3].weight)))

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]
            
        if not self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i].clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])
                                     
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd',  beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax

        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax

        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()                        
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
                
            for t in range(Kmax):
                for i in range(self.ns):
                ###############################nS COMPUTATION#####################################
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                ####################################################################################
          

            return s, nS     
            
        elif (method == 'dSDT'):               
                DT = []
                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)                        
                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        

                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
             
        for i in range(self.ns - 1):
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[i]), 0, 1), rho(s[i + 1])) - torch.mm(torch.transpose(rho(seq[i]), 0, 1), rho(seq[i + 1])))) 
            gradw.append(None)            
            gradw_bias.append((1/(beta*batch_size))*(rho(s[i]) - rho(seq[i])).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(rho(s[-1]) - rho(seq[-1]), 0, 1), rho(data)))
        gradw_bias.append((1/(beta*batch_size))*(rho(s[-1]) - rho(seq[-1])).sum(0))
               
        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i] 

#*****************************EP, prototypical *********************************#

class EPdisc(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab, T, Kmax, beta, dt = 1):
        super(EPdisc, self).__init__()
        self.T = T
        self.Kmax = Kmax        
        self.dt = dt
        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.beta = beta

        w = nn.ModuleList([])
                           
        for i in range(self.ns - 1):
            w.append(nn.Linear(size_tab[i + 1], size_tab[i], bias = True))
            w.append(None)
            
        w.append(nn.Linear(size_tab[-1], size_tab[-2]))                             
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + rho(self.w[0](s[1])))
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + rho(self.w[2*i](s[i + 1]) + torch.mm(s[i - 1], self.w[2*(i-1)].weight)))

        dsdt.append(-s[-1] + rho(self.w[-1](data) + torch.mm(s[-2], self.w[-3].weight)))

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]	
                                     
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd', beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
               
            for t in range(Kmax):
                for i in range(self.ns):
                    ###############################nS COMPUTATION#####################################
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                    ####################################################################################

                                      
               
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)

                
        for i in range(self.ns - 1):
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(s[i], 0, 1), s[i + 1]) - torch.mm(torch.transpose(seq[i], 0, 1), seq[i + 1]))) 
            gradw.append(None)            
            gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[-1] - seq[-1], 0, 1), data))
        gradw_bias.append((1/(beta*batch_size))*(s[-1] - seq[-1]).sum(0))
                                                                                                                                                                
        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i]      


#*****************************toy model, EP, energy based *********************************#

class toyEPcont(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab, T, Kmax, beta, dt = 1, no_clamp = False):
        super(toyEPcont, self).__init__()
        self.T = T
        self.Kmax = Kmax
        self.dt = dt

        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.beta = beta
        self.no_clamp = no_clamp
        w = nn.ModuleList([])

        #fully connected architecture
        w.append(None)
        w.append(nn.Linear(size_tab[1], size_tab[0]))
        w.append(nn.Linear(size_tab[2], size_tab[0]))
        w.append(None)                               
        w.append(nn.Linear(size_tab[2], size_tab[1]))
        w.append(None)            
    
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + torch.mul(rhop(s[0]), self.w[1](rho(s[1])) + self.w[2](rho(data))))

        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        dsdt.append(-s[1] + torch.mul(rhop(s[1]),  self.w[4](rho(data)) + torch.mm(rho(s[0]), self.w[1].weight)))	
        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]
            
        if not self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i].clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])

                                     
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd',  beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):

            with torch.no_grad():                                                                 
                for t in range(T - Kmax):
                    s = self.stepper(data, s)
                        
            for t in range(Kmax):            
                if t == Kmax - 1 - K:
                    for i in range(len(s)):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            with torch.no_grad():                                                                 
                for t in range(T - Kmax):
                    s = self.stepper(data, s)
                

            for i in range(len(s)): 
                s[i].requires_grad = True                                       
                s_tab[i].append(s[i])                    
                s_tab[i][0].retain_grad()
            
            for t in range(Kmax):
                s = self.stepper(data, s) 
                for i in range(len(s)):                                   
                    s_tab[i].append(s[i])                    
                    s_tab[i][t + 1].retain_grad()               
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
            
            for t in range(Kmax):
                ###############################nS COMPUTATION#####################################
                for i in range(self.ns):
                     if ((i > 0) & (t == 0)):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                     else:      
                        nS[i][t, :, :] = (s_tab[i][-1 - t].grad).sum(0).unsqueeze(0)                
                ####################################################################################
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if self.w[i] is not None:    
                            if gradw[i] is not None:
                                DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
                                                                                                                                                      
        gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[0]), 0, 1), rho(s[0])) - torch.mm(torch.transpose(rho(seq[0]), 0, 1), rho(seq[0]))))
        gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[0]), 0, 1), rho(s[1])) - torch.mm(torch.transpose(rho(seq[0]), 0, 1), rho(seq[1]))))
        gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[0]), 0, 1), rho(data)) - torch.mm(torch.transpose(rho(seq[0]), 0, 1), rho(data))))   
        gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[1]), 0, 1), rho(s[1])) - torch.mm(torch.transpose(rho(seq[1]), 0, 1), rho(seq[1]))))
        gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[1]), 0, 1), rho(data)) - torch.mm(torch.transpose(rho(seq[1]), 0, 1), rho(data))))

        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i] 




#*****************************Convolutional EP, prototypical *********************************#                
class convEP(nn.Module):
    def __init__(self, input_size, device_label, size_classifier_tab, C_tab,  lr_tab, 
                T, Kmax, beta, F = 5, Fpool = 2, padding = 0):
        super(convEP, self).__init__()
        
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        
        self.C_tab = C_tab
        self.lr_tab = lr_tab
        self.T = T
        self.Kmax = Kmax
        
        self.beta = beta
        self.F = F
        self.Fpool = Fpool
        
        self.n_cp = len(C_tab) - 1
        self.n_classifier = len(size_classifier_tab)
        if padding:
            P = int((F - 1)/2)
        else:
            P = 0 

        self.P = P            
        
        conv = nn.ModuleList([])
        fc = nn.ModuleList([])      
        
        size_convpool_tab = [input_size] 
        size_conv_tab = [input_size]

        #Define conv operations
        for i in range(self.n_cp):
            conv.append(nn.Conv2d(C_tab[i + 1], C_tab[i], F, padding = P))

            size_conv_tab.append(size_convpool_tab[i] - F + 1 + 2*P)

            size_convpool_tab.append(int(np.floor((size_convpool_tab[i] - F + 1 + 2*P -Fpool)/2 + 1)))
        
        
            
        self.conv = conv
        
        #Define pool operations          	
        self.pool = nn.MaxPool2d(Fpool, stride = Fpool, return_indices = True)	        
        self.unpool = nn.MaxUnpool2d(Fpool, stride = Fpool)    
        
        size_convpool_tab = list(reversed(size_convpool_tab))
        self.size_convpool_tab = size_convpool_tab

        size_conv_tab = list(reversed(size_conv_tab))
        self.size_conv_tab = size_conv_tab

        self.nconv = len(size_convpool_tab) - 1
        size_tab = list(size_classifier_tab)
        size_tab.append(C_tab[0]*size_convpool_tab[0]**2)
        self.size_classifier_tab = size_tab
        del size_tab
        self.nc = len(self.size_classifier_tab) - 1
        
        #Define fully connected operations
        for i in range(self.n_classifier):
            fc.append(nn.Linear(self.size_classifier_tab[i + 1], self.size_classifier_tab[i]))    
        
        self.fc = fc
        
        self = self.to(device)
        

    def stepper(self, data, s, inds, target = None, beta = 0, return_derivatives = False, inplace = False): 
        
        dsdt = []
        
        #CLASSIFIER PART
                              
        #last classifier layer
        dsdt.append(-s[0] + rho(self.fc[0](s[1].view(s[1].size(0), -1))))
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target - s[0])
             
        
        #middle classifier layer
        for i in range(1, len(self.size_classifier_tab) - 1):
            dsdt.append(-s[i] + rho(self.fc[i](s[i + 1].view(s[i + 1].size(0), -1)) + torch.mm(s[i - 1], self.fc[i - 1].weight)))

        
        
        #CONVOLUTIONAL PART
               
        #last conv layer
        s_pool, ind = self.pool(self.conv[0](s[self.nc + 1]))
        inds[self.nc] = ind
        dsdt.append(-s[self.nc] + rho(s_pool + torch.mm(s[self.nc - 1], self.fc[- 1].weight).view(s[self.nc].size())))


        del s_pool, ind
        
        
        #middle layers
        for i in range(1, self.nconv - 1):	
            s_pool, ind = self.pool(self.conv[i](s[self.nc + i + 1]))
            inds[self.nc + i] = ind
            
            if inds[self.nc + i - 1] is not None:      

                output_size = [s[self.nc + i - 1].size(0), s[self.nc + i - 1].size(1), self.size_conv_tab[i - 1], self.size_conv_tab[i - 1]]                                            
                s_unpool = F.conv_transpose2d(self.unpool(s[self.nc + i - 1], inds[self.nc + i - 1], output_size = output_size), 
                                                weight = self.conv[i - 1].weight, padding = self.P)                                                
                                                                                                                                                               
            dsdt.append(-s[self.nc + i] + rho(s_pool + s_unpool))
            del s_pool, s_unpool, ind, output_size
            
        #first conv layer
        s_pool, ind = self.pool(self.conv[-1](data))
        inds[-1] = ind
        if inds[-2] is not None:
            output_size = [s[-2].size(0), s[-2].size(1), self.size_conv_tab[-3], self.size_conv_tab[-3]]
            s_unpool = F.conv_transpose2d(self.unpool(s[-2], inds[-2], output_size = output_size), weight = self.conv[-2].weight, padding = self.P)
        dsdt.append(-s[-1] + rho(s_pool + s_unpool))
        del s_pool, s_unpool, ind, output_size
        
        if not inplace:                  
            for i in range(len(s)):
                s[i] = s[i] + dsdt[i]
        else:
            for i in range(len(s)):
                s[i] += dsdt[i]            

        
        if return_derivatives:
           return s, inds, dsdt
        else:
            del dsdt  
            return s, inds
                      
    
    def forward(self, data, s, inds, seq = None, indseq = None, method = 'nograd', beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax

        if (method == 'withgrad'):                
            with torch.no_grad():                                                                 
                for t in range(T - Kmax):
                    #print(t)
                    s, inds = self.stepper(data, s, inds)
                        
            for t in range(Kmax):            
                if t == Kmax - 1 - K:
                    for i in range(len(s)):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s, inds = self.stepper(data, s, inds)
            return s, inds
            
        
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                  
                    s, inds = self.stepper(data, s, inds)
                    torch.cuda.empty_cache()
            else:
                for t in range(Kmax):                      
                    s, inds = self.stepper(data, s, inds, target, beta)
            return s, inds                 
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(len(s)):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')           
                    
            with torch.no_grad():                                                                 
                for t in range(T - Kmax):

                    s, inds = self.stepper(data, s, inds)
                
            for i in range(len(s)): 
                s[i].requires_grad = True                                       
                s_tab[i].append(s[i])                    
                s_tab[i][0].retain_grad()
            
            for t in range(Kmax):
                s, inds = self.stepper(data, s, inds) 
                for i in range(len(s)):                                   
                    s_tab[i].append(s[i])                    
                    s_tab[i][t + 1].retain_grad()  
                      
             
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(len(s)):
                if len(s[i].size()) < 3:
                    nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
                else:
                    nS.append(torch.zeros(Kmax, 1, s[i].size(1), s[i].size(2), s[i].size(3), device = self.device))
                    

            for t in range(Kmax):
                for i in range(len(s)):
                    ###############################nS COMPUTATION#####################################
                    if (t < i):
                        nS[i][t, :] = torch.zeros_like(nS[i][t, :])
                    else:    
                        nS[i][t, :] = (s_tab[i][-1 - t].grad).sum(0).unsqueeze(0)
                    ####################################################################################
                                                    
            return s, inds, nS     
            
        elif (method == 'dSDT'):
                DT_fc = []
                DT_conv = []

                for i in range(len(self.fc)):
                    DT_fc.append(torch.zeros(Kmax, self.fc[i].weight.size(0), self.fc[i].weight.size(1)))    
                     
                for i in range(len(self.conv)):
                    DT_conv.append(torch.zeros(Kmax, self.conv[i].weight.size(0), self.conv[i].weight.size(1), 
                                            self.conv[i].weight.size(2), self.conv[i].weight.size(3)))                
                
                
                dS = []
                for i in range(len(s)):
                    if len(s[i].size()) < 3:
                        dS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
                    else:
                        dS.append(torch.zeros(Kmax, 1, s[i].size(1), s[i].size(2), s[i].size(3), device = self.device))              
                
                    
                for t in range(Kmax):
                    #print(t)
                    s, inds, dsdt = self.stepper(data, s, inds, target, beta, return_derivatives = True)
                    ############################dS COMPUTATION################################
                    for i in range(len(s)):
                        if (t < i):
                            dS[i][t, :] = torch.zeros_like(dS[i][t, :])
                        else:
                            dS[i][t, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                      
                    ##############################################################################


                    ##########################DT COMPUTATION################################
                    gradconv, _, gradfc, _ = self.computeGradients(beta, data, s, inds, seq, indseq)
                    for i in range(len(gradconv)):
                            DT_conv[i][t, :] = - gradconv[i]
                    for i in range(len(gradfc)):
                            DT_fc[i][t, :] = - gradfc[i]
                    ################################################################################
                                                       
                                                                             
        return s, inds, dS, DT_conv, DT_fc
        
               
    def initHidden(self, batch_size):
            
        s = []
        inds = []
        for i in range(self.nc):
            s.append(torch.zeros(batch_size, self.size_classifier_tab[i], requires_grad = True))
            inds.append(None)
        for i in range(self.nconv):
            s.append(torch.zeros(batch_size, self.C_tab[i], self.size_convpool_tab[i], self.size_convpool_tab[i], requires_grad = True))    
            inds.append(None)
             
        return s, inds    
        

    def computeGradients(self, beta, data, s, inds, seq, indseq):
        gradfc= []
        gradfc_bias = []
        gradconv= []
        gradconv_bias = []
        batch_size = s[0].size(0)
               
        #CLASSIFIER       
        for i in range(self.nc - 1):
            gradfc.append((1/(beta*batch_size))*(torch.mm(torch.transpose(s[i], 0, 1), s[i + 1]) - torch.mm(torch.transpose(seq[i], 0, 1), seq[i + 1])))          
            gradfc_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))                                                                           

        gradfc.append((1/(beta*batch_size))*(torch.mm(torch.transpose(s[self.nc - 1], 0, 1), s[self.nc].view(s[self.nc].size(0), -1)) - torch.mm(torch.transpose(seq[self.nc - 1], 0, 1), seq[self.nc].view(seq[self.nc].size(0), -1))))          
        gradfc_bias.append((1/(beta*batch_size))*(s[self.nc - 1] - seq[self.nc - 1]).sum(0))
        
                                                                             
        #CONVOLUTIONAL
        for i in range(self.nconv - 1):

            output_size = [s[self.nc + i].size(0), s[self.nc + i].size(1), self.size_conv_tab[i], self.size_conv_tab[i]]                                            

            gradconv.append((1/(beta*batch_size))*(F.conv2d(s[self.nc + i + 1].permute(1, 0, 2, 3), self.unpool(s[self.nc + i], inds[self.nc + i], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)
                            - F.conv2d(seq[self.nc + i + 1].permute(1, 0, 2, 3), self.unpool(seq[self.nc + i], indseq[self.nc + i], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)).permute(1, 0, 2, 3))
            gradconv_bias.append((1/(beta*batch_size))*(self.unpool(s[self.nc + i], inds[self.nc + i], output_size = output_size) - self.unpool(seq[self.nc + i], indseq[self.nc + i], output_size = output_size)).permute(1, 0, 2, 3).contiguous().view(s[self.nc + i].size(1), -1).sum(1))
            
          
        output_size = [s[-1].size(0), s[-1].size(1), self.size_conv_tab[-2], self.size_conv_tab[-2]]

        gradconv.append((1/(beta*batch_size))*(F.conv2d(data.permute(1, 0, 2, 3), self.unpool(s[-1], inds[-1], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)
                        - F.conv2d(data.permute(1, 0, 2, 3), self.unpool(seq[-1], indseq[-1], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)).permute(1, 0, 2, 3))
        gradconv_bias.append((1/(beta*batch_size))*(self.unpool(s[-1], inds[-1], output_size = output_size) - self.unpool(seq[-1], indseq[-1], output_size = output_size)).permute(1, 0, 2, 3).contiguous().view(s[-1].size(1), -1).sum(1))               
        
        
        
        return  gradconv, gradconv_bias, gradfc, gradfc_bias

              
    def updateWeights(self, beta, data, s, inds, seq, indseq):
        gradconv, gradconv_bias, gradfc, gradfc_bias = self.computeGradients(beta, data, s, inds, seq, indseq)
        lr_tab = self.lr_tab

       
        for i in range(len(self.fc)):            
            self.fc[i].weight += lr_tab[i]*gradfc[i]
            self.fc[i].bias += lr_tab[i]*gradfc_bias[i]
        for i in range(len(self.conv)):
            self.conv[i].weight += lr_tab[i + len(self.fc)]*gradconv[i]
            self.conv[i].bias += lr_tab[i + len(self.fc)]*gradconv_bias[i]            


