import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import math

class node():
    """The nodes for every node in the layers

    The class of nodes in the layers, based on the nodes the forward and backward calculation will be performed 

    Attributes:
        bias(float): the bias of the node 
        
        values(float): the values of the nodes, from the former nodes, the value is wx+b, wx\
            come from the former nodes, b is the bias of the node 
        
        delta(float): the delta used to perform the backward propagation 

        weights(float): the weights to the nodes of next layers in this node 
        
        weights_modified(float): the modification for the weights, \
            used to be recorded during the train process

        output(float): the output of the node (after activation)

        diff(float): the output of the node in the differential activation function, used for back propagation 

        ActivateFunc(function): the activation function of the node 

        ActivateFuncDiff(function): the differential activation function, \
            used for backward propagation 

    """
    def __init__(self,LayNum,ActFunc):
        """initialise the attributes randomly  

        args:
            LayNum(int): the number of nodes in the next layer 
            ActFunc(string): the mode of activation function 
        """
        self.bias = np.random.rand(1)[0].astype(np.float32)
        self.value = 0.1  # the value before the activation function of the node 
        self.delta = 0.1
        if LayNum>0:
            self.weights = np.random.rand(1,LayNum)[0].astype(np.float32)  # the weights connected nodes of next layer
            self.weights_modified  =  0.1* np.random.rand(1,LayNum)[0].astype(np.float32)  # the modification for the weights, used to be recorded during the train process
        #self.output = self.ActivateFunc(self.value,ActFunc)  # the output of the node after activation function 
        #self.diff = self.ActivateFuncDiff(self.value,ActFunc)  # the output of the node in the differential activation function, used for back propagation 

    
        
     
class NetStructure():
    """the 3-layer network,if needed it can be extended to more layers
    in this case the net will be set as 3 input nodes, 8 hidden layers, 3 output layers
    attributes:
        layers(class nodes): the nodes form a layer, the numbers of layers can be set according to the tasks
        forward(function): the forward calculation of the network
        backward(function): the backward propagation of the network
        train(function): the training function in the SGD form
        train_batch(function): the training functuon in the batch form
    """
    def __init__(self):
        """the initial part of the networks
        the structure of the 3-layer network will be set here 

        """
        # set the input layer 
        InputNode1 = node(LayNum=5,ActFunc="input")
        InputNode2 = node(LayNum=5,ActFunc="input")
        InputNode3 = node(LayNum=5,ActFunc="input")

        self.Inputlayer = [InputNode1,InputNode2,InputNode3] 
        # set the hidden layer 
        HiddenNode1 = node(LayNum=3,ActFunc="tanh")
        HiddenNode2 = node(LayNum=3,ActFunc="tanh")
        HiddenNode3 = node(LayNum=3,ActFunc="tanh")
        HiddenNode4 = node(LayNum=3,ActFunc="tanh")
        HiddenNode5 = node(LayNum=3,ActFunc="tanh")
        self.HiddenLayer = [HiddenNode1,HiddenNode2,HiddenNode3,HiddenNode4,HiddenNode5 ]
        # set the output layer
        OutputNode1 = node(LayNum=0,ActFunc="sigmoid")
        OutputNode2 = node(LayNum=0,ActFunc="sigmoid")
        OutputNode3 = node(LayNum=0,ActFunc="sigmoid")
        self.OutputLayer = [OutputNode1,OutputNode2,OutputNode3]
    
    def ActivateFunc(self,x,ActFunc):
        """the activation function 
        
        args:
            x(float): the input of the function 
            ActFunc(string): the mode of activation function 
        return:
            the output of the function 
        """
        if (ActFunc=="input"):  # without the activation function, for example the input layer
            return x
        if (ActFunc=="sigmoid"):
            return 1/(1+np.exp(-x))
        if (ActFunc=="tanh"):
            tanh = (np.exp(2*x)-1)/(np.exp(2*x)+1)
            return tanh
    
    def ActivateFuncDiff(self,x,ActFunc):
        """the differential activation function, used for backward propagation 

        args:
            x(float): the input of the function 
            ActFunc(string): the mode of activation function 
        return:
            the output of the function differential
        """
        if (ActFunc=="input"):  # without the activation function
            return 1
        if (ActFunc=="sigmoid"):
            sig = 1/(1+np.exp(-x))
            return sig*(1-sig)
        if (ActFunc=="tanh"):
           
            tanh = (np.exp(2*x)-1)/(np.exp(2*x)+1)
            return 1 - tanh**2

    def forward(self,input_x):
        """forward calculation of the network
        args: 
            input_x: the input data
        return:
            the output after the activate function of the network
        """
        # assignment for the input nodes
        for i in range(len(input_x)):
            self.Inputlayer[i].value = input_x[i]
            
        # calculation for the hidden layers
        for j in range(len(self.HiddenLayer)):
            value = 0
            for k in range(len(self.Inputlayer)):
                value += self.ActivateFunc(self.Inputlayer[k].value,"input") * self.Inputlayer[k].weights[j]  # calculate the value of the hidden layers' nodes
            self.HiddenLayer[j].value = value
            self.HiddenLayer[j].value += self.HiddenLayer[j].bias  # don't forget the bias
            
        # calculation for the output layers
        for m in range(len(self.OutputLayer)):
            value = 0
            for l in range(len(self.HiddenLayer)):
                value += self.ActivateFunc(self.HiddenLayer[l].value,"tanh") * self.HiddenLayer[l].weights[m]
            self.OutputLayer[m].value = value
            self.OutputLayer[m].value += self.OutputLayer[m].bias
            
        output_y = [self.ActivateFunc(self.OutputLayer[n].value,"sigmoid") for n in range(len(self.OutputLayer))]
        #print(output_y)
        return output_y  
    
    def backward(self,input_x,samples_y,update =1,eta = 0.1):
        """backward calculation of the network 
        args: 
            eta(float, default = 0.1): learning rate  
            input_x(list): the input data
            samples_y(list): the label of the samples for training
            update(string, default = "no"): whether or not update the weights 
        """
        output_y = self.forward(input_x)
        # calculation for the modifications of the weights between the hidden layer and the output layer
        for i in range(len(self.OutputLayer)):
            self.OutputLayer[i].delta = self.ActivateFuncDiff(self.OutputLayer[i],"input") * (samples_y[i] - output_y[i])  # calculate the delta for each output node
        
        OutputLayer_delta = np.array([self.OutputLayer[i1].delta for i1 in range(len(self.OutputLayer))])  # record the delta in the output layer for further use
        
        for j in range(len(self.HiddenLayer)):
            self.HiddenLayer[j].weights_modified = np.array([eta*self.ActivateFunc(self.HiddenLayer[j].value,"tanh")*\
                self.OutputLayer[k1].delta for k1 in range(len(self.OutputLayer))])  # calculate and record the modification of the weights
            if update != 0:  # whether or not update the weights
                self.HiddenLayer[j].weights += self.HiddenLayer[j].weights_modified

        # calculate the modifications of the weights between the input layer and the hidden layer 
        for k in range(len(self.HiddenLayer)):
            self.HiddenLayer[k].delta = self.ActivateFuncDiff(self.HiddenLayer[k].value,"input") * \
                np.dot(self.HiddenLayer[k].weights,OutputLayer_delta)
        
        for l in range(len(self.Inputlayer)):
            self.Inputlayer[l].weights_modified = np.array([eta*self.ActivateFunc(self.Inputlayer[l].value,"input")*\
                self.HiddenLayer[m].delta for m in range(len(self.HiddenLayer))])
            if update != 0:
                self.Inputlayer[l].weights += self.Inputlayer[l].weights_modified
            
    def train(self,input_set,samples,epoch=30,rate=0.1):
        """training function in the form of SGD
        args: 
            input_set(list): the input data
            samples(list): the label of the samples for training
            epoch: training times 
            rate: learing rate 
        """  
        losses = []  # recording the losses

        for _ in range(epoch):
            
            i = np.random.randint(0,len(input_set))
            self.backward(input_set[i],samples[i],update=1, eta=rate)
            # calculate the loss
            loss = 0
            for j in range(len(input_set)):
                loss += np.linalg.norm(np.array(self.forward(input_set[j]))-np.array(samples[j]))**2
            losses.append(loss/len(input_set))
            
        plt.plot([i for i in range(1,len(losses)+1)],losses)
        plt.xlabel("training times")
        plt.ylabel("training loss")
        plt.show()
    
    def train_batch(self,input_set,samples,batch_size=15,epoch=5,rate=0.1):
        """training function in the form of SGD
        args: 
            input_set(list): the input data
            samples(list): the label of the samples for training
            batch_size(int, default=15): the batch size
            epoch: times
            rate: learing rate   
        """  
        losses = []  # recording the losses
        for _ in range(epoch):

            batch = np.random.randint(0,len(input_set),batch_size)
            for i in range(batch.shape[0]):
                if i==batch_size-1:
                    self.backward(input_set[ batch[i] ],samples[ batch[i] ],update=1, eta=rate)  # the weights will be updated in the last train of the batch 
                    # calculate the loss
                    loss = 0
                    for j in range(len(input_set)):
                        loss += np.linalg.norm(np.array(self.forward(input_set[j]))-np.array(samples[j]))**2
                    losses.append(loss/len(input_set))
                else:
                    self.backward(input_set[ batch[i] ],samples[ batch[i] ],update=0, eta=0.1)
                
        plt.plot([i for i in range(1,len(losses)+1)],losses)
        plt.xlabel("training times")
        plt.ylabel("training loss")
        plt.show()



        

