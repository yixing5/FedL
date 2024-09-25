'''
Author: Sasha
Date: 2023-03-10 11:18:45
LastEditors: Sasha
LastEditTime: 2023-03-17 16:24:50
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/model/layer.py
'''
#!usr/bin/env python
#-*- coding:utf-8 _*-
from abc import abstractmethod
import numpy as np

class Layer:
    def __init__(self) -> None:
        self.weights = None
        self.grad = None
        self.inputs = None
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward(self):
        pass
    
    def _init_weight(self):
        pass

###################################################activation################################################################


class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.sigmoid(inputs)
        
    def backward(self, grad):
        return  self.grad_func(self.inputs) * grad 
    
    @abstractmethod
    def activate_func(self, x):
        pass
    
    @abstractmethod
    def grad_func(self, x):
        pass
    
    
class Sigmoid(Activation):
    
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def activate_func(self,x):
        return self.sigmoid(x)
    
    def grad_func(self,inputs):
        return self.sigmoid(inputs) * (1-self.sigmoid(inputs))
    


class Relu(Activation):
    
    def relu(self,x):
        return np.maximum(x, 0.0)
    
    def activate_func(self,x):
        return self.relu(x)
    
    def grad_func(self,inputs):
        return inputs > 0

class Softmax(Activation):
    
    def softmax(self,x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    
    def activate_func(self,x):
        return self.softmax(x)
    
    def grad_func(self,inputs):
        return inputs > 0


class Tanh(Activation):
    
    def tanh(self,x):
        return np.tanh(x)

    def activate_func(self, x):
        return self.tanh(x)

    def grad_func(self, x):
        return 1.0 - self.tanh(x) ** 2
    
########################################################################################################################

class Dense(Layer):
    def __init__(self,shape, activation='linear',lambd=None,use_bias=True) -> None:
        self.shape = shape
        self.activation = activation
        self.use_bias = use_bias
        self.lambd = lambd
        self.is_init=False
        
    def init_weight(self):
        self.weights =np.float32 (np.random.normal(size=(self.shape[1], self.shape[0])))
        if self.use_bias:
            self.bias =np.float32 (np.zeros((self.shape[1], 1)))
        
    def forward(self,input):
        if not self.is_init:
            self.parameters = self._init_weight()
            self.is_init = True
        self.m = input.shape[1]
        self.pre_A = input
        Z = np.dot(self.weights, input)
        if self.use_bias:
            Z = Z + self.bias
        self.A = Activation.run(Z,self.activation)
        return self.A

    def backward(self,dZ,learning_rate=0.01):
        self.dWeight = np.float32((1 / self.m) * (np.dot(dZ, self.pre_A.T)) + (self.lambd*self.weights) if  self.lambd else (1 / self.m) * (np.dot(dZ, self.pre_A.T)))
        if self.use_bias :
            self.dbias = np.float32((1 / self.m) * (np.sum(dZ, axis=1, keepdims=True)))
        self.dZ = (np.dot(self.weights.T, dZ)) * np.where( self.pre_A > 0, 1, 0) if self.activation == 'relu'  else (np.dot(self.weights.T, dZ))
        self.update_parameters(learning_rate)
        return self.dZ
            
    def update_parameters(self,learning_rate):
        self.weights = self.weights.astype(np.float32) - learning_rate * self.dWeight
        if self.use_bias :
            self.bias = self.bias.astype(np.float32) - learning_rate * self.dbias
            
class DNN():
    def __init__(self,layers_dims) -> None:
        self.layers_dims = layers_dims
        self.L = len(self.layers_dims)
        self.lambd = 0.001
        self.model_structure = self._build_model()

    
    def _build_model(self):
        model_structure = {}
        for i in range(1, self.L):
            model_structure['layer_%s' % i] = Dense(shape=[self.layers_dims[i-1],self.layers_dims[i]], activation='relu') 
        return model_structure
    
    def call(self,input):
        for i in range(1, self.L):
            input = self.model_structure['layer_%s' % i].forward(input)
    
    def backward(self,dZ):
        for i in range(self.L-1 ,0,-1):
            dZ = self.model_structure['layer_%s' % i].backward(dZ)
    
def test():
    X = np.array([[1,1,1,1,0],[1,1,2,1,1]]).T
    Y = np.array([[1,1]])
    layers_dims = [5,10,5,2]
    dnn = DNN(layers_dims)
    dnn.call(X)
    dZ = np.array([[-0.99888053,  0.00111947],[-0.91888053,  0.00411947]])
    dnn.backward(dZ)
    
            
if __name__ == "__main__":
    test()




