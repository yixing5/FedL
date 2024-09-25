'''
Author: Sasha
Date: 2023-03-22 10:47:06
LastEditors: Sasha
LastEditTime: 2023-04-04 09:16:59
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend/mmoe_interactive_layer.py
'''

from FedL.federatedml.dnn.backend.tensorflow.initializer import XavierUniform
import numpy as np
from FedL.federatedml.util.optimizer import Adam
from  FedL.federatedml.dnn.model.base_nn_model import BaseNNModel
from FedL.federatedml.util.yk_operator import dot

import pickle

class MMoEInteractiveModel(BaseNNModel):
    def __init__(self,shape_host,shape_guest,decay,lr=0.01) -> None:
        super().__init__()
        self.shape_host = shape_host 
        self.shape_guest = shape_guest
        self.lr = lr
        # self.optimizer_host = Adam(lr=self.lr )
        # self.optimizer_guest = Adam(lr=self.lr )
        self.decay = decay 
        self.inputs_host = {}
        self.inputs_guest = {}
        self._init_weight()

    def _init_weight(self):
        weight = XavierUniform()((self.shape_host[0]+self.shape_guest[0],self.shape_host[1]))
        self.weight_host,self.weight_guest = np.split(weight,[self.shape_host[0]],axis = 0)
        
    def forward_guest(self,inputs,name='train'):
        self.inputs_guest[name] = inputs
        return inputs.dot(self.weight_guest)
    
    def forward_host(self,inputs,name='train'):
        self.inputs_host[name] = inputs
        return dot(inputs,self.weight_host) #inputs.dot(self.weight_host)
    
    def backward_host(self,grads_act,name='train'):
        return dot(self.inputs_host[name].T,grads_act) # self.inputs_host[name].T.dot(grads_act) / len(grads_act)
    
    def backward_guest(self,grads_act,name='train'):
        return self.inputs_guest[name].T.dot(grads_act) / len(grads_act)
    
    def get_encryped_grad_bottom_host(self,grads_act,encrypted_acc_noise):
        return dot(grads_act,(self.weight_host + encrypted_acc_noise).T) #grads_act.dot((self.weight_host + encrypted_acc_noise).T) #慢
    
    def get_grad_bottom_guest(self,grads_act):
        return grads_act.dot(self.weight_guest.T)
    
    def train(self,grad_weight_host,grad_weight_guest):
        lr = self.decay.compute_step()
        self.weight_host -= lr * grad_weight_host
        self.weight_guest -=  lr * grad_weight_guest
        # self.weight_host = self.weight_host -  self.optimizer1.compute_step(grad_weight_host/size)
        # self.weight_guest = self.weight_guest - self.optimizer2.compute_step(grad_weight_guest/size)
    
    def save(self,filename):
        with open(self.get_model_path(filename), 'wb') as fw:
            pickle.dump([self.weight_host,self.weight_guest], fw)
    
    def load_model(self,filename):
        with open(self.get_model_path(filename), 'rb') as fr:
            self.weight_host,self.weight_guest =  pickle.load(fr, encoding='bytes')
