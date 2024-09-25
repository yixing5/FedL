'''
Author: Sasha
Date: 2023-03-22 14:44:06
LastEditors: Sasha
LastEditTime: 2023-03-24 17:50:36
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/model/nn_bottom_model.py
'''
import tensorflow as tf
from tensorflow.keras import optimizers
from FedL.federatedml.dnn.backend.tensorflow.layers import DNN
from  FedL.federatedml.dnn.model.base_nn_model import BaseNNModel

class BottomModel(BaseNNModel):
    def __init__(self,layers_dims,input_dim,lr=0.01) -> None:
        super().__init__()
        self.layers_dims = layers_dims
        self.lr = lr
        self.input_dim = input_dim
        self._build_model()
        
    def _build_model(self):
        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.model = DNN(self.layers_dims)
        self.model.build(input_shape =(None,self.input_dim))

    def train(self,inputs,grads):
        with tf.GradientTape() as tape_bottom:
            output_model = self.model(inputs)
            # loss_bottom = tf.math.multiply(output_model,tf.constant(np.mean(grads,axis=0),dtype='float32'))
            loss_bottom = tf.math.multiply(output_model,grads)

        grads = tape_bottom.gradient(loss_bottom, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        

        
        