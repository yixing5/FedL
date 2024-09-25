'''
Author: Sasha
Date: 2023-03-10 09:36:45
LastEditors: Sasha
LastEditTime: 2023-03-24 17:51:22
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/model/nn_top_model.py
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from FedL.federatedml.dnn.backend.tensorflow.layers import DNN
from FedL.util.log import log
from FedL.federatedml.dnn.model.base_nn_model import BaseNNModel

class TopModel(BaseNNModel):
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
        self.valid_loss = keras.losses.MeanSquaredError()
        self.loss_func = keras.losses.MeanSquaredError()

    def train(self,inputs,label):
        inputs = tf.constant(inputs)
        with tf.GradientTape() as tape_top:
            tape_top.watch(inputs)
            predictions = self.model(inputs)
            loss = self.loss_func(tf.reshape(label,[-1]), tf.reshape(predictions,[-1]))
        log.info("##########################################################loss: %s##########################################################" % loss)
        grads_act,grads = tape_top.gradient(loss, [inputs,self.model.trainable_variables])
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return grads_act.numpy()
    
    def valid(self,inputs,label):
        predictions = self.model(inputs)
        return  predictions,self.valid_loss(tf.reshape(label,[-1]), tf.reshape(predictions,[-1]))
    
