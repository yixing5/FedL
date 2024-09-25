'''
Author: Sasha
Date: 2023-03-22 14:44:06
LastEditors: Sasha
LastEditTime: 2023-04-11 15:31:26
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v4/mmoe_bottom_model.py
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from  FedL.federatedml.dnn.model.base_nn_model import BaseNNModel
from FedL.federatedml.dnn.backend.tensorflow.layers import Embeddings



class MMoEBottomModel(BaseNNModel):
    def __init__(self,feature_columns,embed_dims,lr=0.01) -> None:
        super().__init__()
        self.feature_columns = feature_columns
        self.embed_dims = embed_dims
        self.embeddings = {}
        self.lr = lr
        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self._bulid_embedding()

    def _bulid_embedding(self):
        self.model =  Embeddings(self.feature_columns,self.embed_dims)

    def train(self,inputs,grads):
        with tf.GradientTape() as tape_bottom:
            output_model = self.model(inputs)
            loss_bottom = tf.math.multiply(output_model,grads)

        grads = tape_bottom.gradient(loss_bottom, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        

        
        