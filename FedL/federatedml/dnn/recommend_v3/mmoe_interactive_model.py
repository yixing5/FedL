'''
Author: Sasha
Date: 2023-03-22 10:47:06
LastEditors: Sasha
LastEditTime: 2023-04-10 10:58:04
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v3/mmoe_interactive_model.py
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from FedL.federatedml.dnn.backend.tensorflow.layers import MMoE_v3
from FedL.util.log import log
from  FedL.federatedml.dnn.model.base_nn_model import BaseNNModel

class MMoEInteractiveModel(BaseNNModel):
    def __init__(self,experts_units,experts_num,lr=0.01) -> None:
        super().__init__()
        self.experts_units = experts_units
        self.lr = lr
        self.experts_num = experts_num
        self._build_model()
    
    def _build_model(self):
        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.model = MMoE_v3(self.experts_units, self.experts_num)
        
    def train(self,inputs,grads):
        with tf.GradientTape() as tape_interactive:
            output_model = self.model(inputs)
            loss_interactive = tf.math.multiply(output_model,grads)

        grads = tape_interactive.gradient(loss_interactive, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def valid(self,inputs,label1,label2):
        predictions_task1,predictions_task2 = self.model(inputs)

        return  predictions_task1,predictions_task2
    