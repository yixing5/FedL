'''
Author: Sasha
Date: 2023-03-10 09:36:45
LastEditors: Sasha
LastEditTime: 2023-04-10 10:58:12
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v3/mmoe_top_model.py
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from FedL.federatedml.dnn.backend.tensorflow.layers import ESMM
from FedL.util.log import log
from  FedL.federatedml.dnn.model.base_nn_model import BaseNNModel

class MMoETopModel(BaseNNModel):
    def __init__(self,experts_units,experts_num,lr=0.01) -> None:
        super().__init__()
        self.experts_units = experts_units
        self.lr = lr
        self.experts_num = experts_num
        self._build_model()
    
    def _build_model(self):
        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.model = ESMM(self.experts_units, self.experts_num)
        self.valid_loss1 = keras.losses.BinaryCrossentropy()
        self.valid_loss2 = keras.losses.BinaryCrossentropy()
        self.auc_func = keras.metrics.AUC()
        self.loss_func1 = keras.losses.BinaryCrossentropy()
        self.loss_func2 = keras.losses.BinaryCrossentropy()
        # self.weight_loss = tf.Variable([0.5])

    def train(self,inputs,label1,label2):
        inputs = tf.constant(inputs)
        with tf.GradientTape() as tape_top:
            tape_top.watch(inputs)
            predictions_task1,predictions_task2 = self.model(inputs)
            loss_task1 = self.loss_func1(tf.reshape(label1,[-1]), tf.reshape(predictions_task1,[-1]))
            loss_task2 = self.loss_func2(tf.reshape(label2,[-1]), tf.reshape(predictions_task2,[-1]))
            loss_sum = loss_task1 + loss_task2
        train_auc1 = self.auc_func(tf.reshape(label1,[-1]), tf.reshape(predictions_task1,[-1]))
        train_auc2 = self.auc_func(tf.reshape(label2,[-1]), tf.reshape(predictions_task2,[-1]))
        log.info("##########sum_loss: %s ##loss1: %s  loss2: %s###   auc1:  %s########### auc2:  %s##########################################" % (loss_sum.numpy(),loss_task1.numpy(),loss_task2.numpy(),train_auc1.numpy(),train_auc2.numpy()))
        grads_act,grads = tape_top.gradient(loss_sum, [inputs,self.model.trainable_variables])
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return grads_act.numpy()
    
    def valid(self,inputs,label1,label2):
        predictions_task1,predictions_task2 = self.model(inputs)
        valid_auc1 = self.auc_func(tf.reshape(label1,[-1]), tf.reshape(predictions_task1,[-1]))
        valid_auc2 = self.auc_func(tf.reshape(label2,[-1]), tf.reshape(predictions_task2,[-1]))
        valid_loss1 = self.valid_loss1(tf.reshape(label1,[-1]), tf.reshape(predictions_task1,[-1]))
        valid_loss2 = self.valid_loss2(tf.reshape(label2,[-1]), tf.reshape(predictions_task2,[-1]))
        return  predictions_task1,predictions_task2,valid_loss1,valid_loss2,valid_auc1,valid_auc2
    
