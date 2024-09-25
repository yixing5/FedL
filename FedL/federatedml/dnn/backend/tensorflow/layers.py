'''
Author: Sasha
Date: 2023-03-20 17:12:06
LastEditors: Sasha
LastEditTime: 2023-04-10 10:54:16
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/backend/tensorflow/layers.py
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models,layers
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np

print(tf.config.list_logical_devices())


class DNN(models.Model):
    def __init__(self,deep_layers = [32, 16],activation='relu'):
        super(DNN, self).__init__()
        self.activation = activation
        self.deep_layers = deep_layers
        self.denses = []
    def build(self,input_shape):
        for i in range(len(self.deep_layers)):
            self.denses.append(layers.Dense(self.deep_layers[i],kernel_regularizer=keras.regularizers.l2(1e-2),activation = self.activation,name = "dense%s" % i) )
        super(DNN,self).build(input_shape)
        self.compute_output_shape(input_shape)
    # 正向传播
    # @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def call(self,x):
        for layer in self.denses:
            x = layer(x)
        return x


class Embeddings(models.Model):
    def __init__(self,feature_columns,embed_dims):
        super(Embeddings, self).__init__()
        self.feature_columns = feature_columns
        self.embed_dims = embed_dims
        self.embeddings = {}
        
    def build(self,input_shape):
        for columns in self.feature_columns:
            self.embeddings[columns.name] = keras.layers.Embedding(columns.feature_sizes + 1 , self.embed_dims, input_length=columns.input_length)
    # 正向传播
    # @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def call(self,inputs):
        embedding_output = tf.concat([self.embeddings[columns.name](inputs[columns.name].values) for columns in self.feature_columns], axis=1)
        return embedding_output


class Embeddings_v2(models.Model):
    def __init__(self,feature_columns,embed_dims,output_dims):
        super(Embeddings_v2, self).__init__()
        self.feature_columns = feature_columns
        self.output_dims = output_dims
        self.embed_dims = embed_dims
        self.embeddings = {}
        
    def build(self,input_shape):
        self.dense = layers.Dense(self.output_dims,kernel_regularizer=keras.regularizers.l2(1e-2))
        for columns in self.feature_columns:
            self.embeddings[columns.name] = keras.layers.Embedding(columns.feature_sizes + 1 , self.embed_dims, input_length=columns.input_length)
    # 正向传播
    # @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def call(self,inputs):
        embedding_output = tf.concat([self.embeddings[columns.name](inputs[columns.name].values) for columns in self.feature_columns], axis=1)
        return self.dense(embedding_output)
    

class _MMOE(models.Model):
    def __init__(self,experts_units, experts_num,input_dim,task_num = 2,activation = 'relu'):
        self.task_num = task_num
        self.input_dim = input_dim
        self.experts_units = experts_units
        self.experts_num = experts_num
        self.activation = activation
        self.gates = {}
        self.task_output = {}
        
    def build(self,input_shape):
        self.experts_net = layers.Dense(self.experts_units*self.experts_num,kernel_regularizer=keras.regularizers.l2(1e-2),activation=self.activation)
        # gates
        for i in range(self.task_num):
            self.gates[i] = layers.Dense(self.experts_num,kernel_regularizer=keras.regularizers.l2(1e-2),activation='softmax')
        super(_MMOE,self).build(input_shape)
        
    def call(self,inputs):
        experts_output = tf.reshape(self.experts_net(inputs),[-1,self.experts_units,self.experts_num]) 
        # gates
        for i in range(self.task_num):
            gate_output = self.gates[i](inputs)
            task_output = tf.multiply(experts_output, tf.expand_dims(gate_output, axis=1))
            task_output = tf.reduce_sum(task_output, axis=2)
            task_output = tf.reshape(task_output, [-1, self.experts_units])
        return self.task_output

class MMoE(models.Model):
    def __init__(self,experts_units, experts_num,task_num = 2,activation = 'relu'):
        super(MMoE, self).__init__()
        self.task_num = task_num
        self.experts_units = experts_units
        self.experts_num = experts_num
        self.activation = activation
        self.gates = {}

    
    def split(self,inputs):
        indexs = [self.experts_num for  _ in range(self.task_num) ] + [self.experts_units * self.experts_num ]
        output = tf.split(inputs,indexs,axis = 1)
        return output[:self.task_num],output[-1]
    
    def build(self, input_shape):
        self.last_layer = []
        for _ in range(self.task_num):
             self.last_layer.append(layers.Dense(1,kernel_regularizer=keras.regularizers.l2(1e-2),activation=tf.nn.sigmoid))
        return super().build(input_shape)
    
    def call(self,inputs):
        gate_iput,experts_output = self.split(inputs)
        experts_output = tf.reshape(tf.nn.relu(experts_output),[-1,self.experts_units,self.experts_num]) 
        # gates
        task_outputs = []
        for i in range(self.task_num):
            gate_output = tf.nn.softmax(gate_iput[i]) 
            task_output = tf.multiply(experts_output, tf.expand_dims(gate_output, axis=1))
            task_output = tf.reduce_sum(task_output, axis=2)
            task_output = tf.reshape(task_output, [-1, self.experts_units])
            task_output = self.last_layer[i](task_output)
            task_outputs.append(task_output)
        return task_outputs

class MMoE_v3(models.Model):
    def __init__(self,experts_units, experts_num,task_num = 2,activation = 'relu'):
        super(MMoE, self).__init__()
        self.task_num = task_num
        self.experts_units = experts_units
        self.experts_num = experts_num
        self.activation = activation
        self.gates = {}

    
    def split(self,inputs):
        indexs = [self.experts_num for  _ in range(self.task_num) ] + [self.experts_units * self.experts_num ]
        output = tf.split(inputs,indexs,axis = 1)
        return output[:self.task_num],output[-1]
    
    # def build(self, input_shape):
    #     self.last_layer = []
    #     for _ in range(self.task_num):
    #          self.last_layer.append(layers.Dense(1,kernel_regularizer=keras.regularizers.l2(1e-2),activation=tf.nn.sigmoid))
    #     return super().build(input_shape)
    
    def call(self,inputs):
        gate_iput,experts_output = self.split(inputs)
        experts_output = tf.reshape(tf.nn.relu(experts_output),[-1,self.experts_units,self.experts_num]) 
        # gates
        task_outputs = []
        for i in range(self.task_num):
            gate_output = tf.nn.softmax(gate_iput[i]) 
            task_output = tf.multiply(experts_output, tf.expand_dims(gate_output, axis=1))
            task_output = tf.reduce_sum(task_output, axis=2)
            task_output = tf.reshape(task_output, [-1, self.experts_units])
            task_outputs.append(task_output)
        return task_outputs


class ESMM(models.Model):
    def __init__(self,task_num,deep_layers = [32, 16],activation=['relu','sigmoid']):
        super(DNN, self).__init__()
        self.activation = activation
        self.task_num = task_num
        self.deep_layers = deep_layers
        self.denses = []
    def build(self,input_shape):
        for _ in range(self.task_num):
            task_denses = []
            for i in range(len(self.deep_layers)):
                task_denses.append(layers.Dense(self.deep_layers[i],kernel_regularizer=keras.regularizers.l2(1e-2),activation = self.activation[i]))
            self.denses.append(task_denses)
        super(ESMM,self).build(input_shape)
        self.compute_output_shape(input_shape)
    # 正向传播
    # @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def call(self,x):
        for i in range(self.task_num):
            for layer in self.denses[i]:
                x[i] = layer(x[i])
        return x

# if __name__ == "__main__":
#     model = DNN()
#     model.build((None,10))
#     model.save("/data/sasha/FederatedLearning/FedL/federatedml/dnn/model_saved/2023-03-24/test")
#     model.load_weights
