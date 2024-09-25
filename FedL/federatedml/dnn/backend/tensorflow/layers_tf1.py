'''
Author: xueshuai
Date: 2022-04-28 20:25:33
LastEditors: Sasha
LastEditTime: 2023-03-20 14:24:24
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/backend/tensorflow/layers.py
'''
import tensorflow as tf
import numpy as np
class DNN(object):
    def __init__(self, dropout_keep_fm = 0.8,deep_layers = [512, 256], activation = tf.nn.relu):
        self.dropout_keep_fm = dropout_keep_fm
        self.activation = activation
        self.deep_layers = deep_layers
    #定义前向传播过程
    def call(self,input):
        # 一阶部分
        deep_embedding = input
        for i in range(len(self.deep_layers)):
            deep_embedding = tf.layers.dense(deep_embedding, self.deep_layers[i], activation=self.activation)
            deep_embedding = tf.nn.dropout(deep_embedding, self.dropout_keep_fm)
        return deep_embedding
class FM(object):
    def __init__(self, dropout_keep_fm):
        self.dropout_keep_fm = dropout_keep_fm
    #定义前向传播过程
    def call(self,embedding_first,embedding_second):
        # 一阶部分
        first_order = tf.nn.dropout(embedding_first, self.dropout_keep_fm)
        # 二阶部分
        sum_square = tf.square(tf.reduce_sum(embedding_second, 1))
        square_sum = tf.reduce_sum(tf.square(embedding_second), 1) 
        # 1/2*((a+b)^2 - a^2 - b^2)=ab
        second_order = 0.5 * tf.subtract(sum_square, square_sum)
        second_order = tf.nn.dropout(second_order, self.dropout_keep_fm)
        return tf.concat([first_order, second_order], axis=1)

class DeepFM(object):
    def __init__(self,dropout_keep_fm, feat_index_in,feat_value_in,embed_dim,column_features,deep_layers = [32, 16], activation = tf.nn.relu):
        self.column_features = column_features
        self.field_size = len(column_features)
        self.dropout_keep_fm = dropout_keep_fm
        self.activation = activation
        self.embed_dim = embed_dim
        self.deep_layers = deep_layers
        self.feat_index_in = feat_index_in
        self.feat_value_in = feat_value_in
        self.inputs = {}
        self.weight = {}
        self.embedding_second_output = {}
        self.embedding_first_output = {}
        self.FM = FM(self.dropout_keep_fm)
        self.DNN = DNN(self.dropout_keep_fm,self.deep_layers, self.activation)
        self.place_holder()
        self.init_weight()
        self.embedding_lookup()

    def place_holder(self):
        for columns in self.column_features:
            index = [i for i in range(columns.start, columns.end)]
            self.inputs[columns.name+"_index"] = tf.gather(self.feat_index_in, index, axis=1)
            self.inputs[columns.name+"_value"] = tf.gather(self.feat_value_in, index, axis=1)

    def init_weight(self):
        for columns in self.column_features:
            self.weight['embedding_first_%s' % columns.name] = tf.Variable(tf.random_normal([columns.feature_sizes, 1], 0.0, 1),name='embedding_first_%s' % columns.field_id)
            self.weight['embedding_second_%s' % columns.name] = tf.Variable(tf.random_normal([columns.feature_sizes, self.embed_dim], 0.0, 0.01),name='embedding_second_%s' % columns.field_id)
        # bais_first = tf.Variable(tf.zeros([1, 1]), name='zero_first', trainable=False)
        # embedding_first = tf.Variable(tf.random_normal([self.feature_sizes - 1, 1], 0.0, 1),name='embedding_first')
        # self.weight['embedding_first'] = tf.concat([bais_first, embedding_first], 0)

        # bais_second = tf.Variable(tf.zeros([1, self.embed_dim]), name='zero',trainable = False)
        # embedding_second = tf.Variable(tf.random_normal([self.feature_sizes - 1, self.embed_dim], 0.0, 0.01),name='embedding_second')        
        # self.weight['embedding_second'] = tf.concat([bais_second, embedding_second], 0)
      
    def embedding_lookup(self):
        
        for columns in self.column_features:
            # 二阶部分
            embedding_second = tf.nn.embedding_lookup(self.weight['embedding_second_%s' % columns.name],self.inputs[columns.name+"_index"])
            embedding_second_weighted = tf.multiply(embedding_second,tf.reshape(self.inputs[columns.name+"_value"], [-1, columns.input_length, 1]))
            self.embedding_second_output[columns.name] = tf.reduce_mean(embedding_second_weighted, 1)
        
            # 一阶部分
            embedding_first = tf.nn.embedding_lookup(self.weight['embedding_first_%s' % columns.name],self.inputs[columns.name+"_index"])
            embedding_first_weighted = tf.multiply(embedding_first,tf.reshape(self.inputs[columns.name+"_value"], [-1, columns.input_length, 1]))
            self.embedding_first_output[columns.name] = tf.reduce_mean(embedding_first_weighted, 1)
        
    #定义前向传播过程
    def call(self):
        # FM部分
        embedding_first_output = tf.concat([self.embedding_first_output[columns.name] for columns in self.column_features], axis=1)
        embedding_second_output = tf.concat([self.embedding_second_output[columns.name] for columns in self.column_features], axis=1)
        FM_out = self.FM.call(embedding_first_output,tf.reshape(embedding_second_output,[-1,self.field_size,self.embed_dim]))
        # DNN部分
        DNN_out = self.DNN.call(embedding_second_output)
        return tf.concat([FM_out, DNN_out], axis=1)


class MMOE(object):
    def __init__(self,experts_units, experts_num,input_dim,task_num = 2,user_gate_bias = True,use_experts_bias = True,activation = tf.nn.relu):
        self.task_num = task_num
        self.input_dim = input_dim
        self.experts_units = experts_units
        self.experts_num = experts_num
        self.use_experts_bias = use_experts_bias
        self.user_gate_bias = user_gate_bias
        self.activation = activation
        self.gates = {}
        self.gates_bias = {}
        self.task_output = {}
        self.init_weight()
        
    def init_weight(self):
        self.experts_weight = tf.get_variable(name='experts_weight',dtype=tf.float32,
                                     shape=(self.input_dim, self.experts_units, self.experts_num),
                                     initializer=tf.keras.initializers.glorot_normal(seed=None))
        self.experts_bias = tf.get_variable(name='expert_bias',
                                   dtype=tf.float32,
                                   shape=(self.experts_units, self.experts_num),
                                   initializer=tf.contrib.layers.xavier_initializer())
        # gates
        for i in range(self.task_num):
            self.gates[i] = tf.get_variable(name='gate_%d' % i,
                                            dtype=tf.float32,
                                            shape=(self.input_dim, self.experts_num),
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.gates_bias[i] = tf.get_variable(name='gate_bias_%d' % i,
                                                dtype=tf.float32,
                                                shape=(self.experts_num),
                                                initializer=tf.contrib.layers.xavier_initializer())
    def call(self,input):
        experts_output = tf.tensordot(input, self.experts_weight, axes=1)
        use_experts_bias = True
        if use_experts_bias:
            experts_output = tf.add(experts_output, self.experts_bias)
        experts_output = self.activation(experts_output)

        for i in range(self.task_num):
            gate_output = tf.tensordot(input, self.gates[i], axes=1)
            if self.user_gate_bias:
                gate_output = tf.add(gate_output, self.gates_bias[i])
            gate_output = tf.nn.softmax(gate_output)
            self.task_output[i] = tf.multiply(experts_output, tf.expand_dims(gate_output, axis=1))
            self.task_output[i] = tf.reduce_sum(self.task_output[i], axis=2)
            self.task_output[i] = tf.reshape(self.task_output[i], [-1, self.experts_units])

        return self.task_output

class ESMM(object):
    def __init__(self,hidden_units=[512, 256, 128],activation = tf.nn.relu):
        self.hidden_units = hidden_units
        self.activation = activation 
    def call(self,input):
        label1_input = input[0]
        label2_input = input[1]
        len_layers = len(self.hidden_units)
        with tf.variable_scope('ctr_deep'):
            dense_ctr = tf.layers.dense(inputs=label1_input, units=self.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_ctr = tf.layers.dense(inputs=dense_ctr, units=self.hidden_units[i], activation=tf.nn.relu)
            ctr_out = tf.layers.dense(inputs=dense_ctr, units=1)
            
        with tf.variable_scope('cvr_deep'):
            dense_cvr = tf.layers.dense(inputs=label2_input, units=self.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_cvr = tf.layers.dense(inputs=dense_cvr, units=self.hidden_units[i], activation=tf.nn.relu)
            cvr_out = tf.layers.dense(inputs=dense_cvr, units=1)
        ctr_score = tf.identity(tf.nn.sigmoid(ctr_out), name='ctr_score') # 在图中构造新的节点
        cvr_score = tf.identity(tf.nn.sigmoid(cvr_out), name='cvr_score')
        ctrcvr_score = ctr_score * cvr_score
        return ctr_score,cvr_score,ctrcvr_score
           
           
class CGC(object):
    def __init__(self,num_tasks,specific_expert_num,shared_expert_num, is_last=True,expert_dnn_hidden_units=[128,64,32],gate_dnn_hidden_units=[64,32]):
        self.num_tasks = num_tasks
        self.specific_expert_num = specific_expert_num
        self.shared_expert_num = shared_expert_num
        self.is_last = is_last
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
    def call(self,inputs):
        # inputs : [input] * (num_tasks + 1) 
        specific_expert_outputs = []
        
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                expert_network = DNN(deep_layers = self.expert_dnn_hidden_units).call(inputs[j])
                specific_expert_outputs.append(expert_network)

            shared_expert_outputs = []
        for k in range(self.shared_expert_num):
            expert_network = DNN(deep_layers = self.expert_dnn_hidden_units).call(inputs[-1])
            shared_expert_outputs.append(expert_network)

        cgc_outs = []
        for i in range(self.num_tasks):
            cur_expert_num = self.specific_expert_num + self.shared_expert_num
            cur_experts = specific_expert_outputs[i * self.specific_expert_num:(i + 1) * self.specific_expert_num] + shared_expert_outputs
            gate_input = DNN(deep_layers = self.gate_dnn_hidden_units).call(inputs[i])
            gate_out = tf.layers.dense(gate_input, cur_expert_num, activation=tf.nn.sigmoid)
            gate_out  = tf.multiply(tf.stack(cur_experts, axis=1),tf.expand_dims(gate_out, axis=2))
            gate_mul_expert = tf.reduce_sum(gate_out, axis=1)
            cgc_outs.append(gate_mul_expert)

        if not self.is_last:
            cur_expert_num = self.num_tasks * self.specific_expert_num + self.shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs
            gate_input = DNN(deep_layers = self.gate_dnn_hidden_units).call(inputs[-1])
            gate_out = tf.layers.dense(gate_input, cur_expert_num, activation=tf.nn.sigmoid)
            gate_out  = tf.multiply(tf.stack(cur_experts, axis=1),tf.expand_dims(gate_out, axis=2))
            gate_mul_expert = tf.reduce_sum(gate_out, axis=1)
            cgc_outs.append(gate_mul_expert)
        return cgc_outs

class PLE(object):
    def __init__(self, num_tasks,experts_task_num,experts_shared_num,expert_dnn_hidden_units=[128,64,32],gate_dnn_hidden_units=[64,32],num_levels = 2):
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.num_levels = num_levels
        self.experts_task_num = experts_task_num
        self.experts_shared_num = experts_shared_num
        self.num_tasks = num_tasks
    def call(self,input):
        ple_inputs = [input] * (self.num_tasks + 1)
        ple_outputs = []
        for i in range(self.num_levels):
            if i == self.num_levels - 1:  # the last level
                ple_outputs = CGC(num_tasks=self.num_tasks,specific_expert_num=self.experts_task_num,shared_expert_num=self.experts_shared_num,is_last=True).call(ple_inputs)
            else:
                ple_outputs = CGC(num_tasks=self.num_tasks,specific_expert_num=self.experts_task_num,shared_expert_num=self.experts_shared_num,is_last=False).call(ple_inputs)
                ple_inputs = ple_outputs
        return ple_outputs

        
if __name__ == '__main__':
    # model = PLE(num_tasks=2,experts_task_num=2,experts_shared_num=2)
    # input = tf.constant([[1,2,3,4,5,8,3,2,1,1], [1,2,3,4,5,8,3,2,1,1], [1,2,3,4,5,8,3,2,1,1]],dtype =tf.float32)
    # model.call(input)
    
    model = MMOE(experts_units=5, experts_num=3,input_dim=10)
    input = tf.constant([[1,2,3,4,5,8,3,2,1,1], [1,2,3,4,5,8,3,2,1,1], [1,2,3,4,5,8,3,2,1,1]],dtype =tf.float32)
    model.call(input)