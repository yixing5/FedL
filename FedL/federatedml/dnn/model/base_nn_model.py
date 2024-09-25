'''
Author: Sasha
Date: 2023-03-24 15:46:18
LastEditors: Sasha
LastEditTime: 2023-03-24 18:33:00
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/model/base_nn_model.py
'''
from FedL.util.utils import makedirs
from FedL.conf.global_conf import Path ,ModelInfo
import tensorflow as tf
class BaseNNModel():
    def __init__(self) -> None:
        self.model_path = Path.model_path
        self.version = ModelInfo.version
        
    def __call__(self,input):
        return self.model(input)
    
    def get_model_path(self,filename):
        path = "%s/%s" % (self.model_path,self.version)
        makedirs(path)
        return "%s/%s" % (path,filename)
    
    def save(self,filename):
        # self.model.save(self.get_model_path(filename))
        self.model.save_weights(self.get_model_path(filename))
    
    def load_model(self,filename):
        self.model.load_weights(self.get_model_path(filename))
        # self.model = tf.keras.models.load_model(self.get_model_path(filename))