'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-07 16:30:54
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v2/mmoe_host.py
'''
import numpy as np
from federatedml.data_factory import multi_task_split_data
from FedL.federatedml.dnn.recommend_v2.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_host
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.secure_random import  generate_secure_noise
from FedL.federatedml.dnn.recommend_v2.mmoe_bottom_model import MMoEBottomModel
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup 
import pickle
from FedL.federatedml.util.early_stopping import EarlyStopping




class MMoEHost(ModelBase):
    def __init__(self,train_x,feature_columns,config) -> None:
        super().__init__(fed_lmdb=feddb_host)
        self.role = 'host'
        self.config = config
        self.X = train_x
        self.acc_noise = None
        self.interactive_shape = None
        self.model_name = "nn_host_bottom.h5"
        self.acc_noise_name =  "acc_noise.pickle"
        self.feature_columns = feature_columns
        self.bottom_out_dim = self.config['task_num']*self.config['experts_num'] + self.config['experts_num']*self.config['experts_units']
        self.other_public_key = None
        self._generate_key('rsa')
        self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.bottom_model = MMoEBottomModel(feature_columns=self.feature_columns,embed_dims=self.config['embed_dims'],bottom_out_dim = self.bottom_out_dim,lr=self.config['bottom_conf']['lr'])
    
    
    def exchange_public_key(self,target_role):
        self.remote(self.public_key,"%s_public_key" % self.role,ServiceConf.port_guest,ServiceConf.ip_guest)
        self.other_public_key = self.get("%s_public_key" % target_role)
    
    def run(self,n):
        if self.other_public_key is None:
            self.exchange_public_key('guest')
        self.forward(self.X,n)
        self.backward(n)
    
    def forward(self,input,n):
        output_bottom_model = self.bottom_model(input).numpy()
        encrypted_output_bottom_host = self.other_public_key.encrypt_list(output_bottom_model)
        self.remote(encrypted_output_bottom_host,"encrypted_output_bottom_host_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)
        
    def backward(self,n):
        # #####################################################帮助更新交互层参数 #####################################################
        encryped_grads_act = self.get("encryped_grads_act_%s" % n)
        grads_act = self.private_key.decrypt_list(encryped_grads_act)
        self.bottom_model.train(self.X,grads_act)
        
    def predict(self,inputs):
        self.forward(inputs,'predict')
        pred = self.get("pred")
        log.info('pred:',pred)
        
    def valid(self,inputs):
        self.forward(inputs,'valid')
        valid_loss = self.get("valid_loss")
        log.info('host valid_loss: %s' % valid_loss)
        return valid_loss
    
    def save(self):
        self.bottom_model.save(self.model_name)

    def load_model(self):
        self.bottom_model.load_model(self.model_name)


        
if __name__ == "__main__":
    try:
        feature_columns_A,feature_columns_B,XA_train,XB_train,XA_test,XB_test, y1_train, y1_test,y2_train, y2_test = multi_task_split_data()
        # ids = [x+10 for x in range(len(XA_train)-10)]
        # #隐私求交
        # psi_client = PSIClient(ids = ids)
        # common_ids = psi_client.run()
        # XA_train = XA_train[common_ids]    
        mmoe_host = MMoEHost(XA_train,feature_columns_A,config)
        early_stopping = EarlyStopping()

        for n in range(config['epoch']):
            mmoe_host.run(n)
            if n % 50 == 0:
                valid_loss1,valid_loss2 = mmoe_host.valid(XA_test)
                early_stopping(valid_loss1+valid_loss2)
                if early_stopping.early_stop:
                    log.info("Early stopping")
                    mmoe_host.save()
                    break
        # mmoe_host.load_model()
        mmoe_host.predict(XA_test)
    finally:
        mmoe_host.fed_lmdb.drop()