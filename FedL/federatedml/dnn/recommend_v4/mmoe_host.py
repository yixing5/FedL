'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-13 18:15:54
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v4/mmoe_host.py
'''
import numpy as np
from federatedml.data_factory import multi_task_split_data
from FedL.federatedml.dnn.recommend_v4.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_host
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.federatedml.secureprotol.encrypt import PaillierEncrypt
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.secure_random import  generate_secure_noise
from FedL.federatedml.dnn.recommend_v4.mmoe_bottom_model import MMoEBottomModel
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup 
from FedL.federatedml.dnn.recommend_v4.mmoe_interactive_model import MMoEInteractiveModel
import pickle
from FedL.federatedml.util.early_stopping import EarlyStopping

class MMoEHost(ModelBase):
    def __init__(self,train_x,feature_columns,feature_num_B,config) -> None:
        super().__init__(fed_lmdb=feddb_host)
        self.config = config
        self.X = train_x
        self.acc_noise = None
        self.interactive_shape = None
        self.model_name = "nn_host_bottom.h5"
        self.acc_noise_name =  "acc_noise.pickle"
        self.feature_columns = feature_columns
        self._generate_key()
        self.interactive_out_dim = self.config['task_num']*self.config['experts_num'] + self.config['experts_num']*self.config['experts_units']

        self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.bottom_model = MMoEBottomModel(feature_columns=self.feature_columns,embed_dims=self.config['embed_dims'],lr=self.config['bottom_conf']['lr'])
        self.interactive_model = MMoEInteractiveModel(shape_host=(self.config["embed_dims"]*len(self.feature_columns) ,self.interactive_out_dim),
                                                shape_guest=(self.config["embed_dims"]*feature_num_B,self.interactive_out_dim),
                                                lr = self.config['interactive_conf']['lr'],
                                                decay = self.decay)
        
    def _generate_key(self):
        cipher = PaillierEncrypt()
        cipher.generate_key(1024)
        self.public_key ,self.private_key = cipher.get_public_key(),cipher.get_privacy_key()
        
    def run(self,n):
        self.forward(self.X,n)
        self.backward(n)
    
    def forward(self,input,n,name='train'):
        output_bottom_host = self.bottom_model(input).numpy()
        u_host = self.interactive_model.forward_host(output_bottom_host,name=name)
        
        encrypted_output_bottom_guest = self.get("encrypted_output_bottom_guest_%s" % n)
        encrypted_u_guest = self.interactive_model.forward_guest(encrypted_output_bottom_guest,name=name)

        encrypted_output_interactice = u_host + encrypted_u_guest
        self.remote(encrypted_output_interactice,"encrypted_output_interactice_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)
        
    def backward(self,n):
        
        ###step 4
        grads_act = self.get("grads_act_%s" % n)
        encryped_acc_noise = self.get("encryped_acc_noise_%s" % n)
        
        delta_grad_weight_guest = self.get("delta_grad_weight_guest_%s" % n)
        encryped_grad_weight_host =  self.interactive_model.backward_host(grads_act)
        self.interactive_model.train(encryped_grad_weight_host ,delta_grad_weight_guest)
        
        ###step 5
        encryped_grad_bottom_guest = self.interactive_model.get_grad_bottom_guest(grads_act,encryped_acc_noise)
        self.remote(encryped_grad_bottom_guest,"encryped_grad_bottom_guest_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)
        grad_bottom_host = self.interactive_model.get_grad_bottom_host(grads_act)  
        self.bottom_model.train(self.X,grad_bottom_host)
        
    def predict(self,inputs):
        self.forward(inputs,'predict')
        pred = self.get("pred")
        log.info('pred:',pred)
        
    def valid(self,inputs):
        self.forward(inputs,'valid')
        valid_loss = self.get("valid_loss")
        log.info('host valid_loss: %s' % valid_loss)
        return valid_loss
    
    def get_acc_noise_path(self):
        return self.bottom_model.get_model_path
    
    def save(self):
        self.bottom_model.save(self.model_name)
        with open(self.bottom_model.get_model_path(self.acc_noise_name), 'wb') as fw:
            pickle.dump(self.acc_noise, fw)
    
    def load_model(self):
        self.bottom_model.load_model(self.model_name)
        with open(self.bottom_model.get_model_path(self.acc_noise_name), 'rb') as fr:
            self.acc_noise =  pickle.load(fr, encoding='bytes')

        
if __name__ == "__main__":
    try:
        feature_columns_A,feature_columns_B,XA_train,XB_train,XA_test,XB_test, y1_train, y1_test,y2_train, y2_test = multi_task_split_data()
        feature_num_B = len(feature_columns_B)
        # ids = [x+10 for x in range(len(XA_train)-10)]
        # #隐私求交
        # psi_client = PSIClient(ids = ids)
        # common_ids = psi_client.run()
        # XA_train = XA_train[common_ids]    
        mmoe_host = MMoEHost(XA_train,feature_columns_A,feature_num_B,config)
        early_stopping = EarlyStopping()

        for n in range(config['epoch']):
            mmoe_host.run(n)
            # if n % 50 == 0:
            #     valid_loss1,valid_loss2 = mmoe_host.valid(XA_test)
            #     early_stopping(valid_loss1+valid_loss2)
            #     if early_stopping.early_stop:
            #         log.info("Early stopping")
            #         mmoe_host.save()
            #         break
        # mmoe_host.load_model()
        mmoe_host.predict(XA_test)
    finally:
        mmoe_host.fed_lmdb.drop()