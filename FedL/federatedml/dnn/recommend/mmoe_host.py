'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-06 17:31:36
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v2/mmoe_host.py
'''
import numpy as np
from federatedml.data_factory import multi_task_split_data
from FedL.federatedml.dnn.recommend.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_host
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.federatedml.secureprotol.encrypt import PaillierEncrypt
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.secure_random import  generate_secure_noise
from FedL.federatedml.dnn.recommend.mmoe_bottom_model import MMoEBottomModel
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup 
import pickle
from FedL.federatedml.util.early_stopping import EarlyStopping

class MMoEHost(ModelBase):
    def __init__(self,train_x,feature_columns,config) -> None:
        super().__init__(fed_lmdb=feddb_host)
        self.config = config
        self.X = train_x
        self.acc_noise = None
        self.interactive_shape = None
        self.model_name = "nn_host_bottom.h5"
        self.acc_noise_name =  "acc_noise.pickle"
        self.feature_columns = feature_columns
        self._generate_key()
        self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.bottom_model = MMoEBottomModel(feature_columns=self.feature_columns,embed_dims=self.config['embed_dims'],lr=self.config['bottom_conf']['lr'])
    
    def _generate_key(self):
        cipher = PaillierEncrypt()
        cipher.generate_key(1024)
        self.public_key ,self.private_key = cipher.get_public_key(),cipher.get_privacy_key()
        
    def run(self,n):
        self.forward(self.X,n)
        self.backward(n)
    
    def forward(self,input,n):
        output_bottom_model = self.bottom_model(input).numpy()
        encrypted_output_bottom_host = self.public_key.encrypt_list(output_bottom_model)
        self.remote(encrypted_output_bottom_host,"encrypted_output_bottom_host_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)
        
        encrypted_marked_u_host_sub_acc = self.get("encrypted_marked_u_host_sub_acc_%s" % n)
        if self.interactive_shape is None:
            self.interactive_shape = (output_bottom_model.shape[1], encrypted_marked_u_host_sub_acc.shape[1])
        marked_u_host_sub_acc = self.private_key.decrypt_list(encrypted_marked_u_host_sub_acc)
        if self.acc_noise is None:
            self.acc_noise = np.zeros((output_bottom_model.shape[1], marked_u_host_sub_acc.shape[1]))
            
        marked_u_host = marked_u_host_sub_acc + output_bottom_model @ self.acc_noise
        self.remote(marked_u_host,"marked_u_host_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)

    def backward(self,n):
        # #####################################################帮助更新交互层参数 #####################################################
        tmp_noise = generate_secure_noise(self.interactive_shape)
        encryped_marked_grad_weight_host = self.get("encryped_marked_grad_weight_host_%s" % n)
        marked_grad_weight_host = self.private_key.decrypt_list(encryped_marked_grad_weight_host)
        marked_grad_weight_host_sub_noise = marked_grad_weight_host +   tmp_noise / self.decay.compute_step()
        self.remote(marked_grad_weight_host_sub_noise,"marked_grad_weight_host_sub_noise_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)
        # #####################################################反向传播获取梯度,更新bottom_model #####################################################
        encrypted_acc_noise = self.public_key.encrypt_list(self.acc_noise)
        self.remote(encrypted_acc_noise,"encrypted_acc_noise_%s" % n,ServiceConf.port_guest,ServiceConf.ip_guest)
        
        encryped_grad_bottom_host = self.get("encryped_grad_bottom_host_%s" % n)
        grad_bottom = self.private_key.decrypt_list(encryped_grad_bottom_host)
        self.bottom_model.train(self.X,grad_bottom)
        self.acc_noise += tmp_noise
        
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