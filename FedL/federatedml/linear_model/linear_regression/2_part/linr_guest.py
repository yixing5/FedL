'''
Author: Sasha
Date: 2023-02-27 14:46:55
LastEditors: Sasha
LastEditTime: 2023-04-03 17:29:00
Description: 
FilePath: /FederatedLearning/FedL/federatedml/linear_model/linear_regression/2_part/linr_guest.py
'''
import numpy as np
from FedL.federatedml.data_factory import split_data
from FedL.federatedml.linear_model.linear_regression.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_guest
from FedL.federatedml.PSI.psi_service import PSIService
from federatedml.secureprotol.encrypt import PaillierEncrypt
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.yk_operator import dot
from FedL.federatedml.util.secure_random import  generate_secure_noise

class LinrGuest(ModelBase):
    def __init__(self,train_x,label,config) -> None:
        super().__init__(fed_lmdb=feddb_guest)
        self.config = config
        self.X = train_x
        self.y = label
        self.weights = np.zeros(self.X.shape[1])
        
        cipher = PaillierEncrypt()
        cipher.generate_key(1024)
        self.public_key ,self.private_key = cipher.get_public_key(),cipher.get_privacy_key()
               
    def compute_u(self):
        return self.X.dot(self.weights)
    
    def cumputer_d_L(self):
        encrypted_u_a = self.get("encrypted_u_a_%s" % self.n)
        #计算自己的部分
        u_guest = self.compute_u()
        z_b = u_guest - self.y
        encrypted_d = encrypted_u_a + z_b
        L_b = 0.5 * np.sum(z_b**2) + 0.5 * self.config["lambda"] * np.sum(self.weights**2)
        L_ab = np.sum(encrypted_u_a * z_b)
        encrypted_L_a = self.get("encrypted_L_a_%s" % self.n)
        encrypted_L = encrypted_L_a + L_b + L_ab
        encrypted_u_b = np.array(self.public_key.encrypt_list(z_b))
        return encrypted_d,encrypted_L,encrypted_u_b
        # 计算加密梯度
    def compute_encrypted_dL_b(self,encrypted_d):
        encrypted_dL_b = dot(self.X.T,encrypted_d) + self.config["lambda"] * self.weights
        # encrypted_dL_b = self.X.T.dot(encrypted_d) + self.config["lambda"] * self.weights
        return encrypted_dL_b
    
    def run(self,n):
        self.n = n
        #############################################发送加密两次的d给A#########################################################
        self.encrypted_d,encrypted_L,encrypted_u_b = self.cumputer_d_L()
        print("encrypted_u_b_%s" % self.n,ServiceConf.port_host,ServiceConf.ip_host)
        self.remote(encrypted_u_b,"encrypted_u_b_%s" % self.n,ServiceConf.port_host,ServiceConf.ip_host)
        self.remote(encrypted_L,"encrypted_L_%s" % self.n,ServiceConf.port_host,ServiceConf.ip_host)
        Loss = self.get("Loss_%s" % self.n)
        #############################################计算加密梯度[[dL_b]],mask之后发给A方#############################################################
        
        encrypted_dL_b = self.compute_encrypted_dL_b(self.encrypted_d)
        self.mask = generate_secure_noise(encrypted_dL_b.shape) #np.random.rand(len(encrypted_dL_b))
        encrypted_masked_dL_b = encrypted_dL_b + self.mask
        self.remote(encrypted_masked_dL_b,"encrypted_masked_dL_b_%s" % self.n,ServiceConf.port_host,ServiceConf.ip_host)
        
        encrypted_masked_dL_a = self.get("encrypted_masked_dL_a_%s" % self.n)
        masked_dL_a = self.private_key.decrypt_list(encrypted_masked_dL_a)
        self.remote(masked_dL_a,"masked_dL_a_%s" % self.n,ServiceConf.port_host,ServiceConf.ip_host)
        #############################################获取解密后的梯度，解mask，模型更新#############################################################        
        masked_dL_b = self.get("masked_dL_b_%s" % self.n)
        dL_b = masked_dL_b - self.mask
        
        self.weights = self.weights - self.config["lr"] * dL_b / len(self.X)
        # self.weights = self.weights - cosine_decay_with_warmup(self.n,config['epoch'],lr_max=config['lr']) * dL_b / len(self.X)
        log.info("loss: {} guest weights {} : {}".format(round(Loss,4),self.n,self.weights))
        
    def predict(self,X_test):
        u_b = X_test.dot(self.weights)
        self.remote(u_b,"pred_u_b",ServiceConf.port_host,ServiceConf.ip_host)
        pred = self.get("pred_u_a") + u_b
        log.info('pred: %s '% pred)
            
if __name__ == "__main__":
    try:
        XA_train,XB_train,XA_test,XB_test,y_train,y_test = split_data()
        #隐私求交
        # ids = [x for x in range(len(XA_train)-10)]
        # psi_service = PSIService(ids = ids)
        # common_ids = psi_service.run()
        # XB_train ,y_train= XB_train[common_ids],y_train[common_ids]
        #训练
        line_guest = LinrGuest(XB_train,y_train,config)
        for n in range(config['epoch']):
            line_guest.run(n)
        #预测
        line_guest.predict(XB_test)
    finally:
        line_guest.fed_lmdb.drop()
    