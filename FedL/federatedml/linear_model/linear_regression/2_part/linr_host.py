'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-03 17:29:54
Description: 
FilePath: /FederatedLearning/FedL/federatedml/linear_model/linear_regression/2_part/linr_host.py
'''

import numpy as np
from FedL.federatedml.data_factory import split_data
from FedL.federatedml.linear_model.linear_regression.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_host
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.federatedml.secureprotol.encrypt import PaillierEncrypt
from FedL.federatedml.util.learn_rate_decay import CosineDecayWithWarmup
from FedL.conf.global_conf import ServiceConf
from FedL.util.log import log
from FedL.federatedml.util.yk_operator import dot
from FedL.federatedml.util.secure_random import  generate_secure_noise

class LinrHost(ModelBase):
    def __init__(self,train_x,config) -> None:
        super().__init__(fed_lmdb=feddb_host)
        self.config = config
        self.X = train_x
        self.weights = np.zeros(self.X.shape[1])
        cipher = PaillierEncrypt()
        cipher.generate_key(1024)
        self.public_key ,self.private_key = cipher.get_public_key(),cipher.get_privacy_key()
        # self.decay = CosineDecayWithWarmup(self.config['epoch'],self.config['lr_max'])
        self.data = {}
        
    def compute_u(self):
        return self.X.dot(self.weights)
    
    def cumputer_ua_L(self):
        #计算自己的部分
        u_a = self.compute_u()
        u_a_square = u_a**2
        L_a = 0.5 * np.sum(u_a_square) + 0.5 * self.config["lambda"] * np.sum(self.weights**2)
        encrypted_u_a = self.public_key.encrypt_list(u_a)
        encrypted_L_a = self.public_key.encrypt(L_a)
        return u_a,encrypted_u_a,encrypted_L_a
        # 计算加密梯度
    def compute_encrypted_dL_a(self,encrypted_d):
        encrypted_dL_a = dot(self.X.T,encrypted_d) + self.config["lambda"] * self.weights
        # encrypted_dL_a = self.X.T.dot(encrypted_d) + self.config["lambda"] * self.weights
        return encrypted_dL_a
    
    def run(self,n):
        self.n = n
        #############################################计算[[u_a]],[[L_a]]发送给B方#############################################################
        u_a,encrypted_u_a,encrypted_L_a = self.cumputer_ua_L()
        self.remote(encrypted_L_a,"encrypted_L_a_%s" % self.n,ServiceConf.port_guest,ServiceConf.ip_guest)
        self.remote(encrypted_u_a,"encrypted_u_a_%s" % self.n,ServiceConf.port_guest,ServiceConf.ip_guest)
        ##############################################计算加密梯度[[dL_a]]a，加上随机数之后，解码[[dL_a + mark]]a 发送给B############################################################
        encrypted_L = self.get("encrypted_L_%s" % self.n)
        Loss = self.private_key.decrypt(encrypted_L)/len(self.X)
        self.remote(Loss,"Loss_%s" % self.n,ServiceConf.port_guest,ServiceConf.ip_guest)
        
        encrypted_u_b = self.get("encrypted_u_b_%s" % self.n)
        encrypted_d = encrypted_u_b + u_a
        encrypted_dL_a = self.compute_encrypted_dL_a(encrypted_d)
        self.mask = generate_secure_noise(encrypted_dL_a.shape) # np.random.rand(len(encrypted_dL_a))
        
        encrypted_masked_dL_a = encrypted_dL_a + self.mask
        self.remote(encrypted_masked_dL_a,"encrypted_masked_dL_a_%s" % self.n,ServiceConf.port_guest,ServiceConf.ip_guest)
        
        encrypted_masked_dL_b = self.get("encrypted_masked_dL_b_%s" % self.n)
        
        import time
        start = time.time()
        masked_dL_b = self.private_key.decrypt_list(encrypted_masked_dL_b)
        end = time.time()
        print(end - start, 's')
        
        self.remote(masked_dL_b,"masked_dL_b_%s" % self.n,ServiceConf.port_guest,ServiceConf.ip_guest)
        ##############################################获取解密后的masked梯度，减去mask，梯度下降更新############################################################
        masked_dL_a = self.get("masked_dL_a_%s" % self.n)
        dL_a = masked_dL_a - self.mask
        # 注意这里的1/n
        self.weights = self.weights - self.config["lr"] * dL_a / len(self.X)
        # self.weights = self.weights - cosine_decay_with_warmup(self.n,config['epoch'],lr_max=config['lr']) * dL_a / len(self.X)

        log.info("loss: {} guest weights {} : {}".format(round(Loss,4),self.n,self.weights))

        
    def predict(self,X_test):
        u_a = X_test.dot(self.weights)
        self.remote(u_a,"pred_u_a",ServiceConf.port_guest,ServiceConf.ip_guest)
        pred = self.get("pred_u_b") + u_a
        log.info('pred: %s '% pred)
        
if __name__ == "__main__":
    try:
        XA_train,XB_train,XA_test,XB_test,y_train,y_test = split_data()
        # ids = [x+10 for x in range(len(XA_train)-10)]
        # #隐私求交
        # psi_client = PSIClient(ids = ids)
        # common_ids = psi_client.run()
        # XA_train = XA_train[common_ids]    
        line_host = LinrHost(XA_train,config)
        
        for n in range(config['epoch']):
            line_host.run(n)
        line_host.predict(XA_test)
    finally:
        line_host.fed_lmdb.drop()