'''
Author: Sasha
Date: 2023-02-27 14:50:20
LastEditors: Sasha
LastEditTime: 2023-04-03 17:30:20
Description: 
FilePath: /FederatedLearning/FedL/federatedml/linear_model/logistic_regression/linr_Logistic_host.py
'''
import numpy as np
from FedL.federatedml.data_factory import vertical_split_data
from FedL.federatedml.linear_model.linear_regression.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_host
from FedL.federatedml.PSI.psi_client import PSIClient
from FedL.federatedml.util.learn_rate_decay import cosine_decay_with_warmup


class LinrLogisticHost(ModelBase):
    def __init__(self,train_x,config) -> None:
        super().__init__(fed_lmdb=feddb_host)
        self.config = config
        self.X = train_x
        self.weights = np.zeros(self.X.shape[1])
        
    def get_public_key(self):
        self.public_key = self.get("public_key_host")
    
    def compute_u(self):
        return self.X.dot(self.weights)
    
    def cumputer_ua_L(self):
        #计算自己的部分
        u_a = self.compute_u()
        u_a_square = u_a**2
        # L_a = np.sum(u_a_square) +  self.config["lambda"] * np.sum(self.weights**2)
        encrypted_u_a = np.array(self.public_key.encrypt_list(u_a))
        encrypted_u_a_square = np.array(self.public_key.encrypt_list(u_a_square))
        return encrypted_u_a,encrypted_u_a_square
        # 计算加密梯度
    
    def compute_encrypted_dL_a(self,encrypted_d):
        encrypted_dL_a = self.X.T.dot(encrypted_d) + self.config["lambda"] * self.weights
        return encrypted_dL_a
    
    def run(self,n):
        self.n = n
        # 获取公钥
        #############################################计算[[u_a]],[[L_a]]发送给B方#############################################################
        encrypted_u_a,encrypted_u_a_square = self.cumputer_ua_L()
        self.remote(encrypted_u_a_square,"encrypted_u_a_square_%s" % self.n,"50052") 
        self.remote(encrypted_u_a,"encrypted_u_a_%s" % self.n,"50052")
        ##############################################计算加密梯度[[dL_a]]，加上随机数之后，发送给C############################################################
        encrypted_d = self.get("encrypted_d_%s" % self.n)
        encrypted_dL_a = self.compute_encrypted_dL_a(encrypted_d)
        self.mask = np.random.rand(len(encrypted_dL_a))
        encrypted_masked_dL_a = encrypted_dL_a + self.mask
        self.remote(encrypted_masked_dL_a,"encrypted_masked_dL_a_%s" % self.n,"50051")
        ##############################################获取解密后的masked梯度，减去mask，梯度下降更新############################################################
        masked_dL_a = self.get("masked_dL_a_%s" % self.n)
        dL_a = masked_dL_a - self.mask
        # 注意这里的1/n
        # self.weights = self.weights - self.config["lr"] * dL_a / len(self.X)
        self.weights = self.weights - cosine_decay_with_warmup(self.n,config['epoch'],lr_max=config['lr']) * dL_a / len(self.X)
        print("host weights {} : {}".format(self.n,self.weights))
        # self.fed_lmdb.delete_list(["%s%s" % ( x,self.n) for x in ["encrypted_d_","masked_dL_a_"]])

        # self.remote(b'success',"state_host_%s" % self.n,'50051')

    def predict(self,X_test):
        u_a = X_test.dot(self.weights)
        encrypted_u_a = np.array(self.public_key.encrypt_list(u_a))
        self.remote(encrypted_u_a,"encrypted_u_a",'50051')
        pred = self.get("pred")
        print('pred:',pred)
        
if __name__ == "__main__":
    try:
        XA_train,XB_train,XA_test,XB_test,y_train,y_test = vertical_split_data()
        # ids = [x+10 for x in range(len(XA_train)-10)]
        # #隐私求交
        # psi_client = PSIClient(ids = ids)
        # common_ids = psi_client.run()
        # XA_train = XA_train[common_ids]
        
        line_host = LinrLogisticHost(XA_train,config)
        line_host.get_public_key()
        for n in range(config['epoch']):
            line_host.run(n)
        line_host.predict(XA_test)
    finally:
        print("exe finally")
        line_host.fed_lmdb.drop()