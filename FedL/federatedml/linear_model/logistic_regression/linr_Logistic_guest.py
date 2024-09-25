'''
Author: Sasha
Date: 2023-02-27 14:46:55
LastEditors: Sasha
LastEditTime: 2023-04-03 17:30:22
Description: 
FilePath: /FederatedLearning/FedL/federatedml/linear_model/logistic_regression/linr_Logistic_guest.py
'''
import numpy as np
from FedL.federatedml.data_factory import vertical_split_data
from FedL.federatedml.linear_model.linear_regression.conf import config
from FedL.federatedml.model_base import ModelBase
from FedL.database.fed_lmdb import feddb_guest
from FedL.federatedml.PSI.psi_service import PSIService
from FedL.federatedml.util.learn_rate_decay import cosine_decay_with_warmup

class LinrLogisticGuest(ModelBase):
    def __init__(self,train_x,label,config) -> None:
        super().__init__(fed_lmdb=feddb_guest)
        self.config = config
        self.X = train_x
        self.y = label
        self.weights = np.zeros(self.X.shape[1])
        
    def get_public_key(self):
        self.public_key = self.get("public_key_guest")
    
    def compute_u(self):
        return self.X.dot(self.weights)
    
    def cumputer_d_L(self):
        encrypted_u_a = self.get("encrypted_u_a_%s" % self.n)
        encrypted_u_a_square = self.get("encrypted_u_a_square_%s" % self.n)
        #计算自己的部分
        u_guest = self.compute_u()        
        encrypted_d = 0.25*encrypted_u_a + (0.25*u_guest - 0.5 * self.y)
        encrypted_L = np.sum(np.log(2) -0.5*self.y*(encrypted_u_a+u_guest) + 0.125*(u_guest**2 + encrypted_u_a_square + 2*(encrypted_u_a*u_guest)))
        # L_b = 0.5 * np.sum(z_b_square) + 0.5 * self.config["lambda"] * np.sum(self.weights**2)
        # L_ab = np.sum(encrypted_u_a * z_b)
        # encrypted_L = encrypted_L_a + L_b + L_ab
        return encrypted_d,encrypted_L
        # 计算加密梯度
    def compute_encrypted_dL_b(self,encrypted_d):
        encrypted_dL_b = self.X.T.dot(encrypted_d) + self.config["lambda"] * self.weights
        return encrypted_dL_b

    def run(self,n):
        self.n = n
        ##########################################################################################################
        self.encrypted_d,encrypted_L = self.cumputer_d_L()
        self.remote(self.encrypted_d,"encrypted_d_%s" % self.n,"50053")
        self.remote(encrypted_L,"encrypted_L_%s" % self.n,"50051")
        #############################################计算加密梯度[[dL_b]],mask之后发给C方#############################################################
        encrypted_dL_b = self.compute_encrypted_dL_b(self.encrypted_d)
        self.mask = np.random.rand(len(encrypted_dL_b))
        encrypted_masked_dL_b = encrypted_dL_b + self.mask
        self.remote(encrypted_masked_dL_b,"encrypted_masked_dL_b_%s" % self.n,"50051")
        #############################################获取解密后的梯度，解mask，模型更新#############################################################
        masked_dL_b = self.get("masked_dL_b_%s" % self.n)
        dL_b = masked_dL_b - self.mask
        self.weights = self.weights - cosine_decay_with_warmup(self.n,config['epoch'],lr_max=config['lr']) * dL_b / len(self.X)
        print("guest weights {} : {}".format(self.n,self.weights))
        # self.fed_lmdb.delete_list(["%s%s" % ( x,self.n) for x in ["encrypted_u_a_","encrypted_L_a_","masked_dL_b_"]])
        # self.remote(b'success',"state_guest_%s" % self.n,'50051')
    
    
    def predict(self,X_test):
        u_b = X_test.dot(self.weights)
        encrypted_u_b = np.array(self.public_key.encrypt_list(u_b))
        self.remote(encrypted_u_b,"encrypted_u_b",'50051')
        pred = self.get("pred")
        print('pred:',pred)
            
if __name__ == "__main__":
    try:
        XA_train,XB_train,XA_test,XB_test,y_train,y_test = vertical_split_data()
        #隐私求交
        # ids = [x for x in range(len(XA_train)-10)]
        # psi_service = PSIService(ids = ids)
        # common_ids = psi_service.run()
        # XB_train ,y_train= XB_train[common_ids],y_train[common_ids]
        
        line_guest = LinrLogisticGuest(XB_train,y_train,config)
        line_guest.get_public_key()
        for n in range(config['epoch']):
            line_guest.run(n)
            
        line_guest.predict(XB_test)
    finally:
        print(" drop finally")
        line_guest.fed_lmdb.drop()