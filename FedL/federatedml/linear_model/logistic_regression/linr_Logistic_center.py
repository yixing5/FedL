'''
Author: Sasha
Date: 2023-02-27 14:44:50
LastEditors: Sasha
LastEditTime: 2023-03-08 10:26:49
Description: 
FilePath: /FederatedLearning/FedL/federatedml/linear_model/logistic_regression/linr_Logistic_center.py
'''
from FedL.federatedml.model_base import ModelBase
import numpy as np
from FedL.database.fed_lmdb import feddb_center
from FedL.federatedml.linear_model.linear_regression.conf import config
from federatedml.secureprotol.encrypt import PaillierEncrypt

class LinrLogisticCenter(ModelBase):
    def __init__(self) -> None:
        super().__init__(fed_lmdb=feddb_center)
        cipher = PaillierEncrypt()
        cipher.generate_key(1024)
        self.public_key ,self.private_key = cipher.get_public_key(),cipher.get_privacy_key()        
        self.loss_history = []     
    
    def send_public_key(self):
        #############################################分发公钥#############################################################
        self.remote(self.public_key,"public_key_guest","50052")
        self.remote(self.public_key,"public_key_host","50053")  
        
    def run(self,n):
        # self.fed_lmdb.delete_list(["%s%s" % ( x,n-1) for x in ["encrypted_L_","state_guest_","state_host_"]])
        #############################################解密[[L]]、[[masked_dL_a]],[[masked_dL_b]]，分别发送给A、B#############################################################
        encrypted_L = self.get("encrypted_L_%s" % n)
        encrypted_masked_dL_b = self.get("encrypted_masked_dL_b_%s" % n)
        encrypted_masked_dL_a = self.get("encrypted_masked_dL_a_%s" % n)
        L = self.private_key.decrypt(encrypted_L)
        
        self.loss_history.append(L)
        masked_dL_b = np.array(self.private_key.decrypt_list(encrypted_masked_dL_b))
        masked_dL_a = np.array(self.private_key.decrypt_list(encrypted_masked_dL_a))
        self.remote(masked_dL_a,"masked_dL_a_%s" % n,"50053")
        self.remote(masked_dL_b,"masked_dL_b_%s" % n,"50052")
        ###########################################确认一个epoch训练完毕###################################
        # self.get("state_guest_%s" % n)
        # self.get("state_host_%s" % n)
        print("*"*40,"loss : ",L/512,"*"*40)
            
    def predict(self):
        encrypted_u_a = self.get("encrypted_u_a")
        encrypted_u_b = self.get("encrypted_u_b")
        encrypted_u = encrypted_u_a +encrypted_u_b
        u =  np.array(self.private_key.decrypt_list(encrypted_u))
        self.remote(u,"pred" ,"50052")
        self.remote(u,"pred" ,"50053")
        
if __name__ == "__main__":
    # fed_lmdb.reset()
    try:
        line_center = LinrLogisticCenter()
        line_center.fed_lmdb.drop()
        line_center.send_public_key()
        
        for n in range(config['epoch']):
            line_center.run(n)
        line_center.predict()
    finally:
        print("drop finally")
        line_center.fed_lmdb.drop()