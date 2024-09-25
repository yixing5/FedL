'''
Author: Sasha
Date: 2023-03-01 17:22:35
LastEditors: Sasha
LastEditTime: 2023-03-06 15:50:01
Description: 
FilePath: /FederatedLearning/FedL/federatedml/PSI/psi_service.py
'''
from FedL.federatedml.model_base import ModelBase
from Cryptodome.PublicKey import RSA
from FedL.federatedml.PSI.util import hash_bignumber
import gmpy2
from FedL.database.fed_lmdb import feddb_guest

class PSIService(ModelBase):
    def __init__(self,ids) -> None:
        super().__init__(fed_lmdb=feddb_guest)
        self.ids = ids
        self.pk,self.sk = self.gen_key()
        
    def gen_key(self):
        key = RSA.generate(1024)
        pk = (key.n,key.e)
        sk = (key.n,key.d)
        return pk,sk
    
    def deblind(self,hash_arr_blind):
        deblind_hash_arr = []
        for item in hash_arr_blind:
            de_blind_number = gmpy2.powmod(item,self.sk[1],self.sk[0])
            deblind_hash_arr.append(de_blind_number)
        return deblind_hash_arr
    
    def enc_and_hash(self):
        hash_server_list = []
        for item in self.ids:
            hash_num = hash_bignumber(item)
            c_hash_num = gmpy2.powmod(hash_num,self.sk[1],self.sk[0])
            hash_server_list.append(hash_bignumber(c_hash_num))
        return hash_server_list
    
    def element_to_id(self,hash_deblind,common_element):
        common_element = set(common_element)
        res = []
        for ele ,id in zip(hash_deblind,self.ids):
            if ele in common_element:
                res.append(id)
        return res
    def run(self):
        self.remote(self.pk,'RSA_public_key','50053')
        
        ids_blind = self.get('ids_blind')
    
        ids_deblind = self.deblind(ids_blind)
        ids_enc_hash = self.enc_and_hash()
        
        self.remote(ids_deblind,'ids_deblind','50053')
        self.remote(ids_enc_hash,'ids_enc_hash','50053')
        
        common_element = self.get('common_element')
        common_idx = self.element_to_id(ids_enc_hash,common_element)
        return common_idx
        
if __name__ =="__main__":
    psi_service = PSIService(ids = [1,2,3,4,5,6])
    psi_service.run()
        