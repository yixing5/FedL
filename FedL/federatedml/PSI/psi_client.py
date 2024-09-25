'''
Author: Sasha
Date: 2023-03-01 17:20:12
LastEditors: Sasha
LastEditTime: 2023-03-06 15:49:53
Description: 
FilePath: /FederatedLearning/FedL/federatedml/PSI/psi_client.py
'''


from FedL.federatedml.model_base import ModelBase
from FedL.federatedml.PSI.util import hash_bignumber
from Cryptodome.PublicKey import RSA
import hashlib
import binascii 
import gmpy2
import os

from FedL.database.fed_lmdb import feddb_host
        
class PSIClient(ModelBase):
    def __init__(self,ids) -> None:
        super().__init__(fed_lmdb=feddb_host)
        self.ids = ids
        self.rand_bits = 128
        
    def blind(self):
        msg_hash_number_blind = []
        rand_private_list = [] 
        for item in self.ids:
            hash_num = hash_bignumber(item)
            hash_num = hash_num % self.pk[0]
            ra = int(binascii.hexlify(os.urandom(self.rand_bits)),16)
            cipher_ra = gmpy2.powmod(ra,self.pk[1],self.pk[0])
            rand_private_list.append(ra)
            msg_hash_number_blind.append(hash_num*cipher_ra)
        
        return msg_hash_number_blind, rand_private_list
    
    def hash_deblind(self,deblind_hash_arr, rand_list):
        db_client = []
        for item,ra in zip(deblind_hash_arr,rand_list):
            ra_inv = gmpy2.invert(ra,self.pk[0]) # ra*ra_inv == 1 mod n 
            db_client.append(hash_bignumber((item * ra_inv) % self.pk[0]))
        return db_client
    
    def get_common_element(self,ids_enc_hash,hash_deblind):
        return list(set(ids_enc_hash) & set(hash_deblind))
    
    def element_to_id(self,hash_deblind,common_element):
        common_element = set(common_element)
        res = []
        for ele ,id in zip(hash_deblind,self.ids):
            if ele in common_element:
                res.append(id)
        return res
        
    def run(self):
        self.pk = self.get('RSA_public_key')
        ids_blind, rand_private_list = self.blind()
        self.remote(ids_blind,'ids_blind','50052')
        
        ids_deblind = self.get('ids_deblind')
        ids_enc_hash = self.get('ids_enc_hash')
        
        hash_deblind = self.hash_deblind(ids_deblind,rand_private_list)
        
        common_element = self.get_common_element(ids_enc_hash,hash_deblind)
        # print("common_element",common_element)
        self.remote(common_element,'common_element','50052')
        
        common_ids = self.element_to_id(hash_deblind,common_element)
        
        return common_ids
         
if __name__ =="__main__":
    psi_client = PSIClient(ids = [8,7,3,4,5,6])    
    psi_client.run()
        