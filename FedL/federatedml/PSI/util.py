'''
Author: Sasha
Date: 2023-03-01 17:25:53
LastEditors: Sasha
LastEditTime: 2023-03-01 17:25:54
Description: 
FilePath: /FederatedLearning/FedL/federatedml/PSI/util.py
'''
import hashlib
def hash_bignumber(num,method='sha1'):
    '''
        num: an integer 
    '''
    if method == 'sha1':
        hash_obj = hashlib.sha1(str(num).encode('utf-8'))
        digest_hex = hash_obj.hexdigest()
        return int(digest_hex,16)