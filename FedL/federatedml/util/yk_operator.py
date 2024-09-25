'''
Author: Sasha
Date: 2023-03-30 14:18:41
LastEditors: Sasha
LastEditTime: 2023-03-30 16:54:59
Description: 
FilePath: /FederatedLearning/FedL/federatedml/util/yk_operator.py
'''
from FedL.federatedml.util.multi_task import multi_task
import numpy as np

def dot(a,b):
    b = b if len(b.shape)==2 else np.expand_dims(b,axis=1)
    a = a if len(a.shape)==2 else np.expand_dims(a,axis=0)
    results = multi_task.map(np.dot,[(a[i,:],b[:,j]) for i in range(a.shape[0]) for j in range(b.shape[1])] )
    return np.squeeze(np.reshape(results,(a.shape[0],b.shape[1])))
