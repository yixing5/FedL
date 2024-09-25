'''
Author: Sasha
Date: 2023-03-06 14:13:00
LastEditors: Sasha
LastEditTime: 2023-03-06 14:16:03
Description: 
FilePath: /FederatedLearning/FedL/federatedml/secureprotol/pool.py
'''
from multiprocessing import Pool


class MangerPool():
    def __init__(self,n=12) -> None:
        self.p = Pool(n)     
    
    def task(self,func,arr):
        res_l=[]
        for i in arr:
            res=self.p.apply_async(func,args=(i,))
            res_l.append(res)
        self.p.close()
        self.p.join()
        return [res.get() for res in res_l]

pool = MangerPool()