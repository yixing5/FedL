'''
Author: Sasha
Date: 2023-02-23 16:22:24
LastEditors: Sasha
LastEditTime: 2023-03-13 17:18:26
Description: 
FilePath: /FederatedLearning/FedL/database/fed_lmdb.py
'''
from FedL.conf.global_conf import Path
import lmdb
from FedL.util.singleton import Singleton

@Singleton
class FedLmdb():
    def __init__(self, path):
        self.path= path
        self.db = lmdb.open(self.path,map_size=int(1e9))
    
    def drop(self):
        print('ABOUT TO DELETE DB '+str(self.path)+'!!!!')
        with self.db.begin(write=True) as in_txn:
            db = self.db.open_db()
            in_txn.drop(db)
            print(in_txn.stat())
            
    def delete(self,key):
        with self.db.begin(write=True) as txn:
            if txn.get(key.encode("utf8")):
                txn.delete(key.encode("utf8"))
    def delete_list(self,arr):
        with self.db.begin(write=True) as txn:
            for key in arr:
                if txn.get(key.encode("utf8")):
                    txn.delete(key.encode("utf8"))
    def reset(self):
        self.drop()
        # self.db = lmdb.open(self.path)
        
feddb_center = FedLmdb(path=Path.dataset_path_center)
feddb_guest = FedLmdb(path=Path.dataset_path_guest)
feddb_host = FedLmdb(path=Path.dataset_path_host)
