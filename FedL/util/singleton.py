'''
Author: Sasha
Date: 2023-03-13 17:18:02
LastEditors: Sasha
LastEditTime: 2023-03-13 17:18:03
Description: 
FilePath: /FederatedLearning/FedL/util/singleton.py
'''
def Singleton(cls):
    _instance = {}
    def _singleton(*args, **kargs):
        if 'path' in kargs and kargs['path'] not in _instance:
            _instance[kargs['path']] = cls(*args, **kargs)
        return _instance[kargs['path']]
    return _singleton