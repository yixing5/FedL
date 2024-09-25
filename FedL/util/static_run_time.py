'''
Author: Sasha
Date: 2023-03-30 18:33:30
LastEditors: Sasha
LastEditTime: 2023-03-30 18:33:31
Description: 
FilePath: /FederatedLearning/FedL/util/static_run_time.py
'''
import datetime
def time_me(func):
    '''
    @summary: cal the time of the fucntion
    @param : None
    @return: return the res of the func
    '''
 
    def wrapper(*args, **kw):
        start_time = datetime.datetime.now()
        res = func(*args, **kw)
        over_time = datetime.datetime.now()
        print ('{0} run time is {1}'.format(func.__name__, (over_time - start_time).total_seconds()))
        return res
 
    return wrapper