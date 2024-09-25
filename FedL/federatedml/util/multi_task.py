'''
Author: Sasha
Date: 2023-03-30 12:14:12
LastEditors: Sasha
LastEditTime: 2023-05-12 09:50:39
Description: 
FilePath: /FederatedLearning/FedL/federatedml/util/multi_task.py
'''
from mpire import WorkerPool
from multiprocessing import cpu_count 

class MultiTask():
    def __init__(self,n_jobs=None) -> None:
        self.n_jobs = n_jobs or max(1,cpu_count() // 2)
    
    def map(self,func,parms_list):
        with WorkerPool(n_jobs=self.n_jobs) as pool:
            results = pool.map(func,parms_list)
        return results

multi_task = MultiTask(12)

# import ray
# # 启动Ray.
# from multiprocessing import cpu_count 
# # 阻塞等待4个任务完成，超时时间为2.5s
# class MultiTask():
#     def __init__(self,n_jobs=None) -> None:
#         self.n_jobs = n_jobs or max(1,cpu_count() // 2)

#     def map(self,func,parms_list):
#         results = [func.remote(i) for i in parms_list]
#         ready_ids, remaining_ids = ray.wait(results, num_returns=len(parms_list), timeout=2500)

#         # with WorkerPool(n_jobs=self.n_jobs) as pool:
#         #     results = pool.map(func,parms_list)
#         return ready_ids

# multi_task = MultiTask(12)