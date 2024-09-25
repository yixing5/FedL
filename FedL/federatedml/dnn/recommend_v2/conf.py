'''
Author: Sasha
Date: 2023-03-08 09:14:31
LastEditors: Sasha
LastEditTime: 2023-04-04 10:19:31
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend/conf.py
'''
config = {
    'lambda':0.1, #正则项系数
    'decay':0.0001,
    'epoch':2000, # 训练轮数
    'lr_max':0.01,
    "experts_units":4,
    "experts_num":3,
    "task_num":2,
    "batch_size":32,
    "embed_dims":4,
    "bottom_conf" : {
                    "layers_dims" : [8,16],
                    'lr':0.1,    # 学习率
                    'activation':'relu',
    },
    "top_conf" : {
                "layers_dims" : [16,8,1],
                'lr':0.1,    # 学习率
                'activation':'relu',
    },
    
    'interactive_conf':{
        'lr':0.1,    # 学习率
    }
    
}
