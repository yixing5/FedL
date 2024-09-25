'''
Author: Sasha
Date: 2023-03-08 09:14:31
LastEditors: Sasha
LastEditTime: 2023-03-24 12:19:32
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/conf.py
'''
config = {
    'lambda':0.1, #正则项系数
    'decay':0.0001,
    'epoch':2000, # 训练轮数
    'lr_max':0.01,
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
        "layer_dims" : 10,
        'lr':0.1,    # 学习率
    }
    
}
