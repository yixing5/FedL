'''
Author: Sasha
Date: 2023-03-07 10:08:30
LastEditors: Sasha
LastEditTime: 2023-03-23 14:59:12
Description: 
FilePath: /FederatedLearning/FedL/federatedml/util/learn_rate_decay.py
'''
import numpy as np 

class GradDecay():
    def __init__(self,lr,decay):
        self.lr = lr
        self.decay = decay
        self.step = 0
        
    def compute_step(self):
        self.step += 1
        return self.lr * 1.0 / (1.0 + self.decay * self.step)


class CosineDecayWithWarmup():
    def __init__(self,total_steps,warmup_learning_rate=0.001,warmup_steps=100,lr_min=0.0001,lr_max=0.1):
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step = 0

    def compute_step(self):
        self.step += 1
        
        learning_rate = 0
        #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
        if self.step <= self.warmup_steps:
            slope = (self.lr_max - self.warmup_learning_rate) / self.warmup_steps
            learning_rate = slope * self.step + self.warmup_learning_rate
        else:
            learning_rate = self.lr_min + 0.5 * (self.lr_max-self.lr_min) * (1 + np.cos(np.pi *(self.step - self.warmup_steps ) / float(self.total_steps - self.warmup_steps )))
        return learning_rate