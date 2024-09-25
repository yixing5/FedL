'''
Author: Sasha
Date: 2023-03-23 11:30:00
LastEditors: Sasha
LastEditTime: 2023-03-23 11:39:15
Description: 
FilePath: /FederatedLearning/FedL/federatedml/util/optimizer.py
'''
class Adadelta():
    """Adadelta algorithm (https://arxiv.org/abs/1212.5701)"""
    def __init__(self, lr=1.0, decay=0.9, epsilon=1e-8, weight_decay=0.0,):
        self._epsilon = epsilon
        self._rho = decay
        self._rms = 0  # running average of square gradient
        self._delta = 0  # running average of delta

    def compute_step(self, grads):
        self._rms += (1 - self._rho) * (grads ** 2 - self._rms)
        std = (self._delta + self._epsilon) ** 0.5
        delta = grads * (std / (self._rms + self._epsilon) ** 0.5)
        step = - 1 * delta
        self._delta += (1 - self._rho) * (delta ** 2 - self._delta)
        # print(step)
        return step

class Momentum():
    """accumulation = momentum * accumulation + gradient
    variable -= learning_rate * accumulation
    """
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self._momentum = momentum
        self._acc = 0

    def compute_step(self, grads):
        self._acc = self._momentum * self._acc + grads
        step = -self.lr * self._acc
        return step
    
class Adam():

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        self.lr = lr
        self._b1 = beta1
        self._b2 = beta2
        self._epsilon = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

    def compute_step(self, grads):
        self._t += 1
        
        self._m = self._b1*self._m  + (1-self._b1) * grads
        self._v = self._b2*self._v  + (1-self._b2) * grads ** 2

        # self._m += (1.0 - self._b1) * (grads - self._m)
        # self._v += (1.0 - self._b2) * (grads ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._epsilon)
        return step

# config = {
#     'lambda':0.1, #正则项系数
#     'lr':0.01,    # 学习率
#     'decay':0.1,
#     'epoch':500, # 训练轮数
# }
# opt = Adam(lr= config['lr'])
