'''
Author: Sasha
Date: 2023-03-21 14:27:47
LastEditors: Sasha
LastEditTime: 2023-03-21 14:32:12
Description: 
FilePath: /FederatedLearning/FedL/federatedml/util/secure_random.py
'''

import numpy as np
import random
def generate_secure_noise(shape,upper_bound=2 ** 10,lower_bound=-2 ** 10): # 下界
    size = np.prod(shape)
    return np.reshape([random.SystemRandom().uniform(lower_bound, upper_bound) for _ in range(size)],shape)