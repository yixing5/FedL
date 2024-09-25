'''
Author: Sasha
Date: 2023-03-24 15:18:44
LastEditors: Sasha
LastEditTime: 2023-03-24 15:18:45
Description: 
FilePath: /FederatedLearning/FedL/util/utils.py
'''

import os
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
