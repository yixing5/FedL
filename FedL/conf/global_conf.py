'''
Author: Sasha
Date: 2023-03-13 12:00:26
LastEditors: Sasha
LastEditTime: 2023-04-04 14:25:02
Description: 
FilePath: /FederatedLearning/FedL/conf/global_conf.py
'''
from datetime import datetime

class ServiceConf:
    ip_center = "localhost"
    # ip_guest = "10.30.20.116"
    # ip_host = "10.30.20.10"
    ip_guest = "localhost"
    ip_host = "localhost"
    port_center = "50051"
    port_guest = "50052"
    port_host = "50053"
    

class Path:
    home = "/data/sasha/FederatedLearning/FedL"
    log_path = home + "/log"
    
    dataset_path_center = home + "/database/feddb_center"
    dataset_path_guest = home + "/database/feddb_guest"
    dataset_path_host = home + "/database/feddb_host"
    
    model_path = home + "/federatedml/dnn/model_saved"
    
class ModelInfo:
    version = datetime.today().strftime("%Y-%m-%d")

