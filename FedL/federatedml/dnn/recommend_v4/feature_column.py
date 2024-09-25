'''
Author: Sasha
Date: 2022-07-28 10:45:34
LastEditors: Sasha
LastEditTime: 2023-04-11 09:17:05
Description: 
FilePath: /FederatedLearning/FedL/federatedml/dnn/recommend_v4/feature_column.py
'''
from collections import namedtuple
from feature.data_clean import DataClean
from conf.conf import mpath
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', "field_id",'input_length','feature_sizes'])
class FeatureColumn(object):
    def __init__(self,fields_path=mpath.fields_path):
        self.fields_path = fields_path
    def get_feature_columns(self,df):
        feature_columns = []
        with open(self.fields_path) as f:
            for item in f:
                item = item.strip()
                if item:
                    name,field_id,input_length,feature_sizes = item.split('\t')
                    if name not in df.columns:continue
                    feature_columns.append(VarLenSparseFeat(name=str(name),field_id=int(field_id) ,input_length=int(input_length), feature_sizes = int(feature_sizes)))
        return feature_columns
