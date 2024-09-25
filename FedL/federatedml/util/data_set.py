'''
Author: Sasha
Date: 2023-04-03 11:30:34
LastEditors: Sasha
LastEditTime: 2023-04-03 11:37:24
Description: 
FilePath: /FederatedLearning/FedL/federatedml/util/data_set.py
'''
import tensorflow as tf

# %%
class DateSetFed(object):
    def __init__(self,batch_size) -> None:
        self.batch_size = batch_size
    def get_dateset_from_dataframe(self,df,label):
        ds = tf.data.Dataset.from_tensor_slices((df.to_dict("list"),label.values))
        # ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(self.batch_size)
        ds = ds.repeat(-1)
        return ds
    
    # def get_next(self,df,label):
    #     ds = self.get_dateset_from_dataframe(df,label)
    #     return ds.make_one_shot_iterator().get_next()
    