'''
Author: Sasha
Date: 2023-02-24 17:47:28
LastEditors: Sasha
LastEditTime: 2023-04-04 10:27:21
Description: 
FilePath: /FederatedLearning/FedL/federatedml/data_factory.py
'''
import numpy as np
from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from collections import namedtuple

import pandas as pd

def split_data():
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test  = train_test_split(dataset.data,dataset.target,test_size=0.1,random_state=2023)
    # 堆叠一列1，把偏置合并到w中
    X_train = np.column_stack((X_train,np.ones(len(X_train))))
    X_test = np.column_stack((X_test,np.ones(len(X_test))))
    # 打印数据形状
    for temp in [X_train, X_test, y_train, y_test]:
        print(temp.shape)
    idx_A = list(range(6))
    idx_B = list(range(6,11))
    XA_train,XB_train = X_train[:,idx_A], X_train[:,idx_B]
    XA_test,XB_test = X_test[:,idx_A], X_test[:,idx_B]
    return XA_train,XB_train,XA_test,XB_test,y_train,y_test

    
def vertical_split_data():
    # 加载数据
    breast = load_breast_cancer()
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.1,random_state=2023)
    y_test = np.where(y_test==1,y_test,-1)
    for temp in [X_train, X_test, y_train, y_test]:
        print(temp.shape)
    # 数据标准化
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    
    A_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    B_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    XA_train = X_train[:, A_idx]  
    XB_train = X_train[:, B_idx]  
    XB_train = np.c_[np.ones(X_train.shape[0]), XB_train]
    
    XA_test = X_test[:, A_idx]
    XB_test = X_test[:, B_idx]
    XB_test = np.c_[np.ones(XB_test.shape[0]), XB_test]
    return XA_train,XB_train,XA_test,XB_test,y_train,y_test

# mul_task
def multi_task_split_data():
    VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', "field_id",'input_length','feature_sizes'])
    
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    data = pd.read_csv('/data/sasha/FederatedLearning/FedL/federatedml/data/census-income.sample', header=None, names=column_names)
    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                    'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                    'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                    'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'vet_question']
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    
    
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        
    feature_columns = [VarLenSparseFeat(name=feat,field_id=i ,input_length=1, feature_sizes =  data[feat].max() + 1) for i,feat in enumerate(sparse_features)]

    train = data[sparse_features]
    X_train, X_test, y1_train, y1_test,y2_train, y2_test  = train_test_split(train,data['label_income'],data['label_marital'],test_size=0.1,random_state=2023)
    feature_columns_A = feature_columns[:20]
    feature_columns_B = feature_columns[20:]
    XA_train = X_train.iloc[:,:20]
    XB_train = X_train.iloc[:,20:]
    XA_test = X_test.iloc[:,:20]
    XB_test = X_test.iloc[:,20:]
    return feature_columns_A,feature_columns_B,XA_train,XB_train,XA_test,XB_test, y1_train, y1_test,y2_train, y2_test


if __name__ == '__main__':
    feature_columns_A,feature_columns_B,XA_train,XB_train,XA_test,XB_test, y1_train, y1_test,y2_train, y2_test = multi_task_split_data()