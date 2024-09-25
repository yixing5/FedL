'''
Author: Sasha
Date: 2023-11-14 16:11:32
LastEditors: Sasha
LastEditTime: 2024-07-25 08:14:22
Description: 
FilePath: /project_learn/Machine_learning/MyAutoML/trainer/base_trainer.py
'''
from pandas import DataFrame
import copy
from collections import defaultdict
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from  toad.transform import Combiner
import json
import toad
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import log_loss
"""
预处理顺序:
1.去掉只有一个取值的列

"""


#######################################################################################模型区域 #######################################################################################  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier 
from xgboost import XGBClassifier,plot_importance
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from toad.metrics import KS,AUC
from sklearn.metrics import classification_report
import optuna
from sklearn.metrics import auc, roc_auc_score
import xgboost as xgb

class BaseModel():
    def __init__(self):
        self.save_dir = "/home/jovyan/work/data/sasha/loan_predict/model/output"
        self.need_select_feature = False
    def fit(self):
        pass
          
    def objective(self):
        raise NotImplementedError("Subclasses must implement the 'objective' method.")
    
    def predict(self):
        pass
    def plot_importance(self,*args,**kwargs):
        pass
        
    def predict_and_report(self,model,X,Y,types="train"):
        pred_porba = model.predict_proba(X)[:,1]
        y_pred = model.predict(X)
        ks = KS(pred_porba,Y)
        auc = AUC(pred_porba,Y)
        print(f"{types} ks : {ks}")
        print(f"{types} auc : {auc}")
        print(classification_report(Y,y_pred))
        return ks,auc
        
    def report(self,model,X_train,y_train,X_test,y_test):
        self.predict_and_report(model,X_train,y_train)
        self.predict_and_report(model,X_test,y_test,types="test")

    def save(self):
        pass
    
class LogisticRegressionModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.need_select_feature = True

    def objective(self,trial,X,y):
        # 设置逻辑回归的超参数搜索范围
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        # 创建逻辑回归模型

        # 预测测试集
        # 评估模型性能
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = LogisticRegression(C=C, random_state=42)
            # 训练模型
            model.fit(X_train, y_train)
            # 模型预测
            pred_proba = model.predict_proba(X_test)[:, 1]
            # 优化指标logloss最小
            cv_scores[idx] = roc_auc_score(y_test, pred_proba)

        return np.mean(cv_scores)
    
    def fit(self, X_train, y_train, X_val, y_val):
        # 创建 Optuna 优化对象
        study = optuna.create_study(direction='maximize')
        # 执行优化过程
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=30)
         # 输出最优超参数
        best_params = study.best_params
        print("Best hyperparameters:")
        print(best_params)
        self.best_params = best_params

        # 使用最优超参数重新训练模型
        self.model =  LogisticRegression(C=best_params["C"], random_state=42)
        self.model.fit(X_train, y_train)
        
    def plot_importance(self):
        df = pd.DataFrame(zip(self.columns,self.model.coef_[0]),columns=['columns','weight']) 
        print(df.sort_values("weight",ascending=False).head(60),'plot_importance')
        
class KNNModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.need_select_feature = True
        
    def objective(self, trial,X,y):
        # 设置 KNN 的超参数搜索范围
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 5)

        # 创建 KNN 模型
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # LGBM建模
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
            # 训练模型
            model.fit(X_train, y_train)
            # 模型预测
            pred_proba = model.predict_proba(X_test)[:, 1]
            # 优化指标logloss最小
            cv_scores[idx] = roc_auc_score(y_test, pred_proba)

        return np.mean(cv_scores)
    
    def fit(self, X_train, y_train, X_val, y_val):
        # 创建 Optuna 优化对象
        study = optuna.create_study(direction='maximize')
        # 执行优化过程
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=30)

        # 输出最优超参数
        best_params = study.best_params
        print("Best hyperparameters:")
        print(best_params)
        self.best_params = best_params

        # 使用最优超参数重新训练模型
        self.model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], 
                                          weights=best_params['weights'], 
                                          p=best_params['p'])
        self.model.fit(X_train, y_train)
    
class XGBoostModel(BaseModel):
    def __init__(self,train_way= '',default_params=None) -> None:
        super().__init__()
        self.model = None
        self.default_params = default_params
        self.train_way = train_way
        
    
    def objective(self, trial, X,y):
        # 设置 XGBoost 的超参数搜索范围
        # The above code is not doing anything. It appears to be a placeholder or incomplete code
        # snippet.
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # LGBM建模
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, 
                y_train, 
                eval_set=[(X_test, y_test)], 
                early_stopping_rounds=10,
                # callbacks = [xgb.early_stopping(stopping_rounds=20)] ,
                verbose=False
            )
            # 模型预测
            pred_proba = model.predict_proba(X_test)[:, 1]
            # 优化指标logloss最小
            cv_scores[idx] = roc_auc_score(y_test, pred_proba)

        return np.mean(cv_scores)
    
    def fit(self, X_train, y_train, X_val, y_val):
        if self.train_way == 'sample':
            best_params = self.default_params
        else:
            # 创建 Optuna 优化对象
            study = optuna.create_study(direction='maximize')

            # 执行优化过程
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=30)

            # 输出最优超参数
            best_params = study.best_params
            print("Best hyperparameters:")
            print(best_params)
        self.best_params = best_params

        # 使用最优超参数重新训练模型
        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
       
class RandomForestModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
    
    def objective(self, trial,X,y):
        # 设置随机森林的超参数搜索范围
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 6)
        min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
        min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # LGBM建模
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            # 模型预测
            pred_proba = model.predict_proba(X_test)[:, 1]
            # 优化指标logloss最小
            cv_scores[idx] = roc_auc_score(y_test, pred_proba)

        return np.mean(cv_scores)
    
    def fit(self, X_train, y_train,X_val, y_val):
        # 创建 Optuna 优化对象
        study = optuna.create_study(direction='maximize')

        # 执行优化过程
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=30)
        

        # 输出最优超参数
        best_params = study.best_params
        print("Best hyperparameters:")
        print(best_params)
        self.best_params = best_params

        # 使用最优超参数重新训练模型
        self.model = RandomForestClassifier(**best_params)
        self.model.fit(X_train, y_train)
from lightgbm import LGBMClassifier,early_stopping

class LightGBMModel(BaseModel):
    def __init__(self,train_way= '',default_params=None) -> None:
        super().__init__()
        self.model = None
        self.default_params = default_params
        self.train_way = train_way
        
    def objective(self,trial, X, y):
        # 参数网格
        param_grid = {
            "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
            "random_state": 2021,
        }
        # 5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # LGBM建模
            model = LGBMClassifier(objective="binary", **param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                # eval_metric="binary_logloss",
                # early_stopping_rounds=100,
                callbacks = [early_stopping(stopping_rounds=20)] 
                # callbacks=[
                #     LightGBMPruningCallback(trial, "binary_logloss")
                # ],
            )
            # 模型预测
            # preds = model.predict_proba(X_test)
            # # 优化指标logloss最小
            # cv_scores[idx] = log_loss(y_test, preds)
            # 模型预测
            pred_proba = model.predict_proba(X_test)[:, 1]
            # 优化指标logloss最小
            cv_scores[idx] = roc_auc_score(y_test, pred_proba)

        return np.mean(cv_scores)
    
    def fit(self, X_train, y_train, X_val, y_val,):
        if self.train_way == 'sample':
            best_params = self.default_params
        else:
            # 创建 Optuna 优化对象
            study = optuna.create_study(direction='maximize')#minimize
            # 执行优化过程
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=30)

            # 输出最优超参数
            best_params = study.best_params
            print("Best hyperparameters:")
            print(best_params)
        self.best_params = best_params
        # 使用最优超参数重新训练模型
        self.model = LGBMClassifier(**best_params, random_state=42)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

class CatBoostModel(BaseModel):
    def __init__(self,train_way= '',default_params=None) -> None:
        super().__init__()
        self.model = None
        self.default_params = default_params
        self.train_way = train_way
    
    def objective(self, trial, X,y):
        # 设置 CatBoost 的超参数搜索范围
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 1, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'random_strength': trial.suggest_uniform('random_strength', 0.1, 10),
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # LGBM建模
            model = CatBoostClassifier(**params, random_state=42, silent=True)

        # 训练模型
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, verbose=False)
            # 模型预测
            # 模型预测
            pred_proba = model.predict_proba(X_test)[:, 1]
            # 优化指标logloss最小
            cv_scores[idx] = roc_auc_score(y_test, pred_proba)

        return np.mean(cv_scores)
    
    def fit(self, X_train, y_train, X_val, y_val):
        if self.train_way == 'sample':
            best_params = self.default_params
        else:
            # 创建 Optuna 优化对象
            study = optuna.create_study(direction='maximize')

            # 执行优化过程
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=30)

            # 输出最优超参数
            best_params = study.best_params
            print("Best hyperparameters:")
            print(best_params)
        self.best_params = best_params
        # 使用最优超参数重新训练模型
        self.model = CatBoostClassifier(**best_params, random_state=42, silent=True)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10, verbose=False)

        
class StackingModel(BaseModel):
    def __init__(self,random_state = 43):
        self.res = []
        self.clf0 = KNeighborsClassifier(**{'weights': 'uniform', 'n_jobs': 32})
        self.clf1 = KNeighborsClassifier(**{'weights': 'distance', 'n_jobs': 32})
        self.clf2 = LGBMClassifier(**{
                                        'learning_rate': 0.05,
                                        'eval_metric':'AUC',
                                        'extra_trees': True, 
                                        'num_threads': 32, 
                                        'objective': 'binary', 
                                        'metric': 'binary_logloss',
                                        'seed': 0,
                                        # "early_stopping_rounds":10,
                                        # "eval_set":[(X_test, y_test)],
                                        "random_state":42}
                                        )

        self.clf3 = LGBMClassifier(**{
                                        'learning_rate': 0.05,
                                        'eval_metric':'AUC',
                                        'num_threads': 32, 
                                        'objective': 'binary', 
                                        'metric': 'binary_logloss',
                                        'seed': 0,
                                        # "early_stopping_rounds":10,
                                        # "eval_set":[(X_test, y_test)],
                                        }
                                )
        self.clf4 = RandomForestClassifier(**{
                                        'n_estimators': 300, 
                                        'max_leaf_nodes': 15000, 
                                        'n_jobs': -1, 
                                        'random_state': 0,
                                        'bootstrap': True, 
                                        'criterion': 'gini',
                                        'verbose': False, 
                                        "class_weight" : 'balanced'
                                        })        
        
        self.clf5 = RandomForestClassifier(**{
                                        'n_estimators': 300, 
                                        'max_leaf_nodes': 15000, 
                                        'n_jobs': -1, 
                                        'random_state': 0,
                                        'bootstrap': True, 
                                        'criterion': 'entropy',
                                        'verbose': False, 
                                        "class_weight" : 'balanced'
                                        })  
        
        self.clf6 = CatBoostClassifier(**{
                                        'iterations': 10000, 
                                        'learning_rate': 0.05, 
                                        'random_seed': 0, 
                                        'allow_writing_files': False,
                                        'eval_metric': 'Logloss',
                                        'thread_count': 32,
                                        'verbose': False, 
                                        "early_stopping_rounds":10,
                                        })
        self.clf7 = ExtraTreesClassifier(**{'n_estimators': 300, 'max_leaf_nodes': 15000, 'n_jobs': -1, 'random_state': 0, 'bootstrap': True, 'criterion': 'gini'}) 
        self.clf8 = ExtraTreesClassifier(**{'n_estimators': 300, 'max_leaf_nodes': 15000, 'n_jobs': -1, 'random_state': 0, 'bootstrap': True, 'criterion': 'entropy'})  
        self.clf9 = xgb.XGBClassifier(**{
                                        'n_estimators': 10000, 
                                        'learning_rate': 0.1, 
                                        'n_jobs': 32, 
                                        'objective': 'binary:logistic', 
                                        'booster': 'gbtree', 
                                        'eval_metric': 'logloss', 
                                        'tree_method': 'hist',
                                        'verbose': False, 
                                        # "early_stopping_rounds":10,
                                        # "eval_set":[(X_test, y_test)],
                                        })
        self.clf10 = LGBMClassifier(**{
                                        'learning_rate': 0.03,
                                        'num_leaves': 128,
                                        'feature_fraction': 0.9,
                                        'min_data_in_leaf': 5,
                                        'num_threads': 32, 
                                        'objective': 'binary', 
                                        'metric': 'binary_logloss',
                                        'seed': 0,
                                        # "early_stopping_rounds":10,
                                        # "eval_set":[(X_test, y_test)],
                                        })
        
        
        self.clf11 = LogisticRegression(random_state=random_state)

        self.estimators = [
                            ("KNeighborsClassifier_uniform",self.clf0),
                            ("KNeighborsClassifier_distance",self.clf1),
                            ("LightGBMModel_extra_trees",self.clf2),
                            ("LightGBMModel_no_extra_trees",self.clf3),
                            ("RandomForestClassifier_gini",self.clf4),
                            ("RandomForestClassifier_entropy",self.clf5),
                            ("CatBoostModel",self.clf6),
                            ("ExtraTreesClassifier_gini",self.clf7),
                            ("ExtraTreesClassifier_entropy",self.clf8),
                            ("XGBoostModel",self.clf9),
                            ("LightGBMModel",self.clf10),
                            ("LogisticRegression",self.clf11),
        ]
        self.final_estimator = LogisticRegression(random_state=random_state)
        self.model = StackingClassifier(
            estimators=self.estimators,
            final_estimator = self.final_estimator,
        )
        
    def fit(self,X_train,y_train,X_test,y_test):
        self.model.fit(X_train,y_train)

    def report(self,X_train,y_train,X_test,y_test):
        res = []
        for name,sub_model in self.estimators:
            print("#"*50,"cur model :",name,"#"*50)
            sub_model.fit(X_train,y_train)
            train_ks,train_auc = self.predict_and_report(sub_model,X_train,y_train)
            test_ks,test_auc = self.predict_and_report(sub_model,X_test,y_test,types="test")
            res.append([name,train_ks,train_auc,test_ks,test_auc])
        print("#"*50,"stacking model :","#"*50)
        train_ks,train_auc = self.predict_and_report(self.model,X_train,y_train)
        test_ks,test_auc = self.predict_and_report(self.model,X_test,y_test,types="test")
        res.append(['stacking',train_ks,train_auc,test_ks,test_auc])
        self.res_df = pd.DataFrame(res,columns=['模型名字','训练集ks','训练集auc','测试集ks','测试集auc'])
        print(self.res_df)


class TrainAllModel(BaseModel):
    def __init__(self,selected_columns):
        super().__init__()
        self.selected_columns = selected_columns
        self.res = []
        self.clf0 = KNNModel()
        self.clf1 = LogisticRegressionModel()
        self.clf2 = XGBoostModel()
        self.clf3 = RandomForestModel()
        self.clf4 = LightGBMModel()
        self.clf5 = CatBoostModel()
        self.estimators = [
            ("KNNModel",self.clf0),
            ("LogisticRegressionModel",self.clf1),
            ("XGBoostModel",self.clf2),
            ("RandomForestModel",self.clf3),
            ("LightGBMModel",self.clf4),
            ("CatBoostModel",self.clf5),
        ]
    def need_select_feature(self,need_select_feature,X_train,X_test):
        return( X_train,X_test) if not need_select_feature else (X_train[self.selected_columns],X_test[self.selected_columns])

    def fit(self,X_train,y_train,X_test,y_test,):
        base_params = {}
        for name,sub_model in self.estimators:
            print("#"*50,"cur model :",name,"#"*50)
            x_train,x_test = self.need_select_feature(sub_model.need_select_feature,X_train,X_test)
            sub_model.fit(x_train,y_train,x_test,y_test)
            base_params[name] = sub_model.best_params
        print(json.dumps(base_params,ensure_ascii=False,indent=4))
        json.dump(base_params, open('base_params.json', 'w'),ensure_ascii=False,indent=4)

    def report(self,X_train,y_train,X_test,y_test):
        res = []
        for name,sub_model in self.estimators:
            print("#"*50,"cur model :",name,"#"*50)
            x_train,x_test = self.need_select_feature(sub_model.need_select_feature,X_train,X_test)
            train_ks,train_auc = self.predict_and_report(sub_model.model,x_train,y_train)
            test_ks,test_auc = self.predict_and_report(sub_model.model,x_test,y_test,types="test")
            res.append([name,train_ks,train_auc,test_ks,test_auc])
        self.res_df = pd.DataFrame(res,columns=['模型名字','训练集ks','训练集auc','测试集ks','测试集auc'])
        print(self.res_df)
##########################################################################################训练器#####################################################################################  
import sweetviz as sv
def display_df(df,name):
    print("#"*50,name,"#"*50)
    print(df)
    
class EDA():
    def __init__(self):
        pass
    def report_eda(self,train,test):
        display_df(train,'train_df')
        my_report = sv.compare([train,"train set "],[test,"test set "],target_feat="target",pairwise_analysis='off')
        my_report.show_html()
        
    def quality(self,df,target='target',iv_only=False):
        self.quality_df = toad.quality(df,target,iv_only=iv_only)
        display_df(self.quality_df,'quality_df')
        
    def describe(self,df):
        self.describe_df = toad.detector.detect(df).sort_values("missing",ascending=False).head(2000)
        display_df(self.describe_df,'describe_df')
        
    def psi(self,train,test):
        self.psi_df = toad.metrics.PSI(train,test).sort_values(ascending=False)
        display_df(self.psi_df,'psi_df')


class KBins():
    def __init__(self):
        self.c = Combiner()
        
    def fit_transform(self,train,test,y='target',method='dt',min_samples=0.05):
        print("start combiner...")
        self.c.fit(train,y=y,method=method,min_samples=min_samples,empty_separate=True)# empty_separate=True
        train_tmp = self.c.transform(train)
        test_tmp  = self.c.transform(test)
        # print(json.dumps(self.c.export(),indent=4))
        return train_tmp,test_tmp #,train_tmp.drop(['target'],axis=1),test_tmp.drop(['target'],axis=1)
    
    def export(self,columns):
        export_dict = self.c.export()
        new_export = {}
        for key,bucket in export_dict.items():
            if key in columns:
                new_export[key] = bucket
        print("#"*50,'分桶字典',"#"*50)
        print(json.dumps(new_export,indent=4,ensure_ascii=False))
        
class Processing:
    def __init__(self) -> None:
        self.kbins = None
        self.need_bins = False
        self.target_encoders = {}
        self._scaler = None
    def drop_unique_columns(self,df:DataFrame):
        # 使用 nunique() 获取每列的唯一值个数
        unique_counts = df.nunique()
        # 找到唯一值个数小于等于1的列
        low_unique_columns = unique_counts[unique_counts <= 1].index
        # 删除这些列
        df = df.drop(columns=low_unique_columns)
        return df,low_unique_columns

    def auto_fillna_strategy(self,df:DataFrame, strategy:str='mean'):
        # 确定数值型和分类型特征
        self.numeric_features = df.select_dtypes(include=['number']).columns
        self.categorical_features = df.select_dtypes(include=['object']).columns
        if self.need_bins:return df

        # 对数值型特征进行填充
        numeric_imputer = SimpleImputer(strategy=strategy)
        df[self.numeric_features] = numeric_imputer.fit_transform(df[self.numeric_features])

        # 对分类型特征进行填充
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[self.categorical_features] = categorical_imputer.fit_transform(df[self.categorical_features])
        return df
    
    def feature_encoder(self,df:DataFrame):
        # 使用LabelEncoder对分类特征进行编码
        for feature in self.categorical_features:
            self.target_encoders[feature] = LabelEncoder()
            df[feature] = self.target_encoders[feature].fit_transform(df[feature].astype(str))  
        return df
    
    def scaler(self,train,test):
        # 初始化标准化缩放器 
        
        if self._scaler is None:
            self._scaler = StandardScaler()
        # 对指定列进行标准化缩放
        train[self.numeric_features] = self._scaler.fit_transform(train[self.numeric_features])
        test[self.numeric_features] = self._scaler.transform(test[self.numeric_features])
        return train,test
    
    def bucketize(self,train,test):
        if self.kbins is None:
            self.kbins : KBins = KBins()
        return self.kbins.fit_transform(train,test)
                
        # else:
class CustomObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
class BaseTrainer():
    def __init__(self,policy:dict,label:str="target"):
        self.args(policy)
        self.target = label
        self.process:Processing = Processing()
        self.fast_stack = True
        self.eda = EDA()
        
    def args(self,policy:dict):
        for key, value in policy.items():
            setattr(self, key, value)
            
    def save(self):
        pass

    def load(self):
        pass
    
    def extract_label(self, X:DataFrame, error_if_missing:bool=True):
        y = X[self.target].copy()
        X = X.drop(self.target, axis=1)
        return X, y
    

    def upsample(self,train, strategy='random'):
        X, y = self.extract_label(train)
        if not self.need_upsample: return X, y
        strategies = {
            'random'    : RandomOverSampler(random_state=42),
            'SMOTE'     : SMOTE(random_state=42),
            'ADASYN'    : ADASYN(random_state=42)
            # 可以根据需要添加其他策略，如 BorderlineSMOTE、SVMSMOTE 等
        }

        if strategy not in strategies:
            raise ValueError("Invalid strategy. Please choose a valid upsampling strategy.")

        sampler = strategies[strategy]
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return pd.concat([X_resampled, y_resampled],axis=1)

        
    def _ensure_no_duplicate_column_names(self, X: DataFrame): #检测列名冲突
        if len(X.columns) != len(set(X.columns)):
            count_dict = defaultdict(int)
            invalid_columns = []
            for column in list(X.columns):
                count_dict[column] += 1
            for column in count_dict:
                if count_dict[column] > 1:
                    invalid_columns.append(column)
            raise AssertionError(f"Columns appear multiple times in X. Columns must be unique. Invalid columns: {invalid_columns}")
        
    def _ensure_no_label_none(self,X:DataFrame):
        with pd.option_context("mode.use_inf_as_na", True):  # treat None, NaN, INF, NINF as NA
            invalid_labels = X[self.target].isna()
        if invalid_labels.any():
            first_invalid_label_idx = invalid_labels.idxmax()
            raise ValueError(f"Label column cannot contain non-finite values (NaN, Inf, Ninf). First invalid label at idx: {first_invalid_label_idx}")
    
    def split_train_test(self,X, y,test_size=0.2,random_state=23):
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return pd.concat([X_train,self.y_train],axis=1), pd.concat([X_test,self.y_test],axis=1)
    
    def get_train_test(self,train,test):
        self.train,self.test = train,test
        self.X_train,self.y_train = self.extract_label(train)
        self.X_test,self.y_test   = self.extract_label(test)

    def train_test_split_by_time(self,df,feature_cols,target='target',split_date='2023-08-01',date_col='apply_create_time'):
        train = df[df[date_col] < split_date]
        test = df[df[date_col] >= split_date]
        y_train = train[target]
        y_test = test[target]
        self.X_train, self.X_test, self.y_train, self.y_test =  train.drop(target,axis=1)[feature_cols],test.drop(target,axis=1)[feature_cols],y_train,y_test
        self.train = pd.concat([self.X_train,self.y_train],axis=1)
        self.test  = pd.concat([self.X_test,self.y_test],axis=1)
        
    def general_data_processing(self, X: DataFrame):
        """General data processing steps used for all models."""
        X = copy.deepcopy(X)
        # 通用特征处理
        self._ensure_no_duplicate_column_names(X) # 确保列名不重复
        self._ensure_no_label_none(X)
        X, y = self.extract_label(X)
        self._original_features = list(X.columns)
        X,self.low_unique_columns = self.process.drop_unique_columns(X)
        X = self.process.auto_fillna_strategy(X)
        X = self.process.feature_encoder(X) #分桶或者编码
        train, test = self.split_train_test(X, y)
        train, test  = self.process.scaler(train, test )
        train = self.upsample(train)
        self.get_train_test(train, test)
        
    def feature_select(self,train,test):
        train_selected,dropped=toad.selection.select(train,target=self.target,empty=0.5,iv=0.05,corr=0.6,return_drop=True) 
        self.selected_columns = [col for col in train_selected.columns.tolist() if col not in [self.target]]
        # return train[self.selected_columns],test[self.selected_columns]
     
    def fit(self,X):
        self.general_data_processing(X)
        # 建模1模型融合路线
        if self.fast_stack:
            # model = StackingModel()
            model = TrainAllModel()
        model.fit(self.X_train,self.y_train,self.X_test,self.y_test)
        model.report(self.X_train,self.y_train,self.X_test,self.y_test)

    def fit_risk(self,X):
        X = copy.deepcopy(X)
        # 通用特征处理
        self._ensure_no_duplicate_column_names(X) # 确保列名不重复
        self._ensure_no_label_none(X)
        X, y = self.extract_label(X)
        self._original_features = list(X.columns)
        X,self.low_unique_columns = self.process.drop_unique_columns(X)
        train, test = self.split_train_test(X, y)
        # self.get_train_test(X_train, X_test)
        train, test = self.process.bucketize(train, test)
        self.feature_select(train, test)
        train = self.upsample(train)
        self.get_train_test(train, test)
        
        # self.eda.report_eda(self.train,self.test)
        self.eda.describe(self.train)
        self.eda.psi(self.train,self.test)
        # 建模1模型融合路线

        model_type = CustomObject(
            train_all   = TrainAllModel,
            stack       = StackingModel,

        )
        model = TrainAllModel(selected_columns = self.selected_columns)
        model.fit(self.X_train,self.y_train,self.X_test,self.y_test)
        model.report(self.X_train,self.y_train,self.X_test,self.y_test)     



class Policy :
    common_train_all_model=dict(
        need_bins       = False,
        need_upsample   = False
    )
    common_train=dict(
        need_bins       = False,
        need_upsample   = False
    )
    common_train_upsample=dict(
        need_bins       = False,
        need_upsample   = True
    )
    bins_train=dict(
        need_bins       = True,
        need_upsample   = False
    )
    bins_train_upsample=dict(
        need_bins       = True,
        need_upsample   = True
    )

# if __name__ == "__main__":
# 路线1,全部分桶
import pandas as pd
base_df = pd.read_csv("/data/sasha/project_risk_manage/risk_manage/model/data/csv/base_df.0807.csv")
from sklearn.model_selection import train_test_split
cols_common = ['loan_num','call_num_mean','call_num_sum','call_code',"merchant_num",'call_time_h','key_words','call_time_range_day','loan_range_day']   #,
trainer = BaseTrainer(Policy.common_train_upsample)
trainer.fit_risk(base_df[cols_common+['target']])