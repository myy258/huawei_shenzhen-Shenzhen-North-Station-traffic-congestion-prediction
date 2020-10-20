# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:39:25 2020

@author: myy
"""

import os, sys
import numpy as np
import pandas as pd

import time
import datetime

np.random.seed(123)  

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from xgboost import XGBRegressor

def train_xgb(train_file_name,feature_parm_prefix,model_prefix):
    
    step_list = ['step1','step2','step3']

    road_list = ['276183','276184','275911','275912','276240','276241','276264','276265','276268','276269','276737','276738']
    
    df_train_feat = pd.read_csv(train_file_name)

    for step in step_list:

        print('\nTraining %s at %s'%(step,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )

        # 读取配置文件，得到本轮所需要的特征
        mfile_name = feature_parm_prefix + step + "_features.csv"
        df_features = pd.read_csv(mfile_name)

        for road in road_list:

            print('\nTraining %s road %s at %s'%(step,road,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )

            df_train = df_train_feat.loc[df_train_feat.id_road == int(road)]
            curr_features = df_features[road].dropna().tolist()

            ALL_X = df_train[curr_features].values
            ALL_Y = df_train['TTI'].values

            model_xgb = XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.05, min_child_weight=2,seed = 0,
                                    reg_lambda=1, subsample=0.7 , colsample_bytree=0.8 , gamma=0.3 , reg_alpha = 0.3 )

            model_xgb.fit(ALL_X, ALL_Y)        
            print('estimation of : ',road)
            print('mae of training set:',mean_absolute_error(ALL_Y, model_xgb.predict(ALL_X)))

            mfile_name = model_prefix + step + '_' + road+'.pkl'
            joblib.dump(model_xgb, mfile_name)
                
    return

if __name__ == "__main__":
    
    feature_parm_prefix = "./config/v7_"
    # 标准化处理训练集
#    train_file_name_z_score = "../data/trainset_z_score_stage2.csv"
    
    train_file_name = "../data/trainset_stage2.csv"
    
    #输出xgb模型的文件名前缀
    xgb_model_prefix = "./model/xgboost/xgb_stage2_"
    
#    xgb_model_prefix_zscore = "./model/xgboost/xgb_z_score_stage2_"  

    train_xgb(train_file_name,feature_parm_prefix,xgb_model_prefix)    
#    train_xgb(train_file_name_z_score,feature_parm_prefix,xgb_model_prefix_zscore)
