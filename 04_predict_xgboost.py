# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:40:51 2020

@author: myy
"""

import os, sys
import numpy as np
import pandas as pd

import time
import datetime

np.random.seed(123)  

## predict by road
## use step 1 / 2 / 3 gps

from xgboost import XGBRegressor
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")

def predict_score(predict_file_name,feature_parm_prefix,model_prefix):

    df_test_feat = pd.read_csv(predict_file_name)
    
    file_name = feature_parm_prefix + "step1_features.csv"
    df_step1_feature_name = pd.read_csv(file_name)

    file_name = feature_parm_prefix + "step2_features.csv"
    df_step2_feature_name = pd.read_csv(file_name)

    file_name = feature_parm_prefix + "step3_features.csv"
    df_step3_feature_name = pd.read_csv(file_name)

    road_list = ['276183','276184','275911','275912','276240','276241','276264','276265','276268','276269','276737','276738']

    df_out = pd.DataFrame(data=None)

    for road in road_list:

        print('Predicting %s at %s'%(road,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )
        
        mfile_name = model_prefix + 'step1_' + road+'.pkl'
        regr_step1 = joblib.load(mfile_name)

        mfile_name = model_prefix + 'step2_' + road+'.pkl'
        regr_step2 = joblib.load(mfile_name)

        mfile_name = model_prefix + 'step3_' + road+'.pkl'
        regr_step3 = joblib.load(mfile_name)
        
        df_out_tmp = pd.DataFrame(data=None)
        sample_id_list = df_test_feat.loc[(df_test_feat.id_road == int(road)) &
                                          (df_test_feat.id_sample != -1)]['id_sample'].values.tolist()

        step_idx = 0

        for i in sample_id_list:

            if float(df_test_feat.loc[df_test_feat.id_sample == i]['TTI_mean'].values[0]) > -1:
                curr_features = df_step1_feature_name[road].dropna().tolist()
                test_X = df_test_feat.loc[df_test_feat.id_sample == i][curr_features].values
                test_Y = regr_step1.predict(test_X)
                df_test_feat.loc[df_test_feat.id_sample == i,'TTI'] = test_Y[0]
                step_idx = 1
            else:
                ind = df_test_feat.loc[df_test_feat.id_sample == i].index
                t1 = df_test_feat.loc[ind - 1]['TTI'].values[0]
                t2 = df_test_feat.loc[ind - 2]['TTI'].values[0]
                t3 = df_test_feat.loc[ind - 3]['TTI'].values[0]

                df_test_feat.loc[df_test_feat.id_sample == i,'TTI_mean'] = t1
                df_test_feat.loc[df_test_feat.id_sample == i,'TTI_mean2'] = (t2 + t1) / 2
                df_test_feat.loc[df_test_feat.id_sample == i,'TTI_mean3'] = (t3 + t2 + t1) / 3

                df_test_feat.loc[df_test_feat.id_sample == i,'TTI_dtl_rate'] = (t1 - t2) / t2 
                df_test_feat.loc[df_test_feat.id_sample == i,'TTI_dtl2_rate'] = (t1 - t3) / t3 

                if step_idx == 1 :
                    curr_features = df_step2_feature_name[road].dropna().tolist()
                    test_X = df_test_feat.loc[df_test_feat.id_sample == i][curr_features].values
                    test_Y = regr_step2.predict(test_X)                
                    step_idx += 1
                elif step_idx == 2 :
                    curr_features = df_step3_feature_name[road].dropna().tolist()
                    test_X = df_test_feat.loc[df_test_feat.id_sample == i][curr_features].values
                    test_Y = regr_step3.predict(test_X)                
                    step_idx += 1                
                else:
                    print('Error : out of 3 steps.')

                df_test_feat.loc[df_test_feat.id_sample == i,'TTI'] = test_Y[0]            

    df_out['id_sample'] = df_test_feat['id_sample']
    df_out['tti'] = df_test_feat['TTI']
    
    df_out = df_out.loc[df_out.id_sample>=0].sort_values('id_sample')

    return df_out

def predict_score_z_score (predict_file_name_z_score,feature_parm_prefix,xgb_model_prefix_z_score,Y_scaler):

    df_xgb_z_score = predict_score(predict_file_name_z_score,feature_parm_prefix,xgb_model_prefix_z_score) 
    df_xgb_z_score.columns=['id_sample','tti']
    
    df_xgb_z_score['tti'] = Y_scaler.inverse_transform(df_xgb_z_score['tti'])
        
    return df_xgb_z_score

if __name__ == "__main__":

    feature_parm_prefix = "./config/v7_"
#    predict_file_name = "../data/predictset_stage2.csv"
    predict_file_name_z_score = "../data/predictset_z_score_stage2.csv"
#    xgb_model_prefix = "./model/xgboost/xgb_stage2_"
    xgb_model_prefix_z_score = "./model/xgboost/xgb_z_score_stage2_"
    
    #提交的结果文件(标准化和未标准化)
    submit_file_z_score = "../data/submit_xgb_zscore_stage2.csv"
#    submit_file = "../data/submit_xgb_stage2.csv"
   
#    df_xgb = predict_score(predict_file_name,feature_parm_prefix,xgb_model_prefix)    

#    df_xgb.columns=['id_sample','tti']
#    df_xgb[['id_sample','tti']].sort_values('id_sample').to_csv(submit_file,encoding='utf8',index=0)            

    # 导入Y_scale   
    Y_scaler_filename = "../python/Y_scale.save"
    Y_scaler = joblib.load(Y_scaler_filename)
    
    df_xgb_z_score = predict_score_z_score(predict_file_name_z_score,feature_parm_prefix,xgb_model_prefix_z_score,Y_scaler)
    df_xgb_z_score[['id_sample','tti']].sort_values('id_sample').to_csv(submit_file_z_score,encoding='utf8',index=0)    