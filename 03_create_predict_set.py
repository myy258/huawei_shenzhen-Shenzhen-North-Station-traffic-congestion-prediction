import os, sys
import numpy as np
import pandas as pd
import joblib
import time
import datetime
import calendar
from sklearn.preprocessing import StandardScaler

np.random.seed(123)  

# 构造基础预测集特征

def create_predict_basic_featue(tti_file_name,nolabel_file_name):

    df_noLabel = pd.read_csv(nolabel_file_name,encoding='utf8')
    df_noLabel['speed'] = -1 
    df_noLabel['TTI'] = -1 
    df_noLabel.head()

    df_TTI = pd.read_csv(tti_file_name,encoding='utf8')
    df_TTI['id_sample'] = -1
    df_TTI.head()

    df_predict = pd.concat([df_noLabel,df_TTI])

    df_out = pd.DataFrame(data=None)

    id_sample_list = df_predict.sort_values(['id_road','time'])['id_sample'].tolist()
    road_list = df_predict.sort_values(['id_road','time'])['id_road'].tolist()
    time_list = df_predict.sort_values(['id_road','time'])['time'].tolist()
    speed_list = df_predict.sort_values(['id_road','time'])['speed'].tolist()
    TTI_list = df_predict.sort_values(['id_road','time'])['TTI'].tolist()

    tti_buf = []    
    TTI_mean_list = []
    TTI_mean2_list = []
    TTI_mean3_list = []
    TTI_dtl_rate_list = []
    TTI_dtl2_rate_list = []

    for i in range(len(time_list)):
        if len(tti_buf) <= 2 :
            tti_buf.append(TTI_list[i])
            TTI_mean_list.append('Nan')
            TTI_mean2_list.append('Nan')
            TTI_mean3_list.append('Nan')    
            TTI_dtl_rate_list.append('Nan')
            TTI_dtl2_rate_list.append('Nan')        
            continue

        startTime = datetime.datetime.strptime(time_list[i-1], "%Y-%m-%d %H:%M:%S")
        endTime = datetime.datetime.strptime(time_list[i], "%Y-%m-%d %H:%M:%S")
        seconds = (endTime - startTime).seconds

        if seconds != 600:
            tti_buf = []
            tti_buf.append(TTI_list[i])
            TTI_mean_list.append('Nan')
            TTI_mean2_list.append('Nan')
            TTI_mean3_list.append('Nan')    
            TTI_dtl_rate_list.append('Nan')
            TTI_dtl2_rate_list.append('Nan')        
            continue

        TTI_mean_list.append(tti_buf[2])
        TTI_mean2_list.append((tti_buf[2]+tti_buf[1])/2)
        TTI_mean3_list.append((tti_buf[0]+tti_buf[1]+tti_buf[2])/3)
        TTI_dtl_rate_list.append((tti_buf[2] - tti_buf[1])/tti_buf[1])
        TTI_dtl2_rate_list.append((tti_buf[2] - tti_buf[0])/tti_buf[0])

        tti_buf.pop(0)
        tti_buf.append(TTI_list[i])

    df_out['id_sample'] = pd.Series(id_sample_list)    
    df_out['id_road'] = pd.Series(road_list)
    df_out['TTI'] = pd.Series(TTI_list)
    df_out['speed'] = pd.Series(speed_list)
    df_out['time'] = pd.Series(time_list)
    df_out['TTI'] = pd.Series(TTI_list)
    df_out['TTI_mean'] = pd.Series(TTI_mean_list)
    df_out['TTI_mean2'] = pd.Series(TTI_mean2_list)
    df_out['TTI_mean3'] = pd.Series(TTI_mean3_list)
    df_out['TTI_dtl_rate'] = pd.Series(TTI_dtl_rate_list)
    df_out['TTI_dtl2_rate'] = pd.Series(TTI_dtl2_rate_list)

    df = df_out

    df['date'] = df['time'].apply(lambda x : x[0:10])
    df['hhmm'] = df['time'].apply(lambda x : float(x[11:16].replace(':','.')))
    df['hhmm'] = df['hhmm'].apply(lambda k : int(k) + (k - int(k))/0.6)

    df['week_day'] = df['date'].apply(lambda x : 1 + calendar.weekday(int(x[0:4]),int(x[5:7]),int(x[8:10])))

    df['week_day_1'] = df['week_day'].apply(lambda x : 1 if x == 1 else 0 )
    df['week_day_2'] = df['week_day'].apply(lambda x : 1 if x == 2 else 0 )
    df['week_day_3'] = df['week_day'].apply(lambda x : 1 if x == 3 else 0 )
    df['week_day_4'] = df['week_day'].apply(lambda x : 1 if x == 4 else 0 )
    df['week_day_5'] = df['week_day'].apply(lambda x : 1 if x == 5 else 0 )
    df['week_day_6'] = df['week_day'].apply(lambda x : 1 if x == 6 else 0 )
    df['week_day_7'] = df['week_day'].apply(lambda x : 1 if x == 7 else 0 )

    holiday_list = ['2019-01-01','2019-02-04','2019-02-05','2019-02-06','2019-02-07','2019-02-08','2019-02-09',
                    '2019-02-20','2019-02-04','2019-03-05','2019-03-06','2019-03-07','2019-04-05','2019-04-06',
                    '2019-04-07','2019-05-01','2019-05-02','2019-05-03','2019-05-04','2019-06-07','2019-06-08',
                    '2019-06-09','2019-09-13','2019-09-14','2019-09-15','2019-10-01','2019-10-02','2019-10-03',
                    '2019-10-04','2019-10-05','2019-10-06','2019-10-07','2020-01-01']

    df['holiday'] = df['date'].apply(lambda x : 1 if x in holiday_list else 0 )

    df['holiday_a1'] = df['date'].apply(lambda x : 1 if x not in holiday_list and
                                        str(datetime.datetime.strptime(x,'%Y-%m-%d').date() + datetime.timedelta(days=-1)) in holiday_list else 0 )

    df['holiday_b1'] = df['date'].apply(lambda x : 1 if x not in holiday_list and
                                        str(datetime.datetime.strptime(x,'%Y-%m-%d').date() + datetime.timedelta(days=1)) in holiday_list else 0 )

    return df

def merge_tti_feature (df_base , df_feature , n_shift):
    df_feature['time'] = df_feature['time'].apply(lambda x : x + pd.Timedelta(minutes=10*n_shift))
    df_train_feat = pd.merge(df_base,df_feature,on='time',how='left') 
    df_train_feat.interpolate(inplace=True)
    return df_train_feat

def create_predict_gps_feagure(df_tti,feature_parm_filename,gps_file_name):

    df_gps = pd.read_csv(gps_file_name)
    df_gps['time'] = df_gps['time'].astype('datetime64')

    df_tti['time'] = df_tti['time'].astype('datetime64')

    df_feature_parm = pd.read_csv(feature_parm_filename)

    df_base = pd.DataFrame(data=None)
    df_base['time'] = df_tti['time'].unique()

    for index,row in df_feature_parm.iterrows():      
        p_grid_id=row["p_grid_id"]
        field_list=['time',row["field"]]        
        feature=row["feature"]
        n_shift=row["n_shift"]
        p_dx_flag=row["p_dx_flag"]
        p_dy_flag=row["p_dy_flag"]

        df_cur = df_gps[(df_gps.p_grid_id == p_grid_id ) & (df_gps.p_dx_flag == p_dx_flag) & (df_gps.p_dy_flag == p_dy_flag)][field_list]
        df_cur.columns = ['time',row["feature"]] 

        if not df_cur.empty :
            df_base=merge_tti_feature(df_base,df_cur,n_shift)

    print('Finished at %s'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )

    df_base.fillna(0,inplace=True)

    return df_base

def predict_scaler(df_predict_feat,X_scaler_filename,Y_scaler_filename):
    
    df_predict_feat1 = df_predict_feat.drop(['id_sample','id_road','TTI','speed','time','date','week_day','week_day_1','week_day_2','week_day_3','week_day_4','week_day_5','week_day_6','week_day_7','holiday','holiday_a1','holiday_b1'],axis=1)
    df_predict_feat2 = df_predict_feat[['id_sample','id_road','speed','time','date','week_day','week_day_1','week_day_2','week_day_3','week_day_4','week_day_5','week_day_6','week_day_7','holiday','holiday_a1','holiday_b1']]
    df_predict_feat3 = df_predict_feat[['TTI']]
    list_X = df_predict_feat1.columns 
    list_Y = df_predict_feat3.columns
    
    df_predict_feat4 = df_predict_feat1.replace('-1.0','Nan')
    df_predict_feat3 = df_predict_feat3.replace(-1.00000,np.nan)
    
    X_scaler = joblib.load(X_scaler_filename)
    Y_scaler = joblib.load(Y_scaler_filename)

    df_scaler_X = X_scaler.transform(df_predict_feat4)
    df_scaler_X1 = pd.DataFrame(df_scaler_X,columns=list_X)

    df_scaler_Y = Y_scaler.transform(df_predict_feat3)
    df_scaler_Y1 = pd.DataFrame(df_scaler_Y,columns=list_Y)
    
    predict_df = pd.concat([df_predict_feat2,df_scaler_Y1,df_scaler_X1],axis=1)
    predict_df = predict_df.replace(np.nan,'Nan')

    return predict_df

if __name__ == "__main__":
    
    tti_file_name = "../data/toPredict_train_TTI_stage2.csv"
    nolabel_file_name = "../data/toPredict_noLabel_stage2.csv"
    df_basic = create_predict_basic_featue(tti_file_name,nolabel_file_name)

    feature_parm_filename = "./config/v7_extract_feature.csv"
    gps_file_name = "../data/v_road_level_2_summary_predict_v7.csv"
    df_gps = create_predict_gps_feagure(df_basic,feature_parm_filename,gps_file_name)
    
    df_predict_feat = pd.merge(df_basic,df_gps,on='time',how='left')
    df_predict_feat.interpolate(inplace=True)

#    # 输出测试集特征文件
#    predict_file_name = "../data/predictset_stage2.csv"
#    df_predict_feat.to_csv(predict_file_name,encoding='utf8',index=0)
    
    # 导入标准化储存文件         
    X_scaler_filename = "../python/X_scale.save"
    Y_scaler_filename = "../python/Y_scale.save"
    predict_df = predict_scaler(df_predict_feat,X_scaler_filename,Y_scaler_filename)
    predict_df['TTI'] = predict_df['TTI'].astype(float)
    
    # 输出归一化训练集特征文件    
    predict_file_name1 = "../data/predictset_z_score_stage2.csv"
    predict_df.to_csv(predict_file_name1,encoding='utf8',index=0)
