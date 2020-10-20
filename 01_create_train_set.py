import os, sys
import numpy as np
import pandas as pd
import joblib
import time
import datetime
import calendar
from sklearn.preprocessing import StandardScaler

np.random.seed(123)  


def create_train_basic_featue (tti_file_name) :

    df = pd.read_csv(tti_file_name)
    
    df_out = pd.DataFrame(data=None)

    road_list = df.sort_values(['id_road','time'])['id_road'].tolist()
    time_list = df.sort_values(['id_road','time'])['time'].tolist()
    speed_list = df.sort_values(['id_road','time'])['speed'].tolist()

    TTI_list = df.sort_values(['id_road','time'])['TTI'].tolist()

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

    df_out['date'] = df_out['time'].apply(lambda x : x[0:10])
    df_out['hhmm'] = df_out['time'].apply(lambda x : float(x[11:16].replace(':','.')))
    df_out['hhmm'] = df_out['hhmm'].apply(lambda k : int(k) + (k - int(k))/0.6)

    df_out['week_day'] = df_out['date'].apply(lambda x : 1 + calendar.weekday(int(x[0:4]),int(x[5:7]),int(x[8:10])))

    df_out['week_day_1'] = df_out['week_day'].apply(lambda x : 1 if x == 1 else 0 )
    df_out['week_day_2'] = df_out['week_day'].apply(lambda x : 1 if x == 2 else 0 )
    df_out['week_day_3'] = df_out['week_day'].apply(lambda x : 1 if x == 3 else 0 )
    df_out['week_day_4'] = df_out['week_day'].apply(lambda x : 1 if x == 4 else 0 )
    df_out['week_day_5'] = df_out['week_day'].apply(lambda x : 1 if x == 5 else 0 )
    df_out['week_day_6'] = df_out['week_day'].apply(lambda x : 1 if x == 6 else 0 )
    df_out['week_day_7'] = df_out['week_day'].apply(lambda x : 1 if x == 7 else 0 )

    holiday_list = ['2019-01-01','2019-02-04','2019-02-05','2019-02-06','2019-02-07','2019-02-08','2019-02-09','2019-02-20',
                    '2019-02-04','2019-03-05','2019-03-06','2019-03-07','2019-04-05','2019-04-06','2019-04-07','2019-05-01',
                    '2019-05-02','2019-05-03','2019-05-04','2019-06-07','2019-06-08','2019-06-09','2019-09-13','2019-09-14',
                    '2019-09-15','2019-10-01','2019-10-02','2019-10-03','2019-10-04','2019-10-05','2019-10-06','2019-10-07',
                    '2020-01-01']

    df_out['holiday'] = df_out['date'].apply(lambda x : 1 if x in holiday_list else 0 )

    df_out['holiday_a1'] = df_out['date'].apply(lambda x : 1 if x not in holiday_list and
                                        str(datetime.datetime.strptime(x,'%Y-%m-%d').date() + datetime.timedelta(days=-1)) in holiday_list else 0 )

    df_out['holiday_b1'] = df_out['date'].apply(lambda x : 1 if x not in holiday_list and
                                        str(datetime.datetime.strptime(x,'%Y-%m-%d').date() + datetime.timedelta(days=1)) in holiday_list else 0 )

    df_out = df_out.loc[df_out['TTI_mean']!='Nan']
        
    print('Basic feature data created at %s'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )
    
    return df_out


def merge_tti_feature (df_base , df_feature , n_shift):
    df_feature['time'] = df_feature['time'].apply(lambda x : x + pd.Timedelta(minutes=10*n_shift))
    df_train_feat = pd.merge(df_base,df_feature,on='time',how='left') 
    df_train_feat.interpolate(inplace=True)
    return df_train_feat

def create_train_gps_feagure(tti_file_name,feature_parm_filename,gps_file_name):
    
    df_feature_parm = pd.read_csv(feature_parm_filename)
    
    df_gps = pd.read_csv(gps_file_name)
    df_gps['time'] = df_gps['time'].astype('datetime64')

    df_tti = pd.read_csv(tti_file_name)
    
    df_tti['time'] = df_tti['time'].astype('datetime64')
    df_tti = df_tti[(df_tti.time.dt.date <= datetime.date(2019,12,20))]
    
    road_list = ['276183','276184','275911','275912','276240','276241','276264','276265','276268','276269','276737','276738']

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

    print('GPS feature data created at %s'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )

    return df_base

def train_scaler(df_train_feat,X_scaler_filename,Y_scaler_filename):
    
    df_train_feat1 = df_train_feat.drop(['id_road','TTI','speed','time','date','week_day','week_day_1','week_day_2','week_day_3','week_day_4','week_day_5','week_day_6','week_day_7','holiday','holiday_a1','holiday_b1'],axis=1)
    df_train_feat2 = df_train_feat[['id_road','speed','time','date','week_day','week_day_1','week_day_2','week_day_3','week_day_4','week_day_5','week_day_6','week_day_7','holiday','holiday_a1','holiday_b1']]    
    df_train_feat3 = df_train_feat[['TTI']]
    list_X = df_train_feat1.columns 
    list_Y = df_train_feat3.columns  
    
    X_scaler = StandardScaler()
    X_scaler.fit(df_train_feat1)
    
    Y_scaler = StandardScaler()
    Y_scaler.fit(df_train_feat3)
        
    df_scaler_X = X_scaler.fit_transform(df_train_feat1)
    df_scaler_X1 = pd.DataFrame(df_scaler_X,columns=list_X)

    df_scaler_Y = Y_scaler.fit_transform(df_train_feat3)
    df_scaler_Y1 = pd.DataFrame(df_scaler_Y,columns=list_Y)
    
    train_df = pd.concat([df_train_feat2,df_scaler_Y1,df_scaler_X1],axis=1)
    
    joblib.dump(X_scaler, X_scaler_filename)  # save
    joblib.dump(Y_scaler, Y_scaler_filename)  # save
    
    return train_df
       
# batch1

if __name__ == "__main__":
    
    tti_file_name = "../data/train_TTI.csv"
    df_basic = create_train_basic_featue(tti_file_name)
    
    feature_parm_filename = "./config/v7_extract_feature.csv"
    gps_file_name = "../data/v_road_level_2_summary_v7.csv"
    df_gps = create_train_gps_feagure(tti_file_name,feature_parm_filename,gps_file_name)
    
    df_basic['time'] = df_basic['time'].astype('datetime64')
    df_basic = df_basic[(df_basic.time.dt.date <= datetime.date(2019,12,20))]

    df_gps['time'] = df_gps['time'].astype('datetime64')

    df_train_feat = pd.merge(df_basic,df_gps,on='time',how='left')
    df_train_feat.interpolate(inplace=True)
    df_train_feat.dropna(inplace=True)

#    # 输出训练集特征文件
#    train_file_name = "../data/trainset_stage2.csv"
#    df_train_feat.to_csv(train_file_name,encoding='utf8',index=0)     
    
    X_scaler_filename = "../python/X_scale.save"
    Y_scaler_filename = "../python/Y_scale.save"
    train_df = train_scaler(df_train_feat,X_scaler_filename,Y_scaler_filename)
      
    # 输出归一化训练集特征文件    
    train_file_name1 = "../data/trainset_z_score_stage2.csv"
    train_df.to_csv(train_file_name1,encoding='utf8',index=0)
