# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:15:16 2021

@author: Lab
"""

import numpy as np
import pandas as pd
import os
from os import listdir

path = './data'
data_list = listdir(path)
Data = np.array(pd.read_csv(os.path.join(path, data_list[0])))
IDX_NAME_LIST = ['SO2', 'CO', 'O3', 'PM10', 'NOx', 'NO', 'NO2', 'THC', 'NMHC', 'WIND_SPEED',
                 'AMB_TEMP', 'CH4', 'PM2.5', 'RH', 'WS_HR']

#%% Get station name
Station = []
for i in range(84):
    A = np.array(np.where(Data[:,0]==i+1)).squeeze()
    if A.size==0:
        continue
    else:
        Station.append(Data[A[0], 1])

#%% Get idx data
for idd, idx_n in enumerate(IDX_NAME_LIST):
    Sta_Data = []
    idx_Data = []
    F_idx_Data = []
    Date = []
    print("=====" + idx_n + '=====')
    for f in range(len(data_list)):
    # f = 367
        data = np.array(pd.read_csv(os.path.join(path, data_list[f])))
        Date.append(data_list[f].split('_')[4].split('.')[0])
        #print(data_list[f] + "=====" + idx_n)
        for nj, j in enumerate(Station):
            B = np.array(np.where(data[:,1]==j)).squeeze()
            if B.size==0:
                C = 0
            else:
                for bi, bidx in enumerate(B):
                    if bi==0:
                        C = data[bidx, 1:8]
                    else:
                        C = np.vstack((C, data[bidx, 1:8]))
            Sta_Data.append(C)
            
        for nk, k in enumerate(Sta_Data):
            k = np.array(k)
            if k.size==1:
                idx=0
            else:
                idx_loc = np.array(np.where(k[:,3]==idx_n)).squeeze()
                if idx_loc.size==0:
                    idx=0
                else:
                   idx = k[idx_loc, 6]
            idx_Data.append(idx)
        
        F_idx_Data.append(idx_Data)
        Sta_Data = []
        idx_Data = []
        
    F_idx_Data = np.array(F_idx_Data)
    F_idx_Data[F_idx_Data=='x'] = 0
    
    #%% Save as csv
    Station_en = ['Keelung','Xizhi','Wanli','Xindian','Tucheng','Banqiao','Xinzhuang','Cailiao','Linkou','Dashui','Shilin', 
                  ' Zhongshan','Wanhua','Guting','Songshan','Datong','Taoyuan','Dayuan','Guanyin','Pingzhen','Longtan','Hukou',
                  'Zhudong ','Xinzhu','Toufen','Miaoli','Sanyi','Fengyuan','Shalu','Dali','Zhongming','Xitun','Changhua','Xianxi', 
                  'Erlin','Nantou','Douliu','Lunbei','Xingang','Piaozi','Taixi','Chiayi','Xinying','Shanhua','Annan', 'Tainan','Meinong',
                  'Qiaotou','Renwu','Fengshan','Daliao','Lin Garden','Nanzi','Zuoying','Qianjin','Qianzhen ','Xiaogang','Pingtung','Chaozhou',
                  'Hengchun','Taitung','Hualian','Yangming','Yilan','Winter Mountain','Sanchong','Zhongli', 'Zhushan','Yonghe','Fuxing','Puli',
                  'Mazu','Jinmen','Magong','Guanshan','Mailiao','Fugui Jiao']
    
    select_df = pd.DataFrame(F_idx_Data, index=Date, columns=Station_en)
    select_df.to_csv(idx_n + ".csv")
    F_idx_Data = []
     