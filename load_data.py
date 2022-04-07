import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import torch


datapath = 'data/demand-boxed-202106to10.csv'
device = torch.device("cuda")

class MinMaxNormalization(object):
    """
    MinMax Normalization --> [-1, 1]
    x = (x - min) / (max - min).
    x = x * 2 - 1
    """
    def __init__(self,min=None,max=None):
        self.min = min
        self.max = max
    
    def fit(self, x):
        if self.min is None:
            self.min = x.min()
        if self.max is None:
            self.max = x.max()
    
    def transform(self, x):
        x = 1. * (x - self.min) / (self.max - self.min)
        x = x * 2. - 1.
        return x
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        x = (x + 1.) / 2.
        x = 1. * x * (self.max - self.min) + self.min
        return x


class CreataSequentialData():
    def __init__(self,datapath,shorttime_history,longtime_periodicity_days,pred_len=1,split_ratio=[0.6,0.2,0.2] ,datashape=(12,7),channel_last=False):
        self.datapath = datapath
        self.datashape = datashape
        self.shorttime_history = shorttime_history
        self.longtime_periodicity_days = longtime_periodicity_days
        self.channel_last = channel_last
        self.split_ratio = split_ratio
        assert pred_len>0 and isinstance(pred_len, int)
        self.pred_len = pred_len
        self.create_inputoutput()

    def load_raw_np(self):
        rawdata_df = pd.read_csv(self.datapath)
        rawdata_df['datetime'] = pd.to_datetime(rawdata_df['datetime'])
        rawdata_df = rawdata_df.sort_values(by=['datetime','region']).reset_index(drop=True)
        # rawdatastep: hours
        self.rawdatastep = int(round((rawdata_df['datetime'][self.datashape[0]*self.datashape[1]] - rawdata_df['datetime'][0]).to_pytimedelta().total_seconds()/3600))
        self.rawdata_np = rawdata_df['demand'].to_numpy().reshape((-1,)+self.datashape).astype('float64')
        self.minmaxnormalization = MinMaxNormalization(min=0,max=300)
        # data_np is the data after MinMaxNormalization
        self.data_np = self.minmaxnormalization.fit_transform(self.rawdata_np)
        #return self.rawdata_np

    def create_inputoutput(self):
        self.load_raw_np()
        data_per_day = int(24/self.rawdatastep)
        fulldata_length = len(self.data_np)
        # history_length + 1 is the length per sample
        history_length = int(max(data_per_day*self.longtime_periodicity_days,self.shorttime_history))
        seq_length = self.longtime_periodicity_days + self.shorttime_history
        sample_num = fulldata_length - history_length + 1 - self.pred_len
        self.sample_num =sample_num
        self.X_data = np.zeros((sample_num,seq_length,self.datashape[0],self.datashape[1])) # suppose 'channel-first'
        self.Y_data = np.zeros((sample_num,self.datashape[0],self.datashape[1]))
        for i in range(sample_num):
            index_longtime = slice(i + history_length - data_per_day*self.longtime_periodicity_days, i + history_length, data_per_day)
            self.X_data[i,:self.longtime_periodicity_days,:,:] = self.data_np[index_longtime]
            self.X_data[i,self.longtime_periodicity_days:,:,:] = self.data_np[(i+ history_length- self.shorttime_history) : (i+history_length)]
            self.Y_data[i,:,:] = self.data_np[i+history_length+self.pred_len-1]
        if self.channel_last:
            self.X_data = self.X_data.transpose([0,2,3,1])
        return self.X_data, self.Y_data
    
    def load_train_data(self):
        train_index = slice(0,int(self.sample_num*self.split_ratio[0]/np.sum(self.split_ratio)))
        return self.X_data[train_index], self.Y_data[train_index]
    
    def load_val_data(self):
        val_index = slice(int(self.sample_num*self.split_ratio[0]/np.sum(self.split_ratio)),int(self.sample_num*(self.split_ratio[0]+self.split_ratio[1])/np.sum(self.split_ratio)))
        return self.X_data[val_index], self.Y_data[val_index]
    
    def load_test_data(self):
        test_index = slice(int(self.sample_num*(self.split_ratio[0]+self.split_ratio[1])/np.sum(self.split_ratio)),self.sample_num)
        return self.X_data[test_index], self.Y_data[test_index]


class BikeNYCDataset(Dataset):
    def __init__(self, data_XY):
        super(BikeNYCDataset, self).__init__()
        self.X_data = torch.from_numpy(data_XY[0]).to(device)
        self.Y_data = torch.from_numpy(data_XY[1]).to(device)
    
    def __getitem__(self, index):
        X = self.X_data[index]
        Y = self.Y_data[index]
        return X.float(), Y.float()

    def __len__(self):
        return self.X_data.shape[0]
