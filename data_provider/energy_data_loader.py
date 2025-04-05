# data_provider/energy_data_loader.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class EnergyDataset(Dataset):
    """能源数据集"""
    
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', data_parser=None, target='OT', scale=True, 
                 future_vars=None, timeenc=0, freq='h', seasonal_patterns=None):
        # 初始化
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.scale = scale
        self.target = target
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.future_vars = future_vars  # 未来已知变量列表
        
        # 数据大小 [序列长度, 标签长度, 预测长度]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # 解析数据
        self.__read_data__()
        
    def __read_data__(self):
        """读取数据"""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 处理索引
        if not 'date' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw.index)
        else:
            df_raw['date'] = pd.to_datetime(df_raw['date'])
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        # 划分数据集
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[0] if self.flag == 'train' else (border1s[1] if self.flag == 'val' else border1s[2])
        border2 = border2s[0] if self.flag == 'train' else (border2s[1] if self.flag == 'val' else border2s[2])
        
        # 特征选择
        cols_data = df_raw.columns[1:]
        if self.features == 'M' or self.features == 'MS':
            # 多变量
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 单变量
            df_data = df_raw[[self.target]]
        
        # 标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 提取时间特征
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_stamp = time_features(pd.DatetimeIndex(df_stamp.index), freq=self.freq)
        
        # 处理未来已知变量
        if self.future_vars is not None and len(self.future_vars) > 0:
            # 选择未来已知变量
            available_future_vars = [var for var in self.future_vars if var in df_raw.columns]
            if available_future_vars:
                future_data = df_raw[available_future_vars][border1:border2].values
            else:
                # Create dummy data if none of the future vars are available
                print(f"Warning: None of the specified future variables {self.future_vars} are in the dataset. Using zeros instead.")
                future_data = np.zeros((border2-border1, len(self.future_vars)))
            # 标准化
            if self.scale:
                future_train_data = df_raw[self.future_vars][border1s[0]:border2s[0]].values
                self.future_scaler = StandardScaler()
                self.future_scaler.fit(future_train_data)
                future_data = self.future_scaler.transform(future_data)
            self.future_data = future_data
        else:
            self.future_data = None
        
        # 设置数据
        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[border1:border2, [df_data.columns.get_loc(self.target)]]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        """获取单个样本"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 区间索引
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # 处理未来已知变量
        if self.future_data is not None:
            future_vars = self.future_data[s_end:s_end + self.pred_len]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, future_vars
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        """数据集长度"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """反标准化"""
        return self.scaler.inverse_transform(data)

def energy_data_provider(args, flag):
    """提供能源数据"""
    data_set = EnergyDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        future_vars=args.future_vars if hasattr(args, 'future_vars') else None,
        timeenc=0 if args.embed != 'timeF' else 1,
        freq=args.freq
    )
    
    # 确定是否返回未来变量
    has_future_vars = hasattr(args, 'use_future_vars') and args.use_future_vars
    
    # 创建数据加载器
    if has_future_vars:
        # 自定义collate_fn处理未来变量
        def collate_fn(batch):
            seq_x = torch.FloatTensor(np.array([item[0] for item in batch]))
            seq_y = torch.FloatTensor(np.array([item[1] for item in batch]))
            seq_x_mark = torch.FloatTensor(np.array([item[2] for item in batch]))
            seq_y_mark = torch.FloatTensor(np.array([item[3] for item in batch]))
            future_vars = torch.FloatTensor(np.array([item[4] for item in batch]))
            return seq_x, seq_y, seq_x_mark, seq_y_mark, future_vars
        
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=flag == 'train',
            num_workers=args.num_workers,
            drop_last=flag == 'train',
            collate_fn=collate_fn
        )
    else:
        # 标准数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=flag == 'train',
            num_workers=args.num_workers,
            drop_last=flag == 'train'
        )
    
    return data_set, data_loader