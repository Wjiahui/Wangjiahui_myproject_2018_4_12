# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
import os
from config import *

class StockDataSet(object):
    def __init__(self, config, stock_name):
        self.stock_name = stock_name
        self.normalized = config['normalized']
        self.time_step = config['time_step']
        self.batch_size = config['batch_size']

        # 读入股票数据
        f = open(os.path.join("data", '%s.csv' % stock_name))
        df = pd.read_csv(f)
        data_df = df.iloc[:, 1:7]  # 取第2-7列
        close_price = data_df['Close']

        # 添加label列
        c = np.array(close_price[1:])
        last_price = c[-1]
        data_df['label'] = np.append(c, last_price)
        self.label_column = 6

        self.data_df = data_df.values
        self.train_x, self.train_y, self.test_x, self.test_y, self.batch_size = self.prepare_data()
        # print(self.train_x, self.train_y, self.test_x, self.test_y, self.batch_size)

    def info(self):
        print("StockName:%s train: %d test: %d" % (self.stock_name, len(self.train_x), len(self.test_y)))

    def prepare_data(self):
        train_len = 16800

        train_x, train_y, test_x, test_y = [], [], [], []
        batch_index = []

        # 标准化
        if self.normalized:
            normalized_train_data = (self.data_df - np.mean(self.data_df, axis=0)) / np.std(self.data_df, axis=0)

        data_train = normalized_train_data[: train_len]
        data_test = normalized_train_data[train_len:]

        # 训练集
        for i in range(len(data_train)-self.time_step):
           if i % self.batch_size == 0:
                batch_index.append(i)
           x = normalized_train_data[i:i+self.time_step, :self.label_column]
           y = normalized_train_data[i:i+self.time_step, self.label_column, np.newaxis]
           train_x.append(x.tolist())
           train_y.append(y.tolist())
        batch_index.append((len(data_train)-self.time_step))

        # 测试集
        for i in range(len(data_test)-self.time_step):
            x = normalized_train_data[i:i + self.time_step, :self.label_column]
            y = normalized_train_data[i:i + self.time_step, self.label_column, np.newaxis]
            test_x.append(x.tolist())
            test_y.append(y.tolist())

        return train_x, train_y, test_x, test_y, batch_index

if __name__ == "__main__":
    s = StockDataSet(CONFIG['lstm'], 'SP500')
    s.info()
