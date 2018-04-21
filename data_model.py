# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

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
        self.batch_index, self.train_x, self.train_y = self.get_train_data()
        self.mean, self.std, self.test_x, self.test_y = self.get_test_data()
        # self.train_x, self.train_y, self.test_x, self.test_y, self.batch_index = self.prepare_data()
        # print(self.train_x, self.train_y, self.test_x, self.test_y, self.batch_size)

    def info(self):
        print("hello")
        print("StockName:%s train: %d test: %d" % (self.stock_name, len(self.train_x), len(self.test_y)))

    def prepare_data(self):
        train_len = 50

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
           # y = normalized_train_data[i:i + self.time_step, self.label_column]
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

    # 获取训练集
    def get_train_data(self, train_begin=0, train_end=15282):
        batch_index = []
        data_train = self.data_df[train_begin:train_end]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data) - self.time_step):
            if i % self.batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + self.time_step, :self.label_column]
            y = normalized_train_data[i:i + self.time_step, self.label_column, np.newaxis]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - self.time_step))
        return batch_index, train_x, train_y

    # 获取测试集
    def get_test_data(self, test_begin=15282):
        data_test = self.data_df[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std  # 标准化
        size = (len(normalized_test_data) + self.time_step - 1) // self.time_step  # 有size个sample
        test_x, test_y = [], []
        for i in range(size - 1):
            x = normalized_test_data[i * self.time_step:(i + 1) * self.time_step, :self.label_column]
            y = normalized_test_data[i * self.time_step:(i + 1) * self.time_step, self.label_column]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append((normalized_test_data[(i + 1) * self.time_step:, :self.label_column]).tolist())
        test_y.extend((normalized_test_data[(i + 1) * self.time_step:, self.label_column]).tolist())
        return mean, std, test_x, test_y

    # 生成器
    def generate_one_epoch(self):
        print(self.batch_size)
        num_batches = int(len(self.train_x)) // self.batch_size
        if self.batch_size * num_batches < len(self.train_x):
            num_batches += 1

        batch_indices = range(num_batches)
        # 每个epoch走一遍所有训练数据，并shuffle一下提供好的随机性
        # random.shuffle(): 返回随机排序后的序列
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_x = self.train_x[j * self.batch_size: (j + 1) * self.batch_size]
            batch_y = self.train_y[j * self.batch_size: (j + 1) * self.batch_size]
            # print("batch_x:", tf.shape(batch_x))
            # print("batch_y:", tf.shape(batch_y))
            assert set(map(len, batch_x)) == {self.time_step}
            # 函数是顺序执行，遇到return或最后一行函数语句就返回
            # generator在每次调用next()的时候执行，遇到yield语句就返回，再次执行时从上一次返回的yield语句处继续执行
            yield batch_x, batch_y

if __name__ == "__main__":
    s = StockDataSet(CONFIG['lstm'], 'SP500')
    s.info()
    for i in s.generate_one_epoch():
        print(i)
