# -*-coding=utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import random
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector

from data_model import *
from config import *
from lstm import LSTM

CONFIG = CONFIG['lstm']
#————————————————预测模型————————————————————
def prediction():
    label_column = 6
    X = tf.placeholder(tf.float32, shape=[None, CONFIG["time_step"], CONFIG['input_size']])
    stock_dataset = StockDataSet(CONFIG, "SP500")
    test_x = stock_dataset.test_x
    test_y = stock_dataset.test_y
    mean = stock_dataset.mean
    std = stock_dataset.std
    lstm = LSTM("SP500")
    pred, _ = lstm.build_graph(X)
    # ckpt = tf.train.get_checkpoint_state('models/stock_name:SP500_lr0.00_lr_decay0.990_lstm10_step1_input6_batch60_epoch50')  # 通过检查点文件锁定最新的模型
    # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:

        #参数恢复
        model_file = tf.train.latest_checkpoint('models/stock_name:SP500_lr0.00_lr_decay0.990_lstm10_step20_input6_batch64_epoch50')
        saver.restore(sess, model_file)
        test_predict = []
        for step in range(len(test_x)-1):
          prob = sess.run(pred, feed_dict={X: [test_x[step]]})
          predict = prob.reshape((-1))
          test_predict.extend(predict)
        test_y = np.array(test_y) * std[label_column]+mean[label_column]
        test_predict = np.array(test_predict) * std[label_column]+mean[label_column]
        acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        print(acc)
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction()
