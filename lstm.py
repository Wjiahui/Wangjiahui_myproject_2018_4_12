# -*-coding=utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from data_model import *

class LSTM(object):
    def __init__(self, config, X):
        self.init_learning_rate = config['init_learning_rate']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        dataset = StockDataSet(CONFIG['lstm'], 'SP500')
        self.train_x, self.train_y, self.test_x, self.test_y, self.batch_index = dataset.prepare_data()


    def build_graph(self, config, X):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        input_size = config['input_size']
        lstm_unit = config['lstm_unit']
        tf.reset_default_graph()
        lstm_graph = tf.Graph()
        with lstm_graph.as_default():
            learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
            inputs = tf.placeholder(tf.float32, [None, config.time_step, config.input_size], name="inputs")
            targets = tf.placeholder(tf.float32, [None, config.output_size], name="targets")

            # 输入层
            with tf.name_scope("input_layer"):
                weight = tf.Variable(tf.random_normal([input_size, lstm_unit]))
                bias = tf.Variable(tf.constant(0.1, shape=[lstm_unit, ]))
                input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
                input_rnn = tf.matmul(input, weight) + bias
                input_rnn = tf.reshape(input_rnn, [-1, time_step, lstm_unit])  # 将tensor转成3维，作为lstm cell的输入
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", bias)

            def create_one_cell():
                lstm_cell = tf.tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
                if config.keep_prob < 1.0:
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
                return lstm_cell

            cell = tf.contrib.rnn.MultiRNNCell(
                [create_one_cell() for _ in range(config.num_layers)],
                state_is_tuple=True
            ) if config.num_layers > 1 else create_one_cell()

            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32,
                                                         scope="lstm")
            output = tf.reshape(output_rnn, [-1, lstm_unit])  # 作为输出层的输入

            # 输出层
            with tf.name_scope("output_layer"):
                weight = tf.Variable(tf.random_normal([lstm_unit, 1]))
                bias = tf.Variable(tf.constant(0.1, shape=[1, ]))
                prediction = tf.matmul(output, weight) + bias
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", bias)

            return prediction, final_states

    def train(self, batch_size=80, time_step=15, train_begin=0, train_end=5800):
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, self.output_size])


        prediction, _ = self.build_graph(X)
        # 损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(prediction, [-1]) - tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        module_file = tf.train.latest_checkpoint()
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, module_file)
            # 重复训练2000次
            for i in range(2000):
                for step in range(len(batch_index) - 1):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print(i, loss_)
                if i % 200 == 0:
                    print("保存模型：", saver.save(sess, 'stock2.model', global_step=i))