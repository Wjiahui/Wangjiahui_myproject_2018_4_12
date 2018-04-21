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


CONFIG = CONFIG['lstm']
class LSTM(object):

    # weights = {
    #     'in': tf.Variable(tf.random_normal([CONFIG["input_size"], CONFIG["lstm_unit"]])),
    #     'out': tf.Variable(tf.random_normal([CONFIG["lstm_unit"], 1]))
    # }
    # biases = {
    #     'in': tf.Variable(tf.constant(0.1, shape=[CONFIG["lstm_unit"], ])),
    #     'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    # }

    def __init__(self, stock_name):
        self.logs_dir = LOG_DIR
        self.plots_dir = PLOTS_DIR

        self.input_size = CONFIG['input_size']
        self.output_size = CONFIG['output_size']


        self.time_step = CONFIG['time_step']
        self.lstm_unit = CONFIG['lstm_unit']
        self.num_layers = CONFIG['num_layers']
        self.keep_prob = CONFIG['keep_prob']

        self.batch_size = CONFIG['batch_size']
        self.init_learning_rate = CONFIG['init_learning_rate']
        self.learning_rate_decay = CONFIG['learning_rate_decay']
        self.init_epoch = CONFIG['init_epoch']
        self.max_epoch = CONFIG['max_epoch']
        self.normalized = CONFIG['normalized']

        self.stock_name = stock_name
        self.stock_dataset = StockDataSet(CONFIG, "SP500")
        self.train_x = self.stock_dataset.train_x
        self.train_y = self.stock_dataset.train_y
        self.test_x = self.stock_dataset.test_x
        self.test_y = self.stock_dataset.test_y
        self.batch_index = self.stock_dataset.batch_index
        self.mean = self.stock_dataset.mean
        self.std = self.stock_dataset.std

        # self.build_graph()

    def build_graph(self, X):
        self.batch_size = tf.shape(X)[0]
        self.time_step = tf.shape(X)[1]
        weights = {
            'in': tf.Variable(tf.random_normal([self.input_size, self.lstm_unit])),
            'out': tf.Variable(tf.random_normal([self.lstm_unit, 1]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.lstm_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }
        w_in = weights['in']
        b_in = biases['in']
        input = tf.reshape(X, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.lstm_unit])  # 将tensor转成3维，作为lstm cell的输入
        # cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_unit, reuse=tf.get_variable_scope().reuse)

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_unit,)
            if self.keep_prob < 1.0:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(CONFIG["num_layers"])],
            state_is_tuple=True
        ) if CONFIG["num_layers"] > 1 else _create_one_cell()

        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, dtype=tf.float32)
        # init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        # output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果

        output = tf.reshape(output_rnn, [-1, self.lstm_unit])  # 作为输出层的输入
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        tf.summary.histogram("w_in", w_in)
        tf.summary.histogram("b_in", b_in)
        tf.summary.histogram("w_out", w_out)
        tf.summary.histogram("b_out", b_out)
        return pred, final_states
    # def build_graph(self):
    #     self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
    #     self.inputs = tf.placeholder(tf.float32, [None, self.time_step, self.input_size], name="inputs")
    #     self.targets = tf.placeholder(tf.float32, [None, self.time_step, self.output_size], name="targets")
    #     weight_in = tf.Variable(tf.random_normal([self.input_size, self.lstm_unit]))
    #     bias_in = tf.Variable(tf.constant(0.1, shape=[self.lstm_unit, ]))
    #     input = tf.reshape(self.inputs, [-1, self.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    #     input_rnn = tf.matmul(input, weight_in) + bias_in
    #     input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.lstm_unit])  # 将tensor转成3维，作为lstm cell的输入
    #     tf.summary.histogram("weights_in", weight_in)
    #     tf.summary.histogram("biases_in", bias_in)
    #
    #     cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_unit, reuse=tf.get_variable_scope().reuse)
    #     init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
    #     # def create_one_cell():
    #     #     lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_unit, state_is_tuple=True)
    #     #     if self.keep_prob < 1.0:
    #     #         lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
    #     #     return lstm_cell
    #     #
    #     # cell = tf.contrib.rnn.MultiRNNCell(
    #     #     [create_one_cell() for _ in range(self.num_layers)],
    #     #     state_is_tuple=True
    #     # ) if self.num_layers > 1 else create_one_cell()
    #
    #     init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
    #     output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    #     output = tf.reshape(output_rnn, [-1, self.lstm_unit])  # 作为输出层的输入
    #
    #
    #     weight_out = tf.Variable(tf.random_normal([self.lstm_unit, 1]))
    #     bias_out = tf.Variable(tf.constant(0.1, shape=[1, ]))
    #     prediction = tf.matmul(output, weight_out) + bias_out
    #     tf.summary.histogram("weights_out", weight_out)
    #     tf.summary.histogram("biases_out", bias_out)
    #     print("predicton", tf.reshape(prediction, [-1]))
    #     print("targets: ", tf.reshape(self.targets, [-1]))
    #     self.loss = tf.reduce_mean(tf.square(tf.reshape(prediction, [-1]) - tf.reshape(self.targets, [-1])))
    #     self.train_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.loss)
    #     # tf.summary.scalar("loss", self.loss)
    #     #
    #     # for op in [prediction, self.loss]:
    #     #     tf.add_to_collection('ops_to_restore', op)

    # def train_lstm_graph(self):
    #     stock_dataset = StockDataSet(CONFIG, self.stock_name)
    #     learning_rates_to_use = self.compute_learning_rates()
    #     self.merged_summary = tf.summary.merge_all()
    #     self.writer = tf.summary.FileWriter('logs/', self.sess.graph)
    #     self.writer.add_graph(self.sess.graph)
    #
    #     tf.global_variables_initializer().run()
    #
    #     test_data_feed = {
    #         self.inputs: stock_dataset.test_x,
    #         self.targets: stock_dataset.test_y,
    #         self.learning_rate: 0.0
    #     }
    #
    #     for epoch_step in range(self.max_epoch):
    #         current_lr = learning_rates_to_use[epoch_step]
    #         i = 0
    #         for batch_x, batch_y in stock_dataset.generate_one_epoch():
    #             # print("batch_y shape:", tf.shape(batch_y))
    #             train_data_feed = {
    #                 self.inputs: batch_x,
    #                 self.targets: batch_y,
    #                 self.learning_rate: current_lr
    #             }
    #             i += 1
    #             print(i)
    #             train_loss, _ = self.sess.run([self.train_op, self.loss], train_data_feed)
    #
    #         if epoch_step % 10 == 0:
    #             test_loss, _pred, _summary = self.sess.run([self.loss, self.prediction, self.merged_summary], test_data_feed)
    #             assert len(_pred) == len(stock_dataset.test_y)
    #             print "Epoch %d [%f]:" % (epoch_step, current_lr), test_loss
    #             if epoch_step % 50 == 0:
    #                 print "Predictions:", [(
    #                     map(lambda x: round(x, 4), _pred[-j]),
    #                     map(lambda x: round(x, 4), stock_dataset.test_y[-j])
    #                 ) for j in range(5)]
    #
    #         self.writer.add_summary(_summary, global_step=epoch_step)
    #
    #     print "Final Results:"
    #     final_prediction, final_loss = self.sess.run([self.prediction, self.loss], test_data_feed)
    #     print final_prediction, final_loss
    #
    #
    #
    #
    #     graph_name = "wang"
    #     graph_saver_dir = os.path.join(MODEL_DIR, graph_name)
    #     if not os.path.exists(graph_saver_dir):
    #         os.mkdir(graph_saver_dir)
    #
    #     saver = tf.train.Saver()
    #     saver.save(sess, os.path.join(
    #         graph_saver_dir, "stock_rnn_model_%s.ckpt" % graph_name), global_step=epoch_step)
    #
    #     with open("final_predictions.{}.json".format(graph_name), 'w') as fout:
    #         fout.write(json.dumps(final_prediction.tolist()))
    def train_lstm(self):
        graph_name = "stock_name:%s_lr%.5f_lr_decay%.5f_unit%d_layer%d_step%d_input%d_batch%d_epoch%d" % (
            self.stock_name,
            self.init_learning_rate, self.learning_rate_decay,
            self.lstm_unit, self.num_layers, self.time_step,
            self.input_size, self.batch_size, self.max_epoch)


        X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
        Y = tf.placeholder(tf.float32, shape=[None, self.time_step, self.output_size])
        pred, _ = self.build_graph(X)
        # 损失函数

        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        tf.summary.scalar("loss", loss)
        train_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        graph_saver_dir = os.path.join(MODEL_DIR, graph_name)
        if not os.path.exists(graph_saver_dir):
            os.makedirs(graph_saver_dir)
        # module_file = tf.train.latest_checkpoint()
        merged_summary = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.logs_dir, sess.graph)
            writer.add_graph(sess.graph)

            graph = tf.get_default_graph()
            sess.run(tf.global_variables_initializer())
            # saver.restore(self.sess, module_file)
            # 重复训练2000次
            for i in range(2000):
                for batch_x, batch_y in self.stock_dataset.generate_one_epoch():
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: batch_x,
                                                   Y: batch_y})
                print(i, loss_)
                if i % 200 == 0:
                    print("保存模型：", saver.save(sess, os.path.join(graph_saver_dir, "lstm_model_%s.ckpt" % graph_name), global_step=i))

    def compute_learning_rates(self):
        learning_rates_to_use = [self.init_learning_rate * (self.learning_rate_decay ** max(float(i + 1 - self.init_epoch), 0.0)) for i in range(self.max_epoch)]
        print "Middle learning rate:", learning_rates_to_use[len(learning_rates_to_use) // 2]
        return learning_rates_to_use


if __name__ == '__main__':
    lstm = LSTM("SP500")
    lstm.train_lstm()


