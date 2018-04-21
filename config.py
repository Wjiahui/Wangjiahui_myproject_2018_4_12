import codecs
import yaml


CONFIG = yaml.load(codecs.open('rnn', 'r', 'utf8'))


# class RNNConfig():
#     input_size = 1
#     num_steps = 30
#     lstm_size = 128
#     num_layers = 1
#     keep_prob = 0.8
#
#     batch_size = 64
#     init_learning_rate = 0.006
#     learning_rate_decay = 0.99
#     init_epoch = 5
#     max_epoch = 50

#DEFAULT_CONFIG = RNNConfig()
#print "Default configuration:", DEFAULT_CONFIG.to_dict()

DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_DIR = "models"
PLOTS_DIR = "images"
