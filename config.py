import torch
import numpy as np


class Config(object):
    def __init__(self):
        self.num_units = 512
        self.embed_dim = 1024
        self.intent_dim = 128
        self.model_type = 'full'
        self.num_rnn = 1
        self.iter_slot = 2
        self.iter_intent = 2
        self.optimizer = 'rmsprop'
        self.batch_size = 4
        self.learning_rate = 0.001
        self.margin = 0.4
        self.downweight = 0.5
        self.max_epochs = 5
        self.patience = 40
        self.run_name = 'capsule_nlu'
        self.dataset = 'snips'
        self.model_path = './model'
        self.vocab_path = './vocab'
        self.train_data_path = 'train'
        self.test_data_path = 'test'
        self.valid_data_path = 'valid'
        self.input_file = 'seq.in'
        self.slot_file = 'seq.out'
        self.intent_file = 'label'
        self.pad_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_vocab = 0
        self.slot_size = len([x.strip() for x in open('vocab/slot_vocab').readlines()])
        self.intent_size = len([x.strip() for x in open('vocab/intent_vocab').readlines()])
        self.re_routing = True