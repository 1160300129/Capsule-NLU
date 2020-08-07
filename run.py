import os
from config import Config
import time
from importlib import import_module
import torch
from train_and_test import init_network, train, test

config = Config()

if config.dataset == 'snips':
    print('use snips dataset')
elif config.dataset == 'atis':
    print('use atis dataset')

model_name = 'capsule'

from utils import build_dataset, build_iterator, get_time_dif, load_vocabulary, build_vocab

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
start_time = time.time()
print('加载数据...')
build_vocab(config.input_file, os.path.join(config.vocab_path, 'in_vocab'))
build_vocab(config.slot_file, os.path.join(config.vocab_path, 'slot_vocab'))
build_vocab(config.intent_file, os.path.join(config.vocab_path, 'intent_vocab'), pad=False, unk=False)
in_vocab = load_vocabulary(os.path.join(config.vocab_path, 'in_vocab'))
slot_vocab = load_vocabulary(os.path.join(config.vocab_path, 'slot_vocab'))
intent_vocab = load_vocabulary(os.path.join(config.vocab_path, 'intent_vocab'))
train_data, dev_data, test_data = build_dataset(in_vocab['vocab'], slot_vocab['vocab'], intent_vocab['vocab'])

train_iter = build_iterator(train_data)
dev_iter = build_iterator(dev_data)
test_iter = build_iterator(test_data)
time_dif = get_time_dif(start_time)
print('time usage:', time_dif)

config.n_vocab = len(in_vocab['vocab'])

x = import_module(model_name)
model = x.Model(config).to(torch.device('cuda'))
init_network(model)
print(model.parameters)

train(config, model, train_iter, dev_iter, test_iter)
# test(config, model, test_iter)





