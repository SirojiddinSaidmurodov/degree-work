import multiprocessing

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn, optim

import data
from configuration import Configuration
from data import create_data_loader
from model import PuncRec
from train_funcs import train

if __name__ == '__main__':
    torch.set_num_threads(multiprocessing.cpu_count())
    punctuation = {',': 'COMMA', '.': 'PERIOD', ' ': '0'}
    encoding = {'0': 0, 'PERIOD': 1, 'COMMA': 2}

    config = Configuration(punctuation_names=punctuation,
                           punctuation_encoding=encoding,
                           segment_size=32,
                           epochs=5,
                           iterations=2)
    tokenizer = BertWordPieceTokenizer(config.flavor + '/vocab.txt', lowercase=True)
    train_data = data.load_data('./data/train.tsv')
    test_data = data.load_data('./data/test.tsv')
    X_train, y_train = data.preprocess_data(train_data, tokenizer, encoding, config.segment_size)
    X_test, y_test = data.preprocess_data(test_data, tokenizer, encoding, config.segment_size)

    puncRec = nn.DataParallel(PuncRec(config).to(config.device))

    data_loader_train = create_data_loader(X_train, y_train, True, 1024)
    data_loader_valid = create_data_loader(X_test, y_test, False, 128)
    for p in puncRec.module.bert.parameters():
        p.requires_grad = False
    optim = optim.Adam(puncRec.parameters(), lr=1e-5)
    loss = nn.CrossEntropyLoss()
    puncRec, optimizer, best_val_loss = train(puncRec, optim, loss, config, data_loader_train, data_loader_valid,
                                              best_val_loss=1e9)
