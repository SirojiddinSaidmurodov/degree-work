import multiprocessing

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn, optim

import data
from configuration import Configuration
from data import create_data_loader
from model import PuncRec
from train_funcs import train
from visualize import visualize_results

if __name__ == '__main__':
    torch.set_num_threads(multiprocessing.cpu_count())

    config = Configuration(segment_size=32,
                           top_epochs=1,
                           top_iterations=2,
                           all_epochs=1,
                           all_iterations=2,
                           smoke_run=True)
    tokenizer = BertWordPieceTokenizer(config.flavor + '/vocab.txt', lowercase=True)
    train_data = data.load_data(config.train_path)
    test_data = data.load_data(config.test_path)
    X_train, y_train = data.preprocess_data(train_data, tokenizer, config.punctuation_encoding, config.segment_size)
    X_test, y_test = data.preprocess_data(test_data, tokenizer, config.punctuation_encoding, config.segment_size)

    puncRec = nn.DataParallel(PuncRec(config).to(config.device))

    print('TRAINING TOP LAYER...')
    data_loader_train = create_data_loader(X_train, y_train, True, 1024)
    data_loader_valid = create_data_loader(X_test, y_test, False, 512)
    for p in puncRec.module.bert.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(puncRec.parameters(), lr=1e-5)
    loss = nn.CrossEntropyLoss()
    puncRec, optimizer, best_val_loss = train(puncRec, optimizer, loss, config, data_loader_train, data_loader_valid,
                                              best_val_loss=1e9, top_learning=True)

    print('TRAINING ALL LAYERS...')
    data_loader_train = create_data_loader(X_train, y_train, True, 256)
    data_loader_valid = create_data_loader(X_test, y_test, False, 128)
    for p in puncRec.module.bert.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(puncRec.parameters(), lr=1e-5)
    loss = nn.CrossEntropyLoss()
    bert_punc, optimizer, best_val_loss = train(puncRec, optimizer, loss, config,
                                                data_loader_train, data_loader_valid, best_val_loss=best_val_loss)

    visualize_results(config.save_path, save_file=True)
