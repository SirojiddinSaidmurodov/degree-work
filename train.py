import os
from datetime import datetime

import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim
from tqdm import tqdm
from transformers import BertTokenizer

import data
from configuration import Configuration
from data import create_data_loader
from model import PuncRec


def validate(model, criterion, epoch, epochs, iteration, iterations, data_loader_valid, save_path, train_loss,
             best_val_loss, best_model_path, punctuation_enc):
    val_losses = []
    val_accs = []
    val_f1s = []

    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())

    for inputs, labels in tqdm(data_loader_valid, total=len(data_loader_valid)):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = output.argmax(dim=1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)

    improved = ''

    # model_path = '{}model_{:02d}{:02d}'.format(save_path, epoch, iteration)
    model_path = save_path + 'model'
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        improved = '*'
        best_val_loss = val_loss
        best_model_path = model_path

    f1_cols = ';'.join(['f1_' + key for key in label_keys])

    progress_path = save_path + 'progress.csv'
    if not os.path.isfile(progress_path):
        with open(progress_path, 'w') as f:
            f.write('time;epoch;iteration;training loss;loss;accuracy;' + f1_cols + '\n')

    f1_vals = ';'.join(['{:.4f}'.format(val) for val in val_f1])

    with open(progress_path, 'a') as f:
        f.write('{};{};{};{:.4f};{:.4f};{:.4f};{}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            iteration,
            train_loss,
            val_loss,
            val_acc,
            f1_vals
        ))

    print("Epoch: {}/{}".format(epoch + 1, epochs),
          "Iteration: {}/{}".format(iteration, iterations),
          "Loss: {:.4f}".format(train_loss),
          "Val Loss: {:.4f}".format(val_loss),
          "Acc: {:.4f}".format(val_acc),
          "F1: {}".format(f1_vals),
          improved)

    return best_val_loss, best_model_path


def train(model, optimizer, criterion, configuration: Configuration, data_loader_train, data_loader_valid,
          best_val_loss=1e9):
    print_every = len(data_loader_train) // configuration.iterations + 1
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(configuration.epochs):

        counter = 1
        iteration = 1

        for inputs, labels in data_loader_train:

            inputs, labels = inputs.cuda(), labels.cuda()
            inputs.requires_grad = False
            labels.requires_grad = False
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:
                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, criterion, e, configuration.epochs, iteration,
                                                          configuration.iterations,
                                                          data_loader_valid,
                                                          configuration.save_path, train_loss, best_val_loss,
                                                          best_model_path,
                                                          configuration.punctuation_encoding)
                model.train()
                pbar = tqdm(total=print_every)
                iteration += 1

            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path = validate(model, criterion, e, configuration.epochs, iteration,
                                                  configuration.iterations,
                                                  data_loader_valid,
                                                  configuration.save_path, train_loss, best_val_loss, best_model_path,
                                                  configuration.punctuation_encoding)
        model.train()
        if e < configuration.epochs - 1:
            pbar = tqdm(total=print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss


if __name__ == '__main__':
    punctuation = {',': "COMMA", '.': "PERIOD", ' ': '0'}
    encoding = {'0': 0, 'PERIOD': 1, 'COMMA': 2}

    config = Configuration(punctuation_names=punctuation,
                           punctuation_encoding=encoding,
                           segment_size=32,
                           epochs=5,
                           iterations=2)

    tokenizer = BertTokenizer.from_pretrained(config.flavor, do_lower_case=True)
    train_data = data.load_data('./data/train.tsv')
    test_data = data.load_data('./data/test.tsv')
    X_train, y_train = data.preprocess_data(train_data, tokenizer, encoding, config.segment_size)
    X_test, y_test = data.preprocess_data(test_data, tokenizer, encoding, config.segment_size)

    puncRec = nn.DataParallel(PuncRec(config).to(config.device))

    data_loader_train = create_data_loader(X_train, y_train, True, 1024)
    data_loader_valid = create_data_loader(X_test, y_test, False, 1024)
    for p in puncRec.module.bert.parameters():
        p.requires_grad = False
    optim = optim.Adam(puncRec.parameters(), lr=1e-5)
    loss = nn.CrossEntropyLoss()
    puncRec, optimizer, best_val_loss = train(puncRec, optim, loss, config,
                                              data_loader_train, data_loader_valid, best_val_loss=1e9)
