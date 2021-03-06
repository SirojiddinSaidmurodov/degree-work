import io
import multiprocessing
import os
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm


def get_files_size(files):
    total_size = 0
    for file in files:
        total_size += os.path.getsize(file)
    return total_size


def files_train_test_split(input_files_list: list, test_size):
    """
    Splits files into train and test sets by bytes size
    :param input_files_list: list of paths ot files
    :param test_size: test dataset ratio
    :return: dict of lists, train and test
    """
    files_list = input_files_list.copy()

    total_size = get_files_size(files_list)

    test_files_bytes = 0
    test_files = []
    while test_files_bytes < total_size * test_size:
        file = random.choice(files_list)
        test_files_bytes += os.path.getsize(file)
        files_list.remove(file)
        test_files.append(file)
    return {"train": files_list, "test": test_files}


def load_file(filename) -> str:
    """
    :param filename: path to file
    :return: string with content of file
    """
    result = ''
    with io.open(filename, encoding='utf-8') as f:
        for line in f:
            result += line
    return result


def load_data(filename) -> List[str]:
    """

    :param filename: path to file
    :return: list of string from file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        result = f.readlines()
    return result


def get_array_of_words(files_list):
    texts = []
    print("Reading files")
    for file in files_list:
        texts.append(load_file(file))
    words = []
    for text in tqdm(texts, total=len(texts)):
        words += text.split()
    return words


def markup_and_save_data(words, out_file, punctuation_signs: dict):
    with open(out_file, mode='w', encoding='utf-8') as out:
        print("Writing " + out_file)
        for word in tqdm(words, total=len(words)):
            if word[-1] in punctuation_signs.keys():
                out.write(word[:-1].lower() + '\t' + punctuation_signs[word[-1]] + '\n')
            else:
                out.write(word.lower() + '\t' + '0' + '\n')


def stats_for_cleaning(files_list):
    texts = ''
    print("Reading files")
    for file in files_list:
        texts += load_file(file)
    symbols = {}
    texts = texts.lower()
    for symbol in tqdm(texts, total=len(texts)):
        if symbol in symbols.keys():
            symbols[symbol] += 1
        else:
            symbols[symbol] = 1
    return dict(sorted(symbols.items(), key=lambda item: item[1], reverse=True))


def encode_data(data, tokenizer, punctuation_enc: dict):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    print('Tokenizing')
    for line in tqdm(data, total=len(data)):
        word, punc = line.split('\t')
        punc = punc.strip()
        # tokens = tokenizer.tokenize(word)
        # x = tokenizer.convert_tokens_to_ids(tokens)
        x = tokenizer.encode(word).ids
        y = [0] * 3
        y[punctuation_enc[punc]] = 1
        y = [y]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x) - 1) * [[1, 0, 0]] + y
            X += x
            Y += y
    return X, Y


def insert_target(x, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.
    """
    print('Inserting target')
    X = []
    x_pad = x[-((segment_size - 1) // 2 - 1):] + x + x[:segment_size // 2]
    index = range(len(x_pad) - segment_size + 2)
    for i in tqdm(index, total=len(index)):
        segment = x_pad[i:i + segment_size - 1]
        segment.insert((segment_size - 1) // 2, 0)
        X.append(segment)

    return np.array(X)


def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    X, y = encode_data(data, tokenizer, punctuation_enc)
    X = insert_target(X, segment_size)
    return X, y


def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=multiprocessing.cpu_count())
    return data_loader


def split_train_test(smoke_data, test_size):
    test = smoke_data[:int(-len(smoke_data) * test_size)]
    train = smoke_data[:int(len(smoke_data) * test_size)]
    return train, test


if __name__ == '__main__':
    random.seed(1)
    files = os.listdir('./data/raw')
    for i in range(len(files)):
        files[i] = './data/raw/' + files[i]

    # stat = stats_for_cleaning(files)
    # for key in stat.keys():
    #     print(key + " " + str(stat[key]))

    dataset = files_train_test_split(files, 0.3)
    # print(get_files_size(dataset["train"]) / get_files_size(files))
    # print(get_files_size(dataset["test"]) / get_files_size(files))
    punctuation = {',': "COMMA", '.': "PERIOD", ' ': '0'}
    markup_and_save_data(get_array_of_words(dataset['train']), './data/train.tsv', punctuation)
    markup_and_save_data(get_array_of_words(dataset['test']), './data/test.tsv', punctuation)

    smoke_train_data, smoke_test_data = split_train_test(get_array_of_words(['data/raw/2_88_1_0.doc.txt']), 0.3)
    markup_and_save_data(smoke_train_data, './data/smoke/train.tsv', punctuation)
    markup_and_save_data(smoke_test_data, './data/smoke/test.tsv', punctuation)
