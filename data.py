import os
import random

from tqdm import tqdm


def get_files_size(files):
    total_size = 0
    for file in files:
        total_size += os.path.getsize(file)
    return total_size


def files_train_test_split(input_files_list: list, test_size):
    """Splits files into train and test sets by bytes size"""
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


def loadfile(filename):
    result = ''
    with open(filename, 'r') as f:
        for line in f:
            result += line
    return result


def get_array_of_words(files_list):
    texts = []
    print("Reading files")
    for file in files_list:
        texts.append(loadfile(file))
    words = []
    for text in tqdm(texts, total=len(texts)):
        words += text.split()
    return words


def markup_and_save_data(words, out_file, punctuation_signs: dict):
    with open(out_file, 'w') as out:
        print("Writing " + out_file)
        for word in tqdm(words, total=len(words)):
            if word[-1] in punctuation_signs.keys():
                out.write(word[:-1] + '\t' + punctuation_signs[word[-1]] + '\n')
            else:
                out.write(word + '\t' + '0' + '\n')


if __name__ == '__main__':
    random.seed(1)
    files = os.listdir('./data/raw')
    for i in range(len(files)):
        files[i] = './data/raw/' + files[i]

    dataset = files_train_test_split(files, 0.3)
    # print(get_files_size(dataset["train"]) / get_files_size(files))
    # print(get_files_size(dataset["test"]) / get_files_size(files))
    punctuation = {',': "COMMA", '.': "PERIOD"}
    markup_and_save_data(get_array_of_words(dataset['train']), './data/train.tsv', punctuation)
    markup_and_save_data(get_array_of_words(dataset['test']), './data/test.tsv', punctuation)
