import os
from datetime import datetime

import torch

punctuation = {',': 'COMMA', '.': 'PERIOD', ' ': '0'}
encoding = {'0': 0, 'PERIOD': 1, 'COMMA': 2}


class Configuration:
    def __init__(self, punctuation_names=None, punctuation_encoding=None, segment_size: int = 32,
                 top_epochs=5, top_iterations=2, all_epochs=2, all_iterations=2, smoke_run=False):
        if punctuation_encoding is None:
            punctuation_encoding = encoding
        if punctuation_names is None:
            punctuation_names = punctuation

        if smoke_run:
            self.train_path = './data/smoke/train.tsv'
            self.test_path = './data/smoke/test.tsv'
        else:
            self.train_path = './data/train.tsv'
            self.test_path = './data/test.tsv'

        self.punctuation_names = punctuation_names
        self.punctuation_encoding = punctuation_encoding
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flavor = 'models/tatbert/'
        self.segment_size = segment_size
        self.save_path = 'models/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.mkdir(self.save_path)
        self.top_epochs = top_epochs
        self.top_iterations = top_iterations
        self.all_epochs = all_epochs
        self.all_iterations = all_iterations

#       TODO: add presets (test_run, prod_run)
