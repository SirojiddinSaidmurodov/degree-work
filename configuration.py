import os
from datetime import datetime

import torch


class Configuration:
    def __init__(self, punctuation_names: dict, punctuation_encoding: dict, segment_size: int, epochs=5, iterations=2):
        self.punctuation_names = punctuation_names
        self.punctuation_encoding = punctuation_encoding
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flavor = 'models/tatbert/'
        self.segment_size = segment_size
        self.save_path = 'models/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.mkdir(self.save_path)
        self.epochs = epochs
        self.iterations = iterations
