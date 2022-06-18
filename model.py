import torch.nn.functional as F
from torch import nn
from transformers import BertModel

from configuration import Configuration


class PuncRec(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.flavor)
        self.hl = nn.Linear(768, 768)
        self.punc = nn.Linear(768, len(config.punctuation_names.keys()))
        self.dropout = nn.Dropout(0.2)
        self.to(config.device)

    def forward(self, x):
        output = self.bert(x)
        representations = self.dropout(F.gelu(output['last_hidden_state']))
        punc = self.punc(self.dropout(self.hl(representations)))
        return punc
