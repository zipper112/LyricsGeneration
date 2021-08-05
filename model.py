import torch
from torch import nn
from torch.nn.modules import dropout
import config

class LyricsGenerator(nn.Module):
    def __init__(self, embedding, voc_size, hidden_size, num_layer, drop_out):
        super(LyricsGenerator, self).__init__()
        if isinstance(embedding, int):
            self.embedding = nn.Embedding(voc_size, embedding)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        
        self.embedding_size = self.embedding.weight.shape[0]
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.LSTM = nn.LSTM(input_size=300, hidden_size=hidden_size, \
                            num_layers=num_layer, dropout=drop_out)
        
        self.to_voc = nn.Linear(hidden_size, voc_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor=None):
        x = x.transpose(0, 1)
        x = self.embedding(x)
        if h is not None:
            x, h_out = self.LSTM(x, h)
        else:
            x, h_out = self.LSTM(x)
        x = x.transpose(0, 1)
        x = self.to_voc(x)
        return x, h_out

    def __init_hidden(self, batch_size):
        return (torch.zeros(self.num_layer, batch_size, self.hidden_size),
                torch.zeros(self.num_layer, batch_size, self.hidden_size))
