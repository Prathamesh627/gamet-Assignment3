import torch
import torch.nn as nn

class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation='tanh'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        self.activation = torch.tanh if activation == 'tanh' else torch.relu

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.lin2(x)
        return x

