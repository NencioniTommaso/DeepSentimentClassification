import torch
from torch import nn
from src.data.vocabulary import Vocabulary

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # self.num_layers = num_layers  # will be used for stacked RNNs, but for now we will keep it simple with a single layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)          # x is a tensor of shape (batch_size, seq_length)
        seq_length = x.size(1)

        e = self.embedding(x)           # e is a tensor of shape (batch_size, seq_length, embedding_size)

        h = e.new_zeros(batch_size, self.hidden_size)    # hidden state h is a tensor of shape (batch_size, hidden_size), this way initializes h to the same dtype and device as e

        for t in range(seq_length):             # iterate over the sequence length
            current_embedding = e[:, t, :]

            # Problem: when iterating over a sequence we need to skip PAD tokens, which are represented by 0 in the input tensor. We can do this by checking if the current token is a PAD token and skipping the update of the hidden state if it is.
            h = torch.where(
                (x[:, t] != Vocabulary.PAD_ID).unsqueeze(1),  # Check if the current token is not a PAD token, unsqueeze to match the shape of h, explicit broadcasting making it clear that we want to apply the condition to each element in the batch
                torch.tanh(self.W(h) + self.U(current_embedding)),
                h
            )

        return self.linear(h)


        