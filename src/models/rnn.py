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


class MultiLayerRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=2):
        super(MultiLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.W = nn.ModuleList([                        # each layer has its own W and U matrices, so we use ModuleList to store them
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_layers)
        ])

        self.U = nn.ModuleList([
            nn.Linear(embedding_size if layer == 0 else hidden_size, hidden_size)   # the first layer takes the embedding as input, subsequent layers take the hidden state of the previous layer as input
            for layer in range(num_layers)
        ])

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        e = self.embedding(x)

        #h = e.new_zeros(batch_size, seq_length, self.hidden_size)  # hidden state h is a tensor of shape (batch_size, seq_length, hidden_size) since we now need to save all the hidden states of the current layer and not only the last state
        # another way of seeing it is as a tensor of shape (batch_size, hidden_size) that will be updated at each time step, but we will need to keep track of the hidden states of the previous layer for each time step

        layer_input = e  # the input to the first layer is the embedding, for subsequent layers it will be the hidden states of the previous layer

        for layer in range(self.num_layers):
            hidden = e.new_zeros(batch_size, self.hidden_size)      # This will go across the sequence length, so we need to initialize it for each layer, and it will be updated at each time step
            h = e.new_zeros(batch_size, seq_length, self.hidden_size)   # This will store the hidden states of the current layer for each time step, so that we can use it as input to the next layer

            for t in range(seq_length):
                current_input = layer_input[:, t, :]

                # Apply padding mask and update hidden, meaning this is the standard RNN update equation across a single layer
                hidden = torch.where(
                    (x[:, t] != Vocabulary.PAD_ID).unsqueeze(1),
                    torch.tanh(self.W[layer](hidden) + self.U[layer](current_input)),
                    hidden
                )

                # Store hidden in h[:, t, :], we save each hidden state for each time step in the sequence, so that we can use it as input to the next layer
                h[:, t, :] = hidden

            layer_input = h     # Next layer's input is the hidden states of the current layer

        return self.linear(layer_input[:, -1, :])       # We classify based on the last hidden state of the last layer, which is a common practice in sequence classification tasks
        