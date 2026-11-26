from torch import nn
from src.data.vocabulary import Vocabulary

class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(  
            input_dim,
            embedding_dim,
            padding_idx=Vocabulary.PAD_ID
        )

        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        embeddings = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Masked mean pooling: compute the mean of the embeddings, ignoring padding tokens
        embedding_sum = embeddings.sum(dim=1)  # Mean pooling over the sequence length, shape: (batch_size, embedding_dim)
        counts = (x != Vocabulary.PAD_ID).sum(dim=1, keepdim=True)  # Count non-padding tokens in each sequence, shape: (batch_size, 1)
        counts = counts.clamp(min=1)  # Set zero counts to one to avoid division by zero (for sequences that are entirely padding)
        pooled = embedding_sum / counts   # Shape: (batch_size, embedding_dim)

        return self.linear(pooled)  # Shape: (batch_size, output_dim), in this case, output_dim = 2 for binary classification