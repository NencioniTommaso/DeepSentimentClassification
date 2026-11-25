import torch

def pad_or_truncate(sequence, max_length, pad_id):
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    
    if len(sequence) < max_length:
        return sequence + [pad_id] * (max_length - len(sequence))
    else:
        return sequence[:max_length]

def collate_fn(batch, vocab, max_length=None):      # If max_length is None, it will be set to the length of the longest sequence in the batch
    sequences, labels = zip(*batch)
    encoded_sequences = [vocab.encode(seq) for seq in sequences]

    if max_length is None:
        max_length = max(len(seq) for seq in encoded_sequences)

    padded_sequences = [pad_or_truncate(seq, max_length, vocab.PAD_ID) for seq in encoded_sequences]
    return (
        torch.tensor(padded_sequences, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )

def bow_collate_fn(batch, vocab):       # Bag-of-Words (BoW) collate function for batching reviews and labels.
    reviews, labels = zip(*batch)
    feature_vectors = []

    for review in reviews:
        token_ids = vocab.encode(review)

        # Count each token ID. The result has one entry per vocabulary item.
        counts = torch.bincount(
            torch.tensor(token_ids, dtype=torch.long),
            minlength=len(vocab)
        ).to(torch.float32)

        # PAD is not part of a raw review's BoW representation.
        counts[vocab.PAD_ID] = 0
        feature_vectors.append(counts)

    inputs = torch.stack(feature_vectors)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels