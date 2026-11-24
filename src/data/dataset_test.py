from torch.utils.data import DataLoader
from dataset import IMDbDataset
from tokenizer import *
from vocabulary import Vocabulary

train_dataset = IMDbDataset("data/aclImdb", train=True)
test_dataset = IMDbDataset("data/aclImdb", train=False)

print(len(train_dataset))  # 25000
print(len(test_dataset))   # 25000

review, label = train_dataset[0]

print(review)
print(label)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)

train_reviews, train_labels = next(iter(train_loader)) 
print(type(train_reviews))  # tuple, the default DataLoader batches the strings into a tuple of reviwes
print(type(train_labels))   # automatically converts the labels to a Tensor
print(len(train_reviews))   # 32
print(train_labels.shape)   # torch.Size([32])

print(train_reviews[0])  # prints the first sample in the training dataset

tokenizer = WhitespaceTokenizer()
vocab = Vocabulary(tokenizer, min_freq=2)
vocab.build([review for review, _ in train_dataset])  # Build vocabulary from training reviews

text = "this movie is great"
print(f"Original text: {text}")
encoded_text = vocab.encode(text)
print(f"Encoded text: {encoded_text}")
decoded_text = vocab.decode(encoded_text)
print(f"Decoded text: {decoded_text}")