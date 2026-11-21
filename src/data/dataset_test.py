from dataset import IMDbDataset

train_dataset = IMDbDataset("data/aclImdb", train=True)
test_dataset = IMDbDataset("data/aclImdb", train=False)

print(len(train_dataset))  # 25000
print(len(test_dataset))   # 25000

review, label = train_dataset[0]

print(review)
print(label)
