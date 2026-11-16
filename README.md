# Sequence Learning with PyTorch

A hands-on project to study and implement neural network architectures for sequence processing using PyTorch.

## Overview

The goal of this project is to develop a practical understanding of sequence processing models by implementing them from scratch, using PyTorch.

The project focuses on Natural Language Processing (NLP), using sentiment classification as the main task. Starting from a simple embedding-based classifier, different neural architectures will be progressively implemented, trained and compared.

Rather than relying on high-level pre-built architectures, the objective is to understand the underlying operations, computational graphs, tensor dimensions and mathematical principles behind each model.

## Learning Objectives

The project is intended to explore the following topics:

- **Word Embeddings:** Vocabulary construction, tokenization, embedding layers and padding.
- **Recurrent Neural Networks:** Vanilla RNNs, hidden states, computational graphs and stacked architectures.
- **Gated Architectures:** GRU and LSTM, their internal mechanisms and their ability to retain long-term information.
- **Bidirectional RNNs:** Processing sequences in both forward and backward directions.
- **Attention Mechanisms:** Attention scores, softmax normalization, context vectors and attention-based sequence classification.
- **Transformers:** Self-attention, scaled dot-product attention, multi-head attention and positional encoding.
- **Training and Optimization:** Backpropagation through time, vanishing and exploding gradients, loss functions and optimization algorithms.

## Project Roadmap

The project will be developed incrementally, with each stage building on the previous one.

| Stage | Implementation |
|---|---|
| 1 | Data preprocessing and embedding-based classifier |
| 2 | Vanilla RNN |
| 3 | GRU and LSTM |
| 4 | Stacked and bidirectional RNNs |
| 5 | Attention-based recurrent classifier |
| 6 | Self-attention and multi-head attention |
| 7 | Transformer encoder |
| 8 | Model evaluation and comparison |

Additional experiments may include recurrent language models, encoder-decoder architectures and multi-instance learning.

## Main Task

The primary task is binary sentiment classification using the IMDb movie reviews dataset.

Given a review, the model must predict whether its sentiment is positive or negative.

The same task and dataset will be used across the different architectures to make their performances comparable.

## Methodology

The project follows a few principles:

1. Implement the fundamental operations manually using PyTorch tensor operations and basic neural network layers.
2. Avoid relying on high-level modules that hide the internal mechanisms being studied.
3. Test individual components before integrating them into complete models.
4. Keep training and evaluation procedures consistent across experiments.
5. Compare architectures in terms of performance, parameter count, training behaviour and computational complexity.
6. Document relevant observations, limitations and conclusions.

PyTorch documentation and other learning resources may be used as references, but the implementations will be developed independently.

## Repository Structure

The planned repository structure is:

```text
sequence-learning-pytorch/
│
├── data/
│
├── models/
│   ├── baseline.py
│   ├── rnn.py
│   ├── gru.py
│   ├── lstm.py
│   ├── attention.py
│   └── transformer.py
│
├── training/
│
├── experiments/
│
├── notebooks/
│
├── README.md
└── requirements.txt
```

The structure may evolve as the project develops.

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Project Status

**In development.**

Currently at the initial setup stage. Implementations, experiments and results will be added progressively.

## References

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [Dive into Deep Learning](https://d2l.ai/)
