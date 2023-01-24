"""Prepare Tiny Shakespeare dataset.

From https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
by Andrej Karpathy.

This downloads the dataset, creates the vocabulary, converts characters into tokens,
creates training/validation splits, and then saves the results to disk.

Length of dataset in characters: 1,115,394
Unique characters:
[
    '\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C',
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]
Vocab size: 65
Training dataset has 1,003,854 tokens
Validation dataset has 111,540 tokens

"""

import os
import pickle
import requests
import numpy as np


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# Download the tiny shakespeare dataset.
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    with open(input_file_path, "w") as f:
        f.write(requests.get(DATA_URL).text)

with open(input_file_path, "r") as f:
    data = f.read()
print(f"Length of dataset in characters: {len(data):,}")

# Get all the unique characters that occur in this text.
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Unique characters: {chars}")
print(f"Vocab size: {vocab_size:,}")

# Create a mapping from characters to integers.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # Encoder: take a string, output a list of integers


def decode(l):
    "".join([itos[i] for i in l])  # Decoder: take a list of integers, output a string


# Create the train and test splits.
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Encode both to integers.
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train dataset has {len(train_ids):,} tokens")
print(f"Validation dataset has {len(val_ids):,} tokens")

# Export to bin files.
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# Save the meta information as well, to help us encode/decode later.
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
