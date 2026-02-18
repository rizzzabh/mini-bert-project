import os
from collections import Counter
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def parse_conll_file(filepath):
    sentences = []
    labels = []

    current_sentence = []
    current_labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split()
                token = parts[0]
                tag = parts[-1]

                current_sentence.append(token)
                current_labels.append(tag)

        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

    return sentences, labels



def build_vocab(sentences, min_freq=1):
    word_counter = Counter()

    for sentence in sentences:
        word_counter.update(sentence)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    index = 2
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = index
            index += 1

    return vocab


def build_label_dict(labels):
    unique_labels = set()

    for sentence_labels in labels:
        for label in sentence_labels:
            unique_labels.add(label)

    label2id = {"<PAD>": 0}
    id2label = {0: "<PAD>"}

    idx = 1
    for label in sorted(unique_labels):
        label2id[label] = idx
        id2label[idx] = label
        idx += 1

    return label2id, id2label


def encode_sentences(sentences, vocab):
    encoded = []

    for sentence in sentences:
        encoded_sentence = [
            vocab.get(word, vocab["<UNK>"]) for word in sentence
        ]
        encoded.append(encoded_sentence)

    return encoded


def encode_labels(labels, label2id):
    encoded = []

    for sentence_labels in labels:
        encoded_sentence = [
            label2id[label] for label in sentence_labels
        ]
        encoded.append(encoded_sentence)

    return encoded



def pad_data(X, y, max_len):

    X_padded = pad_sequences(
        X,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=0  # <PAD>
    )

    y_padded = pad_sequences(
        y,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=0
    )

    return X_padded, y_padded



def load_and_prepare_data(data_path, max_len=50, vocab=None, label2id=None):

    sentences, labels = parse_conll_file(data_path)

    avg_length = 0 
    for sentence in sentences : 
        avg_length = avg_length + (len(sentence))


    if vocab is None:
            vocab = build_vocab(sentences)

    if label2id is None:
            label2id, id2label = build_label_dict(labels)
    else:
            id2label = {idx: label for label, idx in label2id.items()}

    X = encode_sentences(sentences, vocab)
    y = encode_labels(labels, label2id)

    X_padded, y_padded = pad_data(X, y, max_len)

    return X_padded, y_padded, vocab, label2id, id2label



# TRAIN_PATH = "data/raw/train.txt"
# VALID_PATH = "data/raw/valid.txt"

# if __name__ == "__main__" : 
#     load_and_prepare_data(TRAIN_PATH , 30)