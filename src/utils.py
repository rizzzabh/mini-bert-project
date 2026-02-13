import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
import numpy as np


class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True  
        )

        self.pos_emb = Embedding(
            input_dim=max_len,
            output_dim=embed_dim
        )

        self.max_len = max_len

    def call(self, x):
        seq_len = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)

        x = self.token_emb(x)

        return x + positions
    



def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]
