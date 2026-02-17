import tensorflow as tf
from src.attention import MultiHeadAttention


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)
    

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)


    def call(self, x, mask=None, training=False):

        attn_output, _ = self.mha(x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    