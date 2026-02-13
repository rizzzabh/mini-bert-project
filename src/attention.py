import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class SelfAttention(Layer):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()

        self.embed_dim = embed_dim

        self.W_q = Dense(embed_dim)
        self.W_k = Dense(embed_dim)
        self.W_v = Dense(embed_dim)

    def call(self, x, mask=None):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = tf.matmul(Q, K, transpose_b=True)

 
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = scores / tf.math.sqrt(dk)

        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights