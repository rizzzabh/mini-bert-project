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
            mask = tf.cast(mask, tf.float32)
            scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights
    

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads


        self.W_q = tf.keras.layers.Dense(embed_dim)
        self.W_k = tf.keras.layers.Dense(embed_dim)
        self.W_v = tf.keras.layers.Dense(embed_dim)

        self.W_o = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        """
        x shape: (batch_size, seq_len, embed_dim)
        returns: (batch_size, num_heads, seq_len, head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, Q, K, V, mask):

        scores = tf.matmul(Q, K, transpose_b=True)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = scores / tf.math.sqrt(dk)

        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights
    

    def call(self, x, mask=None):

      batch_size = tf.shape(x)[0]

      Q = self.W_q(x)
      K = self.W_k(x)
      V = self.W_v(x)

      Q = self.split_heads(Q, batch_size)
      K = self.split_heads(K, batch_size)
      V = self.split_heads(V, batch_size)

      if mask is not None:
            mask = tf.cast(mask, tf.float32)

      attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
      )

      attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

      concat_attention = tf.reshape(
            attention_output,
            (batch_size, -1, self.embed_dim)
      )

      output = self.W_o(concat_attention)

      return output, attention_weights