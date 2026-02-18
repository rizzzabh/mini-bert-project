import tensorflow as tf
from src.encoder import TransformerEncoderBlock



class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=max_len,
            output_dim=embed_dim
        )

    def call(self, x):
        max_len = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.position_embedding(positions)

        token_embeddings = self.token_embedding(x)

        return token_embeddings + positions


class MiniBERT(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        num_labels,
        dropout_rate=0.1,
    ):
        super(MiniBERT, self).__init__()

        self.embedding = EmbeddingLayer(vocab_size, embed_dim, max_len)

        self.encoder_layers = [
            TransformerEncoderBlock(
                embed_dim,
                num_heads,
                ff_dim,
                dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)


        self.classifier = tf.keras.layers.Dense(num_labels)

    def call(self, x, training=False):

      mask = tf.cast(tf.math.equal(x, 0), tf.float32)
      mask = mask[:, tf.newaxis, tf.newaxis, :]

      x = self.embedding(x)
      x = self.dropout(x, training=training)

      for encoder in self.encoder_layers:
            x = encoder(x, mask=mask, training=training)

      logits = self.classifier(x)

      return logits
    
