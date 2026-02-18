import tensorflow as tf
from src.preprocessing import load_and_prepare_data
from src.model import MiniBERT

TRAIN_PATH = "data/raw/train.txt"
VALID_PATH = "data/raw/valid.txt"

MAX_LEN = 30
EMBED_DIM = 128

VOCAB_SIZE = 23626
NUM_LABELS = 9


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none"
)

def masked_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    y_pred = tf.cast(y_pred, y_true.dtype)

    
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)

    matches *= mask

    return tf.reduce_sum(matches) / tf.reduce_sum(mask)



def train():

    X_train, y_train, vocab, label2id, id2label = load_and_prepare_data(TRAIN_PATH, MAX_LEN)

    X_valid, y_valid, _, _, _ = load_and_prepare_data(
      VALID_PATH,
      MAX_LEN,
      vocab=vocab,
      label2id=label2id
      )

    
    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)

    X_valid = tf.convert_to_tensor(X_valid)
    y_valid = tf.convert_to_tensor(y_valid)

    model = MiniBERT(
        vocab_size=len(vocab),
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_heads=8,
        ff_dim=512,
        num_layers=2,
        num_labels=len(label2id)
    )


    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=32,
        epochs=3
    )


    model.save_weights("mini_bert_weights.weights.h5")
    print("Weights saved successfully!")

    print("\nRunning sample prediction experiments...\n")

      # Number of samples to inspect
    num_samples = 5

    for i in range(num_samples):
      sample_input = X_valid[i:i+1]   # use validation set (better than train)
      
      logits = model(sample_input, training=False)
      predictions = tf.argmax(logits, axis=-1)

      print(f"\nSample {i+1}")
      print("Input:       ", X_valid[i].numpy())
      print("Prediction:  ", predictions[0].numpy())
      print("GroundTruth: ", y_valid[i].numpy())


    return model