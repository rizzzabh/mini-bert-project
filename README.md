# mini-bert-project

Mini-BERT (Transformer Encoder from Scratch)

TensorFlow, Keras, Self-Attention, Masked Language Modeling
• Designed and implemented a BERT-style Transformer Encoder architecture from scratch using TensorFlow (subclassed tf.keras.Model)
• Built custom Multi-Head Self-Attention and stacked Transformer encoder blocks
• Implemented masked language modeling objective with custom masked loss and accuracy metric
• Engineered positional embeddings and token embeddings for sequence modeling
• Trained and optimized model on GPU (Apple Metal backend) with Adam optimizer
• Diagnosed and debugged prediction collapse and masking propagation issues
• Implemented model weight serialization and loading for reproducible experiments
• Achieved ~95% masked token accuracy on validation set
