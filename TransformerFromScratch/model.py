import tensorflow as tf
import math

class InputEmbeddings(tf.keras.Model):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(tf.keras.Model):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = tf.keras.layers.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = tf.zeros([seq_len, d_model])
        # create a vector of shape (seq_len, 1)
        position = tf.range(0, seq_len, dtype=tf.float32).unsqueeze(1)
        div_term  = tf.exp(tf.range(0, d_model, 2),float() * (-math.log(10000.0) / d_model))
        # apply the sin to even positions
        pe[:, 0::2] = tf.sin(position * div_term)
        pe[:, 1::2] = tf.cos(position * div_term)

        pe = pe.unsequeeze(0) # (1, seq_len, d_model)

        self.pe = tf.Variable(pe, trainable=False)  # non-trainable variable equivalent in torch (self.register_buffer('pe', pe))

    def call(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    
class LayerNormalization(tf.keras.Model):

    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = tf.Variable(tf.ones(1), trainable=True) # multiplied
        self.bias = tf.Variable(tf.zeros(1), trainable=True) # Added

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = std = tf.math.reduce_std(x, axis=-1, keepdims=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(tf.keras.Model):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(d_ff, input_shape=(d_model,)) # W1 and B1
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear_2 = tf.keras.layers.Dense(d_model, input_shape=(d_ff,))

    def call(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(tf.keras.activations.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(tf.keras.Model):

    def __init__(self, d_model, h, dropout):
        super().__init__()

        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) # Wq
        self.w_k = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) # Wk
        self.w_v = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) # Wv

        self.w_o = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) #Wo
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        return x


