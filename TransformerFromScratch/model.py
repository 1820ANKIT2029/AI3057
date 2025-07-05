import tensorflow as tf
import math

class InputEmbeddings(tf.keras.layers.Layer):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = tf.keras.layers.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = tf.zeros([seq_len, d_model])
        # create a vector of shape (seq_len, 1)
        position = tf.expand_dims(tf.range(0, seq_len, dtype=tf.float32), axis=1)
        div_term  = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))
        # apply the sin to even positions
        pe_even = tf.sin(position * div_term)   # shape (max_seq_len, d_model/2)
        pe_odd  = tf.cos(position * div_term)   # shape (max_seq_len, d_model/2)

         # Interleave them back together
        pe = tf.reshape(
            tf.concat([pe_even[:, :, tf.newaxis], pe_odd[:, :, tf.newaxis]], axis=2),
            (seq_len, d_model)
        )

        pe = tf.expand_dims(pe, axis=0) # (1, seq_len, d_model)

        self.pe = tf.Variable(pe, trainable=False)  # non-trainable variable equivalent in torch (self.register_buffer('pe', pe))

    def call(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    
class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = tf.Variable(tf.ones(1), trainable=True) # multiplied
        self.bias = tf.Variable(tf.zeros(1), trainable=True) # Added

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = std = tf.math.reduce_std(x, axis=-1, keepdims=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(tf.keras.layers.Layer):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(d_ff, input_shape=(d_model,)) # W1 and B1
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear_2 = tf.keras.layers.Dense(d_model, input_shape=(d_ff,))

    def call(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(tf.keras.activations.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(tf.keras.layers.Layer):

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

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = tf.reshape(query, (tf.shape(query)[0], tf.shape(query)[1], self.h, self.d_k))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.reshape(key, (tf.shape(key)[0], tf.shape(key)[1], self.h, self.d_k))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.reshape(value, (tf.shape(value)[0], tf.shape(value)[1], self.h, self.d_k))
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = tf.transpose(x, perm=[0,2,1, 3])
        x = tf.reshape(x, shape=[tf.shape(x)[0], -1, self.h * self.d_k])

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: tf.keras.layers.Dropout):
        d_k = tf.shape(query)[-1]

        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
        if mask is not None:
            attention_scores = tf.where(mask == 0, -1e9, attention_scores)
        attention_scores = tf.keras.activations.softmax(attention_scores, axis=-1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return tf.matmul(attention_scores, value), attention_scores

class ResidualConnection(tf.keras.layers.Layer):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = LayerNormalization()

    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(tf.keras.layers.Layer):

    def __init__(
        self, 
        self_attenttion_block: MultiHeadAttentionBlock, 
        feed_forword_block: FeedForwardBlock, 
        dropout: float
    ):
        super().__init__()
        self.self_attention_block = self_attenttion_block
        self.feed_forword_block = feed_forword_block
        self.residual_connections = [ResidualConnection(dropout) for _ in range(2)]

    def call(self, x, src_mask):
        x = self.residual_connections[0](x, sublayer=lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, sublayer=self.feed_forword_block)

        return x
    
class Encoder(tf.keras.layers.Layer):

    def __init__(self, layers: list[tf.keras.layers.Layer]):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def call(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class DecoderBlock(tf.keras.layers.Layer):

    def __init__(
        self, 
        self_attention_block: MultiHeadAttentionBlock, 
        cross_attention_block: MultiHeadAttentionBlock, 
        feed_forward_block: FeedForwardBlock, 
        dropout: float
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [ResidualConnection(dropout) for _ in range(3)]

    def call(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, sublayer=lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, sublayer=lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, sublayer=self.feed_forward_block)

        return x
    
class Decoder(tf.keras.layers.Layer):

    def __init__(self, layers: list[tf.keras.layers.Layer]):
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def call(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    
class ProjectionLayer(tf.keras.layers.Layer):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = tf.keras.layers.Dense(vocab_size, input_shape=(d_model,))

    def call(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return tf.keras.activations.log_softmax(self.proj(x), axis=-1)
    
class Transformer(tf.keras.Model):

    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        src_embed: InputEmbeddings, 
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int, 
    src_seq_len: int, 
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    transformer.summary()

    return transformer