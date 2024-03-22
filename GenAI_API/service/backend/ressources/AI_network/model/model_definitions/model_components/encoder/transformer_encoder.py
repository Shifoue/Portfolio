import torch
import torch.nn as nn
import numpy as np

class PositionEmbeddingFixedWeights(nn.Module):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)   
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)

        self.word_embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=output_dim,
            #trainable=False
        )
        self.word_embedding_layer.weight = word_embedding_matrix

        self.position_embedding_layer = nn.Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            #trainable=False
        )
        self.position_embedding_layer.weight = position_embedding_matrix
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
 
 
    def call(self, inputs):        
        position_indices = torch.tensor(range(inputs.shape[-1])) #tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)

        return embedded_words + embedded_indices

class AddNormalization(nn.Module):
    def __init__(self, embed_dim):
        super(AddNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization layer
 
    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x
 
        # Apply layer normalization to the sum
        return self.layer_norm(add)
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, inner_fc_dim, model_sublayer_dim):
        super(FeedForward, self).__init__()
        self.fully_connected1 = nn.Linear(embed_dim, inner_fc_dim)  # First fully connected layer
        self.fully_connected2 = nn.Linear(inner_fc_dim, model_sublayer_dim)  # Second fully connected layer
        self.activation = nn.ReLU()  # ReLU activation layer
 
    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)
 
        return self.fully_connected2(self.activation(x_fc1))

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_head, key_dim, value_dim, inner_fc_dim, model_sublayer_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = nn.MultiHeadAttention(embed_dim, num_head, kdim=key_dim, vdim=value_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.add_norm1 = AddNormalization(embed_dim)
        self.feed_forward = FeedForward(inner_fc_dim, model_sublayer_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.add_norm2 = AddNormalization(embed_dim)

    def call(self, x, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)
 
        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)
 
        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, sequence_length, num_layers, num_head, key_dim, value_dim, inner_fc_dim, model_sublayer_dim, dropout):
        super(Encoder, self).__init__()
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, model_sublayer_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layer = [EncoderLayer(sequence_length, num_head, key_dim, value_dim, inner_fc_dim, model_sublayer_dim, dropout) for _ in range(num_layers)]
 
    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)
 
        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)
 
        return x