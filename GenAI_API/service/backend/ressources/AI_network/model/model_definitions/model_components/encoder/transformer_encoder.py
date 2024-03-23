import torch
import torch.nn as nn
import numpy as np

class PositionEmbeddingFixedWeights(nn.Module):
    def __init__(self, sequence_length, vocab_size, output_dim):
        super(PositionEmbeddingFixedWeights, self).__init__()
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim) 
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)

        self.word_embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=output_dim,
            _weight=word_embedding_matrix,
            #trainable=False
        )
        #self.word_embedding_layer.weight = word_embedding_matrix

        self.position_embedding_layer = nn.Embedding(
            num_embeddings=sequence_length, embedding_dim=output_dim,
            _weight=position_embedding_matrix,
            #trainable=False
        )
        #self.position_embedding_layer.weight = position_embedding_matrix
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)

        return torch.tensor(P).to(torch.float)
 
 
    def forward(self, inputs):        
        position_indices = torch.tensor(range(inputs.shape[-1])) #tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)

        return embedded_words + embedded_indices

class AddNormalization(nn.Module):
    def __init__(self, embed_dim):
        super(AddNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization layer
 
    def forward(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x
 
        # Apply layer normalization to the sum
        return self.layer_norm(add)
    
class FeedForward(nn.Module):
    def __init__(self, input_dim, layer_output_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fully_connected1 = nn.Linear(input_dim, layer_output_dim)  # First fully connected layer
        self.fully_connected2 = nn.Linear(layer_output_dim, output_dim)  # Second fully connected layer
        self.activation = nn.ReLU()  # ReLU activation layer
 
    def forward(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)
 
        return self.fully_connected2(self.activation(x_fc1))

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
 
    def forward(self, queries, keys, values, key_dim, mask=None):
        queries = queries.to(float)
        keys = keys.to(float)
        values = values.to(float)

        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = torch.matmul(queries, keys.transpose(2, 3)) / torch.sqrt(torch.tensor(key_dim, dtype=torch.float32)) #queries.matmul(keys.T) / torch.sqrt(key_dim).to(torch.float)
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = nn.functional.softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return weights.matmul(values)
 
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_head, key_dim, value_dim, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = num_head  # Number of attention heads to use
        self.key_dim = key_dim  # Dimensionality of the linearly projected queries and keys
        self.values_dim = value_dim  # Dimensionality of the linearly projected values
        self.output_dim = output_dim  # Dimensionality of the model
        self.W_q = nn.Linear(input_dim, key_dim)  # Learned projection matrix for the queries
        self.W_k = nn.Linear(input_dim, key_dim)  # Learned projection matrix for the keys
        self.W_v = nn.Linear(input_dim, value_dim)  # Learned projection matrix for the values
        self.W_o = nn.Linear(key_dim, output_dim)  # Learned projection matrix for the multi-head output
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], heads, -1)) #x.view(x.shape[0], x.shape[1], heads, -1)
            x = x.permute(0, 2, 1, 3)
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, key_dim)
            x = x.permute(0, 2, 1, 3)
            #x = torch.reshape(x, shape=(x.shape[0], x.shape[1], self.key_dim))
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], self.key_dim)) #x.view(x.shape[0], x.shape[1], self.key_dim)

        return x.to(torch.float)
 
    def forward(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.key_dim, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, values_dim)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, output_dim)
        return self.W_o(output)

class EncoderLayer(nn.Module):
    def __init__(self, num_head, key_dim, value_dim, inner_fc_dim, layer_output_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(layer_output_dim, num_head, key_dim, value_dim, layer_output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.add_norm1 = AddNormalization(layer_output_dim)
        self.feed_forward = FeedForward(layer_output_dim, inner_fc_dim, layer_output_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.add_norm2 = AddNormalization(layer_output_dim)

    def forward(self, x, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output)
 
        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output)
 
        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, sequence_length, num_layers, num_head, key_dim, value_dim, inner_fc_dim, output_dim, dropout):
        super(Encoder, self).__init__()
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layer = [EncoderLayer(num_head, key_dim, value_dim, inner_fc_dim, output_dim, dropout) for _ in range(num_layers)]
 
    def forward(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        x = self.dropout(pos_encoding_output)
 
        #print(x.shape)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask)
 
        return x