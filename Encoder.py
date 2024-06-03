import torch
import math
from torch import nn
import torch.nn.functional as F

# Function for scaled dot-product attention
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # Scaled dot-product attention
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --") 
        # Adding the mask (if provided)
        scaled += mask
    # Softmax to get attention weights
    attention = F.softmax(scaled, dim=-1)
    # Multiplying the attention weights with the values
    values = torch.matmul(attention, v)
    return values, attention

# Multi-Head Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Linear layer to project the input to queries, keys, and values
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        print(f"x.size(): {x.size()}")
        # Apply the linear layer to get q, k, v
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        # Split qkv into q, k, v
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")
        # Scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size: {attention.size()}")
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")
        # Apply the final linear layer
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation ({std.size()})")
        y = (inputs - mean) / std
        print(f"y: {y.size()}")
        out = self.gamma * y + self.beta
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        return out

# Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # First linear layer
        x = self.linear1(x)
        print(f"x after first linear layer: {x.size()}")
        # ReLU activation
        x = self.relu(x)
        print(f"x after activation: {x.size()}")
        # Dropout for regularization
        x = self.dropout(x)
        print(f"x after dropout: {x.size()}")
        # Second linear layer
        x = self.linear2(x)
        print(f"x after 2nd linear layer: {x.size()}")
        return x

# Single Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x
        print("------- ATTENTION ------")
        # Multi-Head Attention
        x = self.attention(x, mask=None)
        print("------- DROPOUT 1 ------")
        # Dropout after attention
        x = self.dropout1(x)
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        # Add & Norm: Add the input (residual connection) and normalize
        x = self.norm1(x + residual_x)
        residual_x = x
        print("------- FEED-FORWARD NETWORK ------")
        # Feed-Forward Network
        x = self.ffn(x)
        print("------- DROPOUT 2 ------")
        # Dropout after feed-forward network
        x = self.dropout2(x)
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        # Add & Norm: Add the input (residual connection) and normalize
        x = self.norm2(x + residual_x)
        return x

# Encoder consisting of multiple Encoder Layers
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        # Stack of Encoder Layers
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x

# Model hyperparameters
d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

# Initialize the Encoder
encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)

# Create a random input tensor with shape (batch_size, max_sequence_length, d_model)
x = torch.randn((batch_size, max_sequence_length, d_model))  # includes positional encoding
# Pass the input through the Encoder
out = encoder(x)

