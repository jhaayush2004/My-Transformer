# Importing Libraries and Packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Initializing sequence length, batch size, and input dimension
sequence_length = 10
batch_size = 1
input_dim = 512
d_model = 512 # Output dimension of the attention model for every single word

# Randomly initializing input x
x = torch.randn((batch_size, sequence_length, input_dim))

# qkv layer is needed to create Query, Key, and Value vectors for input tokens
qkv_layer = nn.Linear(input_dim, 3 * d_model)
qkv = qkv_layer(x)

# Number of heads is initialized to 8
num_heads = 8
head_dim = d_model // num_heads # 512/8=64

# Reshape qkv to separate heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim) # shape = 1*10*8*192
qkv = qkv.permute(0, 2, 1, 3) # shape = 1*8*10*192

# Split qkv into Q, K, and V
q, k, v = qkv.chunk(3, dim=-1) # shape = 1*8*10*64

# d_k is the dimension of the queries and keys
d_k = q.size()[-1] # d_k = 64

# Scaled is the attention score matrix
scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

# Mask has been initialized to prevent data leakage (future tokens should not be visible)
mask = torch.full(scaled.size(), float('-inf'))

# Future positions are masked out, ensuring that the model cannot see future tokens during training
mask = torch.triu(mask, diagonal=1)
scaled += mask

# Apply softmax to get attention weights
attention = F.softmax(scaled, dim=-1)

# This gives self-attention matrix values
values = torch.matmul(attention, v)

# Initializing self_dot_product function
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# Apply scaled dot product attention
values, attention = scaled_dot_product(q, k, v, mask=mask)

# Reshape values to original shape
values = values.reshape(batch_size, sequence_length, num_heads * head_dim)

# Apply a linear layer to get the final output
linear_layer = nn.Linear(d_model, d_model)
out = linear_layer(values)
