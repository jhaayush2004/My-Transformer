# Importing necessary libraries and packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Function to calculate scaled dot-product attention
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]  # Dimension of queries and keys
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # Scaled dot-product
    if mask is not None:
        scaled += mask  # Apply mask if provided
    attention = F.softmax(scaled, dim=-1)  # Apply softmax to get attention weights
    values = torch.matmul(attention, v)  # Multiply attention weights by values
    return values, attention

# Class for Multihead Attention mechanism
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim  # Input dimension
        self.d_model = d_model  # Dimension of the model
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = d_model // num_heads  # Dimension of each head
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)  # Linear layer to generate Q, K, V vectors
        self.linear_layer = nn.Linear(d_model, d_model)  # Linear layer to combine outputs

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()  # Get the dimensions of the input
        print(f"x.size(): {x.size()}")  # Debug print of input size
        
        qkv = self.qkv_layer(x)  # Generate Q, K, V vectors
        print(f"qkv.size(): {qkv.size()}")  # Debug print of qkv size
        
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)  # Reshape qkv
        print(f"qkv.size(): {qkv.size()}")  # Debug print of reshaped qkv size
        
        qkv = qkv.permute(0, 2, 1, 3)  # Rearrange dimensions for multi-head attention
        print(f"qkv.size(): {qkv.size()}")  # Debug print of permuted qkv size
        
        q, k, v = qkv.chunk(3, dim=-1)  # Split qkv into Q, K, V
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")  # Debug print of Q, K, V sizes
        
        values, attention = scaled_dot_product(q, k, v, mask)  # Compute scaled dot-product attention
        print(f"values.size(): {values.size()}, attention.size: {attention.size()}")  # Debug print of attention values and weights
        
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)  # Reshape attention values
        print(f"values.size(): {values.size()}")  # Debug print of reshaped values size
        
        out = self.linear_layer(values)  # Apply linear layer to combine outputs
        print(f"out.size(): {out.size()}")  # Debug print of final output size
        
        return out  # Return the final output
