class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # 1536 .Linear transformation for query, key, and value
        self.linear_layer = nn.Linear(d_model, d_model) # Linear transformation for output
    
    def forward(self, x, mask=None):
       # Get dimensions of the input tensor
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512 
        print(f"x.size(): {x.size()}")
       # Linear transformation to obtain query, key, and value matrices
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        print(f"qkv.size(): {qkv.size()}")
      # Reshape to separate heads
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        print(f"qkv after reshape .size(): {qkv.size()}")
      # Permute to bring the heads dimension before sequence length
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        print(f"qkv after permutation: {qkv.size()}")
       # Split into query, key, and value matrices
        q, k, v = qkv.chunk(3, dim=-1) # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        print(f"q: {q.size()}, k:{k.size()}, v:{v.size()}")
       # Compute scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask) # values: 30 x 8 x 200 x 64
        print(f"values: {values.size()}, attention:{attention.size()}")
      # Reshape the values matrix
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        print(f"values after reshaping: {values.size()}")
      # Linear transformation for output
        out = self.linear_layer(values) # 30 x 200 x 512
        print(f"out after passing through linear layer: {out.size()}")
        return out # 30 x 200 x 512  Linear transformation for output
