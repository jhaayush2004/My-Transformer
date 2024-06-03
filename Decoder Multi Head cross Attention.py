class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Layer for key and value
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
        # Layer for query
        self.q_layer = nn.Linear(d_model , d_model)
        # Linear layer for output
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask=None):
        # Input shape: (batch_size, sequence_length, d_model)
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        
        # Processing key and value
        kv = self.kv_layer(x) # 30 x 200 x 1024
        print(f"kv.size(): {kv.size()}")
        
        # Processing query
        q = self.q_layer(y) # 30 x 200 x 512
        print(f"q.size(): {q.size()}")
        
        # Reshaping key and value for multi-head attention
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # 30 x 200 x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # 30 x 200 x 8 x 64
        
        # Permuting dimensions for multi-head attention
        kv = kv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3) # 30 x 8 x 200 x 64
        
        # Splitting key and value
        k, v = kv.chunk(2, dim=-1) # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        
        # Applying scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask) #  30 x 8 x 200 x 64
        print(f"values: {values.size()}, attention:{attention.size()}")
        
        # Reshaping values
        values = values.reshape(batch_size, sequence_length, d_model) #  30 x 200 x 512
        
        # Applying linear layer for output
        out = self.linear_layer(values)  #  30 x 200 x 512
        print(f"out after passing through linear layer: {out.size()}")
        
        return out  #  30 x 200 x 512
