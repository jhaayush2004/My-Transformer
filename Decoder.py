class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        # Self-Attention Mechanism
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # Layer Normalization for self-attention output
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        # Dropout layer after self-attention
        self.dropout1 = nn.Dropout(p=drop_prob)
        # Cross-Attention Mechanism between decoder and encoder
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        # Layer Normalization for encoder-decoder attention output
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        # Dropout layer after encoder-decoder attention
        self.dropout2 = nn.Dropout(p=drop_prob)
        # Feed-Forward Network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # Layer Normalization for feed-forward output
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        # Dropout layer after feed-forward network
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y # 30 x 200 x 512
        # Save the original input for residual connection
        # Self-Attention Block
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=decoder_mask) # 30 x 200 x 512
        print("DROP OUT 1")
        y = self.dropout1(y) # 30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1(y + _y) # 30 x 200 x 512

        _y = y # 30 x 200 x 512
        print("CROSS ATTENTION")
        y = self.encoder_decoder_attention(x, y, mask=None) #30 x 200 x 512
        print("DROP OUT 2")  #30 x 200 x 512
        y = self.dropout2(y)
        print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2(y + _y)  #30 x 200 x 512

        _y = y  #30 x 200 x 512
       # Feed-Forward Network Block
        print("FEED FORWARD 1")
        y = self.ffn(y) #30 x 200 x 512
        # Dropout after feed-forward network
        print("DROP OUT 3")
        y = self.dropout3(y) #30 x 200 x 512
       # Residual connection and layer normalization after feed-forward network
        print("ADD + LAYER NORMALIZATION 3")
        y = self.norm3(y + _y) #30 x 200 x 512
        return y #30 x 200 x 512

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        # Sequentially pass through each decoder layer
        for module in self._modules.values():
            y = module(x, y, mask) #30 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
       # Stacking multiple decoder layers
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                          for _ in range(num_layers)])

    def forward(self, x, y, mask):
        # Forward pass through the stacked decoder layers
        #x : 30 x 200 x 512 
        #y : 30 x 200 x 512
        #mask : 200 x 200
        y = self.layers(x, y, mask)
        return y #30 x 200 x 512
     
