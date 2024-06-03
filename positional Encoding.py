# Importing necessary libraries and packages
import torch
import torch.nn as nn

# Defining the PositionalEncoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length  # Maximum sequence length for the positional encoding
        self.d_model = d_model  # Dimension of the model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()  # Create a tensor of even indices
        denominator = torch.pow(10000, even_i / self.d_model)  # Calculate the denominator for the sine and cosine functions
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)  # Create a tensor for positions and reshape it
        even_PE = torch.sin(position / denominator)  # Calculate sine values for even positions
        odd_PE = torch.cos(position / denominator)  # Calculate cosine values for odd positions
        stacked = torch.stack([even_PE, odd_PE], dim=2)  # Stack the even and odd positional encodings along a new dimension
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)  # Flatten the stacked tensor to get the final positional encoding
        return PE  # Return the positional encoding tensor

# Creating an instance of the PositionalEncoding class with a model dimension of 6 and maximum sequence length of 10
pe = PositionalEncoding(d_model=6, max_sequence_length=10)

# Calling the forward method to generate the positional encoding
pe.forward()
