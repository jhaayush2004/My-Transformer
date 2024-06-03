import torch
from torch import nn

# Input tensor with shape (Batch, Sequence, Embedding)
inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
B, S, E = inputs.size()

# Reshape the inputs to (Sequence, Batch, Embedding) for easier manipulation
inputs = inputs.reshape(S, B, E)

# Define the parameter shape based on the last two dimensions of the reshaped input
parameter_shape = inputs.size()[-2:]

# Initialize gamma (scale) and beta (shift) parameters
gamma = nn.Parameter(torch.ones(parameter_shape))
beta = nn.Parameter(torch.zeros(parameter_shape))

# Compute the dimensions for which we want to compute the layer norm
dims = [-(i + 1) for i in range(len(parameter_shape))]

# Compute the mean and variance along the specified dimensions
mean = inputs.mean(dim=dims, keepdim=True)
var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)

# Add a small constant epsilon to the variance to avoid division by zero, then compute the standard deviation
epsilon = 1e-5
std = (var + epsilon).sqrt()

# Normalize the inputs by subtracting the mean and dividing by the standard deviation
y = (inputs - mean) / std

# Apply the scale (gamma) and shift (beta) parameters
out = gamma * y + beta

# Define a class for LayerNormalization
class LayerNormalization():
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        # Compute the dimensions for which we want to compute the layer norm
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        
        # Compute the mean and variance along the specified dimensions
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean \n ({mean.size()}): \n {mean}")
        
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        
        # Add a small constant epsilon to the variance to avoid division by zero, then compute the standard deviation
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation \n ({std.size()}): \n {std}")
        
        # Normalize the inputs by subtracting the mean and dividing by the standard deviation
        y = (inputs - mean) / std
        print(f"y \n ({y.size()}) = \n {y}")
        
        # Apply the scale (gamma) and shift (beta) parameters.For instance, if gamma is set to 1.5 and beta to 0.5, it scales the normalized input by 1.5 and shifts it by 0.5. This adjustment can bring the normalized inputs into a range that is more suitable for the activation function of the next layer, thereby improving the network's learning ability and overall performance.
        out = self.gamma * y + self.beta
        print(f"out \n ({out.size()}) = \n {out}")
        
        return out

# Example usage of the LayerNormalization class
batch_size = 3
sentence_length = 5
embedding_dim = 8 
inputs = torch.randn(sentence_length, batch_size, embedding_dim)

# Instantiate the LayerNormalization class with the parameter shape
layer_norm = LayerNormalization(parameters_shape=(sentence_length, batch_size))

# Apply the layer normalization to the inputs
normalized_output = layer_norm.forward(inputs)


