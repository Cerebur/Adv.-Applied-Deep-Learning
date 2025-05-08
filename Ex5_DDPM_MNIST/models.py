import torch

# define the neural network that predicts the amount of noise that was
# added to the data
# the network should have two inputs (the current data and the time step)
# and one output (the predicted noise)

# the network should be a simple MLP with 2 hidden layers
class NoisePredictor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, name='NoisePredictor', num_layers=4):
        """
        Args:
            input_dim (int): The dimension of the input data (excluding time step)
            output_dim (int): The dimension of the output data
            hidden_dim (int): The dimension of the hidden layers
        """
        # Initialize the parameters
        super(NoisePredictor, self).__init__()
        self.input_dim = input_dim  # Add 1 to account for the time step
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Define the layers of the network
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.hidden_dim)])
        for _ in range(num_layers-2):
            self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(torch.nn.Linear(self.hidden_dim, self.output_dim))

        self.name = name

    def forward(self, x, t):
        # Ensure t has the correct shape and concatenate it with x
        t = t.view(-1, 1)  # Reshape t to have a second dimension
        x = x.view(-1, 1)  # Reshape x to match dimensions
        x = torch.cat([x, t], dim=-1)  # Concatenate along the feature dimension
        # Pass through the network
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x

