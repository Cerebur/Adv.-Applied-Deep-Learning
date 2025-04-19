import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool


# Defintion of the GNN model
# Use the DynamicEdgeConv layer from the pytorch geometric package like this:
# MLP is a Multi-Layer Perceptron that is used to compute the edge features, you still need to define it.
# The input dimension to the MLP should be twice the number of features in the input data (i.e., 2 * n_features),
# because the edge features are computed from the concatenation of the two nodes that are connected by the edge.
# The output dimension of the MLP is the new feature dimension of this graph layer.
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x= self.model(x)
        return x


class GNNEncoder(nn.Module):
    def __init__(self, k, input_dim, output_dims=[12, 24, 12], num_targets=2):
        """
        Args:
            k (int): number of nearest neighbors to consider
            input_dim (int): number of features in the input data
            output_dims (list): list of output dimensions for each layer
            num_targets (int): number of target dimensions (e.g., 2 for xpos and ypos)
        """
        super(GNNEncoder, self).__init__()

        def create_layer(in_features, out_features):
            return DynamicEdgeConv(
                MLP(2 * in_features, out_features),  # Correct input dimension: 2 * in_features
                aggr='mean', k=k,  # k is the number of nearest neighbors to consider
            )
        
        # Create layers with the correct input and output dimensions
        self.layers = nn.ModuleList(
            [create_layer(input_dim if i == 0 else output_dims[i-1], output_dims[i]) for i in range(len(output_dims))]
        )

        # Add a final fully connected layer to map to the target dimensions
        self.fc = nn.Linear(output_dims[-1], num_targets)

        self.model_name='DGCNN'

    def forward(self, data):
        x = data.x
        batch = data.batch

        # loop over the DynamicEdgeConv layers:
        for layer in self.layers:
            x = layer(x, batch)

        # the output of the last layer has dimensions (n_batch, n_nodes, graph_feature_dimension)
        # where n_batch is the number of graphs in the batch and n_nodes is the number of nodes in the graph
        # i.e. one output per node (i.e. the hits in the event).
        # To combine all node features into single predictions, we recommend to use global pooling
        x = global_mean_pool(x, batch) # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # Pass through the final fully connected layer
        x = self.fc(x)  # -> (n_batch, num_targets)
        return x