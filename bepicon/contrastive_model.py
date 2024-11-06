import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, Linear, Sequential



def gnn_block(in_channels: int, 
              out_channels: int, 
              activation: str = "leakyrelu",
              **kwargs) -> Sequential:
    
    """
    Constructs a Graph Neural Network (GNN) block using a GATConv layer followed by 
    batch normalization and an activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str, optional): Activation function to use. 
                                    Options are "leakyrelu", "relu", and "silu". 
                                    Default is "leakyrelu".
        **kwargs: Additional keyword arguments to pass to the GATConv layer.

    Returns:
        Sequential: A sequential container of the GNN block components.
    """

    activations = nn.ModuleDict({
        "leakyrelu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(inplace=True),
        "silu": nn.SiLU(inplace=True),
    })

    gnn_block = Sequential('x, edge_index, edge_attr', 
            [
            (GATConv(in_channels=in_channels, out_channels=out_channels, **kwargs), 'x, edge_index, edge_attr -> x'),
            nn.BatchNorm1d(out_channels * kwargs.get("heads") if kwargs.get("heads") else out_channels), 
            activations[activation],
        ]
    )

    return gnn_block


class GNNEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = -1,
                 out_channels: int = 128,
                 heads: int = 8,
                 attention_dropout_ratio: float = 0.1, 
                 dropout_ratio: float = 0.25,
                 num_layers: int = 3,
                 activation: str = "leakyrelu"
        ):

        super().__init__()

        self.dropout_ratio = dropout_ratio

        self.gnn_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_blocks.append(gnn_block(in_channels=in_channels, out_channels=out_channels, activation=activation, heads=heads, dropout=attention_dropout_ratio))


    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for index, blk in enumerate(self.gnn_blocks, start=1):
            if index == len(self.gnn_blocks):
                x = blk(x, edge_index, edge_attr)
            else:
                x = blk(x, edge_index, edge_attr)
                x = F.dropout(x, training=self.training, p=self.dropout_ratio)

        return x
    

class ProjectionHead(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 num_layers:int = 2, 
                 activation: str = "leakyrelu"):
        
        super().__init__()

        activations = nn.ModuleDict({
        "leakyrelu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(inplace=True),
        "silu": nn.SiLU(inplace=True),
        })


        self.lin_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.lin_blocks.append(nn.Sequential(
                Linear(hidden_channels if i > 0 else in_channels, hidden_channels if i != num_layers-1 else out_channels),
                activations[activation],
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.lin_blocks:
            x = blk(x)

        return x



class GAT(nn.Module):
    
    def __init__(self,
                 encoder_in_channels: int = -1,
                 encoder_out_channels: int = 128,
                 encoder_heads: int = 8,
                 encoder_attention_dropout_ratio: float = 0.1,
                 encoder_dropout_ratio: float = 0.25,
                 encoder_num_layers: int = 3,
                 projection_in_channels: int = -1,
                 projection_hidden_channels: int = 512,
                 projection_out_channels: int = 128,
                 projection_num_layers: int = 2,
                 activation: str = "leakyrelu",):
        
        """
        GAT (Graph Attention Network) is a wrapper class that combines a GNNEncoder and a ProjectionHead.
        Args:
            encoder_in_channels (int): Number of input channels for the encoder. Default is -1.
            encoder_out_channels (int): Number of output channels for the encoder. Default is 128.
            encoder_heads (int): Number of attention heads in the encoder. Default is 8.
            encoder_attention_dropout_ratio (float): Dropout ratio for the attention mechanism in the encoder. Default is 0.1.
            encoder_dropout_ratio (float): Dropout ratio for the encoder. Default is 0.25.
            encoder_num_layers (int): Number of layers in the encoder. Default is 3.
            projection_in_channels (int): Number of input channels for the projection head. Default is -1.
            projection_hidden_channels (int): Number of hidden channels for the projection head. Default is 512.
            projection_out_channels (int): Number of output channels for the projection head. Default is 128.
            projection_num_layers (int): Number of layers in the projection head. Default is 2.
            activation (str): Activation function to use. Default is "leakyrelu".
        
        """
        
        super().__init__()

        self.encoder = GNNEncoder(in_channels=encoder_in_channels, 
                                  out_channels=encoder_out_channels, 
                                  heads=encoder_heads, 
                                  attention_dropout_ratio=encoder_attention_dropout_ratio, 
                                  dropout_ratio=encoder_dropout_ratio, 
                                  num_layers=encoder_num_layers, 
                                  activation=activation)
        
        self.projection = ProjectionHead(in_channels=projection_in_channels, 
                                         hidden_channels=projection_hidden_channels, 
                                         out_channels=projection_out_channels, 
                                         num_layers=projection_num_layers, 
                                         activation=activation)
        
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT model.
        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge attributes.
        Returns:
            torch.Tensor: Output features after passing through the encoder and projection head.
        """
        x = self.encoder(x, edge_index, edge_attr)
        x = self.projection(x)
        return x