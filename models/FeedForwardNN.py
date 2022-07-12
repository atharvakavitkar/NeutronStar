#import libraries
import torch
import torch.nn as nn
from typing import Optional, Union

class FeedForwardNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layers: Union[int, list],
                 activation: Optional[Union[nn.Module, list]] = torch.nn.ReLU(),
                 batch_norm: Optional[Union[bool, list]] = False,
                 dropout: Optional[Union[float, list]] = 0,
                 ):
        super().__init__()

        # Assertions
        assert isinstance(output_size, int)
        assert isinstance(input_size, int)
        
        if type(hidden_layers) is int:
            hidden_layers = [hidden_layers]
        elif hidden_layers is None:
            hidden_layers = []

        layers = []
        for layer in hidden_layers:
            if dropout > 0 and dropout < 1:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(input_size, layer))
            input_size = layer
            if activation is not None:
                layers.append(activation)
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=input_size))
        layers.append(nn.Linear(input_size, output_size))

        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_in = torch.rand(128,437)
    print(sample_in[0].shape,sample_in[1].shape)
    model = FeedForwardNN(
                 input_size = 437,
                 output_size= 2,
                 hidden_layers = [10,20,30],
                 activation=torch.nn.ReLU(),
                 batch_norm = True,
                 dropout = 0.2)
    print(model)
    model.to(device)
    sample_out = model(sample_in.cuda())
    print(sample_out.shape)