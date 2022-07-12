import torch.nn as nn
import torch
from FeedForwardNN import FeedForwardNN
from typing import Union

class MultimodalTransformer(nn.Module):
    def __init__(self,
                 num_heads : int,
                 num_encoder_layers : int,
                 dim_feedforward: int,
                 dropout: float,
                 fc_layers: Union[int, list],
                 ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=434, 
                                                   nhead=num_heads,
                                                   dropout=dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                    num_layers=num_encoder_layers,
                                                    dim_feedforward=dim_feedforward,)
        
        self.fc_layers = FeedForwardNN(input_size=437,
                                       output_size=2,
                                       hidden_layers=fc_layers,
                                       batch_norm=True,
                                       dropout=dropout)

    def forward(self, x):
        print(x.shape)
        return x
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_in = torch.rand(128,437)
    model = MultimodalTransformer()
    print(model)
    model.to(device)
    sample_out = model(sample_in.cuda())
    print(sample_out.shape)