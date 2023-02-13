import torch
from torch import nn


class SADE(nn.Module):
    def __init__(self, in_features, out_features, encoder_hidden=[64, 32], decoder_hidden=[32, 64]):
        """ 
        in_features: number, input data feature number
        encoder_hidden: list, hidden layer neuro number in encoder
        decoder_hidden: list, hidden layer neuro number in decoder
        """
        super().__init__()
        
        encoder_hidden = [in_features] + encoder_hidden
        decoder_hidden = decoder_hidden + [out_features]
        
        encoders, idx = [], 0
        while idx+1 < len(encoder_hidden):
            encoders.append( nn.Linear(in_features=encoder_hidden[idx], out_features=encoder_hidden[idx+1]) )
            encoders.append( nn.ReLU() )
            idx += 1
        
        decoders, idx = [], 0
        while idx+1 < len(decoder_hidden):
            decoders.append( nn.Linear(in_features=decoder_hidden[idx], out_features=decoder_hidden[idx+1]) )
            decoders.append( nn.ReLU() )
            idx += 1
        
        self.Encoder = nn.Sequential(*encoders)
        self.Decoder = nn.Sequential(*decoders[:-1])
        self.norm = nn.BatchNorm1d(num_features=in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x