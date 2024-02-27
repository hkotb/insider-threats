import torch.nn as nn
import torch.nn.init as init

class SimpleNN(nn.Module):
    def __init__(self, input_size, dropout_prob=0.01):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        self.model = self.build_model()

        self.initialize_weights()

    def build_model(self):
        layers = [self.build_block(self.input_size, 100, self.dropout_prob)]
        
        # Add repeated blocks
        for _ in range(10):
            layers += [self.build_block(100, 100, self.dropout_prob)]

        # Add final layers
        layers += [
            nn.Linear(100, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_prob),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ]

        return nn.Sequential(*layers)

    def build_block(self, in_features, out_features, dropout_prob):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob)
        )

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        return self.model(x)

