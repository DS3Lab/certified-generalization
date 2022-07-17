import torch.nn as nn

activation_functions = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(alpha=1.0)
}


class NeuralNetV2(nn.Module):
    def __init__(self, activation='elu', num_hidden=1, input_dim=2, num_classes=2, width_multiplier=2):
        super(NeuralNetV2, self).__init__()
        self.num_hidden = num_hidden
        activation = nn.ReLU() if activation == 'relu' else nn.ELU(alpha=1.0)

        # input layer
        layers = [
            nn.utils.spectral_norm(
                nn.Linear(input_dim, width_multiplier * input_dim, bias=False), n_power_iterations=1),
            nn.ReLU() if activation == 'relu' else nn.ELU(alpha=1.0)
        ]

        # hidden layers
        for i in range(1, num_hidden):
            # linear layer with spectral normalization
            layers.append(
                nn.utils.spectral_norm(
                    nn.Linear(width_multiplier * input_dim, width_multiplier * input_dim, bias=False),
                    n_power_iterations=1)
            )

            # activation function
            layers.append(
                nn.ReLU() if activation == 'relu' else nn.ELU(alpha=1.0)
            )

        # final layer
        layers.append(
            nn.utils.spectral_norm(nn.Linear(width_multiplier * input_dim, num_classes, bias=False),
                                   n_power_iterations=1))

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)

    def init_weights_glorot(self):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
