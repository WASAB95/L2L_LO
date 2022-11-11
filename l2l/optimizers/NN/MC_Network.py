from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.first_layer = nn.Linear(1, 15)
        self.second_layer = nn.Linear(15, 300)
        # Output layer
        self.output_layer = nn.Linear(300, 315)

        # Define reulu activation and softmax output
        self.relu = nn.ReLU()

    def forward(self, x):
        first_l = self.first_layer(x)
        first_l = self.relu(first_l)
        second_l = self.second_layer(first_l)
        second_l = self.relu(second_l)
        output = self.output_layer(second_l)

        return output
