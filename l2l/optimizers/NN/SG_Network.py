from torch import nn


class Network(nn.Module):
    def __init__(self, neurons):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.fa = nn.Linear(1, neurons)
        self.fao = nn.Linear(neurons, 1)
        # Output layer
        self.fp = nn.Linear(1, neurons)
        self.fpo = nn.Linear(neurons, 1)

        # Define reulu activation and softmax output
        self.relu = nn.ReLU()

    def forward(self, x):
        amp_x = self.fa(x)
        amp_x = self.relu(amp_x)
        amp = self.fao(amp_x)
        ph_x = self.fp(x)
        ph_x = self.relu(ph_x)
        ph = self.fpo(ph_x)

        return {'amp': amp, 'phase': ph}
