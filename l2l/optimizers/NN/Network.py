from torch import nn


class Network(nn.Module):
    def __init__(self, neurons):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.fa = nn.Linear(1, neurons)
        self.fah = nn.Linear(neurons, neurons)
        self.fao = nn.Linear(neurons, 1)
        # Output layer
        self.fp = nn.Linear(1, neurons)
        self.fph = nn.Linear(neurons, neurons)
        self.fpo = nn.Linear(neurons, 1)

        # Define reulu activation and softmax output
        self.relu = nn.ReLU()

    def forward(self, x):
        amp = self.relu(self.fa(x))
        amp = self.relu(self.fah(amp))
        amp = self.fao(amp)

        ph = self.relu(self.fp(x))
        ph = self.relu(self.fph(ph))
        ph = self.fpo(ph)

        return {'amp': amp, 'phase': ph}
