import torch.nn as nn


class MLP(nn.Module):
    """
    MLP module used for multiclass classification on the encodings.
    """
    def __init__(self, enc_dim, output_dim):
        super().__init__()
        self.activation = nn.SELU()
        self.fc1 = nn.Linear(enc_dim, enc_dim)
        self.bn1 = nn.BatchNorm1d(enc_dim, momentum=0.01, eps=0.001)
        self.fc2 = nn.Linear(enc_dim, enc_dim)
        self.bn2 = nn.BatchNorm1d(enc_dim, momentum=0.01, eps=0.001)
        self.fc3 = nn.Linear(enc_dim, output_dim)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.soft(out)
        return out
