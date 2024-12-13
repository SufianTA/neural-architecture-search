# neural-architecture-search/nas_example.py

import torch
import torch.nn as nn

# Simple example of a neural architecture search (NAS) method
class NASModel(nn.Module):
    def __init__(self):
        super(NASModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 224 * 224, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Example forward pass
model = NASModel()
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output)
