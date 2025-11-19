import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
   
    def __init__(self, state_size: int = 147, action_size: int = 3, seed: int = 42):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)  
        self.fc2 = nn.Linear(64, 64)          
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values