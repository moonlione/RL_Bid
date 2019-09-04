import torch
import torch.nn as nn
import torch.nn.functional as F

neural_nums = 50

class Actor(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_numbers, neural_nums)
        self.fc2 = nn.Linear(neural_nums, neural_nums)
        self.out = nn.Linear(neural_nums, action_numbers)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        x = F.relu(self.dropout(self.fc1(input)))
        x_ = F.relu(self.dropout(self.fc2(x)))
        out = torch.tanh(self.out(x_))

        return out

class Critic(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(feature_numbers, neural_nums)
        self.fc_a = nn.Linear(action_numbers, neural_nums)
        self.fc_q = nn.Linear(2 * neural_nums, neural_nums)
        self.fc_ = nn.Linear(neural_nums, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, action):
        f_s = F.relu(self.dropout(self.fc_s(input)))
        f_a = F.relu(self.dropout(self.fc_a(action)))
        cat = torch.cat([f_s, f_a], dim=1)
        q = F.relu(self.dropout(self.fc_q(cat)))
        q = self.fc_(q)

        return q