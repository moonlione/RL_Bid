import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DDPG_BN.config import config

neural_nums = 50

class Actor(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_numbers, neural_nums)
        self.fc2 = nn.Linear(neural_nums, neural_nums)
        self.out = nn.Linear(neural_nums, action_numbers)

        self.batch_norm_input = nn.BatchNorm1d(feature_numbers,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

        self.batch_norm_layer = nn.BatchNorm1d(neural_nums,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

        self.batch_norm_action = nn.BatchNorm1d(action_numbers,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

    def forward(self, input):
        x = F.relu(self.batch_norm_layer(self.fc1(self.batch_norm_input(input))))
        x_ = F.relu(self.batch_norm_layer(self.fc2(x)))
        out = torch.tanh(self.batch_norm_action(self.out(x_)))

        return out

class Critic(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(feature_numbers, neural_nums)
        self.fc_a = nn.Linear(action_numbers, neural_nums)
        self.fc_q = nn.Linear(2 * neural_nums, neural_nums)
        self.fc_ = nn.Linear(neural_nums, 1)

        self.batch_norm_input = nn.BatchNorm1d(feature_numbers,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        self.batch_norm_layer = nn.BatchNorm1d(neural_nums,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

    def forward(self, input, action):
        xs = F.relu(self.fc1(self.batch_norm_input(input)))
        x = torch.cat([self.batch_norm_layer(xs), self.fc_a(action)], dim=1)
        q = F.relu(self.batch_norm_layer(self.fc_q(x)))
        q = self.fc_(q)

        return q