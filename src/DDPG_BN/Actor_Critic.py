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

        self.batch_norm_input = nn.BatchNorm1d(config['feature_num'],
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

        self.batch_norm_layer = nn.BatchNorm1d(config['neuron_nums'],
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

    def forward(self, input):
        input = self.batch_norm_input(input)
        x = self.fc1(input)
        x = self.batch_norm_layer(x)
        x = F.relu(x)
        x = self.batch_norm_layer(x)
        x_ = self.fc2(x)
        x_ = self.batch_norm_layer(x_)
        x_ = F.relu(x_)
        x_ = self.batch_norm_layer(x)
        out = torch.tanh(self.out(x_))

        return out

class Critic(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(feature_numbers, neural_nums)
        self.fc_a = nn.Linear(action_numbers, neural_nums)
        self.fc_q = nn.Linear(2 * neural_nums, neural_nums)
        self.fc_ = nn.Linear(neural_nums, 1)

        self.batch_norm_input = nn.BatchNorm1d(config['feature_num'],
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        self.batch_norm_action = nn.BatchNorm1d(1,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        self.batch_norm_layer = nn.BatchNorm1d(config['neuron_nums'],
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)
    def forward(self, input, action):
        input = self.batch_norm_input(input)
        input = self.fc_s(input)
        input = self.batch_norm_layer(input)
        f_s = F.relu(input)
        action = self.batch_norm_action(action)
        action = self.fc_a(action)
        action = self.batch_norm_layer(action)
        f_a = F.relu(action)
        cat = torch.cat([f_s, f_a], dim=1)
        cat = self.fc_q(cat)
        cat = self.batch_norm_layer(cat)
        q = F.relu(cat)
        q = self.batch_norm_layer(q)
        q = self.fc_(q)

        return q