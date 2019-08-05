import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.AC.No_threshold_no_reward.config import config

class Critic_net(nn.Module):
    def __init__(self, feature_nums, action_nums):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(feature_nums, config['neuron_nums'])
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(feature_nums, config['neuron_nums'])
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(config['neuron_nums'], action_nums)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x_ = self.fc1(x)
        x_ = F.relu(x_)
        y = self.fc2(x_)
        return y

class Critic():
    def __init__(self,
                 feature_nums,
                 action_nums,
                 reward_decay,
                 learning_rate = 0.1):
        self.feature_nums = feature_nums
        self.action_nums = action_nums
        self.gamma = reward_decay
        self.lr = learning_rate

        self.critic = Critic_net(self.feature_nums, self.action_nums)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def td_error(self, r, v_, v):
        td_er = torch.sqrt(r + self.gamma * v_ - v)
        return td_er

    def learn(self, s, r, s_):
        state = torch.unsqueeze(torch.FloatTensor(s), 0)
        state_ = torch.unsqueeze(torch.FloatTensor(s_), 0)

        v = self.critic.forward(state)
        v_ = self.critic.forward(state_)

        loss = self.td_error(r, v_, v)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


