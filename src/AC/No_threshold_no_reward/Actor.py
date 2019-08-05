import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from src.AC.No_threshold_no_reward.config import config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(1)

def store_para_a(Net):
    torch.save(Net.state_dict(), 'Model/actor_model_params.pth')

class Actor_net(nn.Module):
    def __init__(self, feature_nums, action_nums):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(feature_nums, config['neuron_nums'])
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(config['neuron_nums'], config['neuron_nums'])
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(config['neuron_nums'], action_nums)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x_ = self.fc2(x)
        x_ = F.relu(x_)
        out = self.out(x_)
        action_prob = F.softmax(out, dim=1)
        return action_prob

class Actor():
    def __init__(self,
                 feature_nums,
                 action_nums,
                 learning_rate=0.01,
                 reward_decay=1):
        self.feature_nums = feature_nums
        self.action_nums = action_nums
        self.lr = learning_rate
        self.gamma = reward_decay

        self.actor = Actor_net(self.feature_nums, self.action_nums)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def loss_func(self, action_probs, td_error): # action_probs按照真实动作来选
        log_prob = torch.log(action_probs)
        td_error = torch.FloatTensor(td_error)
        loss = -torch.mean(log_prob * td_error)
        return loss

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_probs = self.actor.forward(state).detach().numpy()
        action = np.random.choice(range(1, action_probs.shape[1]+1), p=action_probs.ravel())
        return action

    def choose_best_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_probs = self.actor.forward(state).detach().numpy()
        action = np.argmax(action_probs)
        return action

    def learn(self, s, a, td_error):
        state = torch.unsqueeze(torch.FloatTensor(s), 0)
        a = torch.unsqueeze(torch.LongTensor([a]), 1)
        act_probs = self.actor.forward(state).gather(1, a-1)
        loss = self.loss_func(act_probs, td_error)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





