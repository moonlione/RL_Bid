import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

np.random.seed(1)

class Net(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_numbers, config['neuron_nums'])
        self.fc1.weight.data.normal_(0, 0.1)  # 全连接隐层 1 的参数初始化
        self.fc2 = nn.Linear(config['neuron_nums'], config['neuron_nums'])
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(config['neuron_nums'], action_numbers)
        self.out.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x_ = self.fc2(x)
        x_ = F.relu(x_)
        y = self.out(x_)
        actions_value = F.softmax(y, dim=1)
        return actions_value

def store_para(Net, model_name):
    torch.save(Net.state_dict(), 'Model/PG_model_params.pth')

class PolicyGradientForTest:
    def __init__(
            self,
            action_nums,
            feature_nums,
            model_name,
    ):
        self.action_nums = action_nums
        self.feature_nums = feature_nums
        self.model_name = model_name

        self.policy_net = Net(self.feature_nums, self.action_nums).cuda()
        self.policy_net.load_state_dict(torch.load('Model/PG_model_params.pth'))

    # 依据概率来选择动作，本身具有随机性
    def choose_best_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()
        prob_weights = self.policy_net.forward(state).detach().cpu().numpy()
        action = np.argmax(prob_weights) + 1
        return action