import torch
import torch.nn as nn
import torch.nn.functional as F
from src.AC.No_threshold_no_reward.config import config

class Actor_net(nn.Module):
    def __init__(self, feature_nums, action_nums):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(feature_nums, config['neuron_nums'])
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(config['neuron_nums'], action_nums)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        y = self.fc2(x)
        y = torch.sigmoid(y)

        return y

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

    def loss_func(self, y_pred, td_error):
        log_prob = torch.log(y_pred)
        loss = torch.mean(log_prob * td_error)
        return loss


    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)

        y_pred = self.actor.forward(state).numpy()

        return y_pred

    def learn(self, y_pred, td_error):
        loss = self.loss_func(y_pred, td_error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



