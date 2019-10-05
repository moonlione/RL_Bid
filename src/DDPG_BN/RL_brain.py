import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from src.DDPG_BN.Actor_Critic import Actor, Critic

if not os.path.exists('result'):
    os.mkdir('result')
elif not os.path.exists('Model'):
    os.mkdir('Model')
elif not os.path.exists('result_profit'):
    os.mkdir('result_profit')
elif not os.path.exists('result_adjust_reward'):
    os.mkdir('result_adjust_reward')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

class DDPG():
    def __init__(
            self,
            feature_nums,
            action_nums,
            lr_A,
            lr_C,
            reward_decay,
            memory_size,
            batch_size = 32,
            tau = 0.001, # for target network soft update
    ):
        self.feature_numbers = feature_nums
        self.action_numbers = action_nums
        self.lr_A = lr_A
        self.lr_C = lr_C
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + self.action_numbers + 1))

        self.Actor = Actor(self.feature_numbers, self.action_numbers)
        self.Critic = Critic(self.feature_numbers, self.action_numbers)

        self.Actor_ = Actor(self.feature_numbers, self.action_numbers)
        self.Critic_ = Critic(self.feature_numbers, self.action_numbers)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Actor.parameters(), lr=self.lr_A)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-2)

        self.loss_func = nn.MSELoss(reduction='mean')

    def store_transition(self, transition):
        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换
        self.memory_counter += 1

    def choose_action(self, state, ):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor.forward(state).numpy()[0][0]
        self.Actor.train()

        return action

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self):
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(batch_memory[:, :self.feature_numbers])
        b_a = torch.FloatTensor(batch_memory[:, self.feature_numbers: self.feature_numbers + self.action_numbers])
        b_r = torch.FloatTensor(batch_memory[:, -self.feature_numbers - 1: -self.feature_numbers])
        b_s_ = torch.FloatTensor(batch_memory[:, -self.feature_numbers:])

        q_target = b_r + self.gamma * self.Critic_.forward(b_s_, self.Actor_(b_s_))
        q = self.Critic.forward(b_s, b_a)
        td_error = F.smooth_l1_loss(q, q_target.detach())
        self.optimizer_c.zero_grad()
        td_error.backward()
        self.optimizer_c.step()

        a_loss = -self.Critic.forward(b_s, self.Actor(b_s)).mean()
        self.optimizer_a.zero_grad()
        a_loss.backward()
        self.optimizer_a.step()

        td_error_r = td_error.detach().numpy()
        a_loss_r = a_loss.detach().numpy()
        return td_error_r, a_loss_r

    # 只存储获得最优收益（点击）那一轮的参数
    def para_store_iter(self, test_results):
        max = 0
        if len(test_results) >= 1:
            for i in range(len(test_results)):
                if i == 0:
                    max = test_results[i][3]
                elif i != len(test_results) - 1:
                    if test_results[i][3] > test_results[i - 1][3] and test_results[i][3] > test_results[i + 1][3]:
                        if max < test_results[i][3]:
                            max = test_results[i][3]
                else:
                    if test_results[i][3] > max:
                        max = test_results[i][3]
        return max

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
