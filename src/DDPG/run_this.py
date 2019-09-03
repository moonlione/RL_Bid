from src.DDPG.RL_brain import DDPG
from src.DDPG.config import config
from src.DDPG.RL_brain import OrnsteinUhlenbeckNoise
import pandas as pd
import numpy as np

def run_env(budget, budget_para):
    # 训练
    print('data loading')
    train_data = pd.read_csv("../../data/train_data.csv")
    train_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = train_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    train_data.iloc[:, config['data_pctr_index']] \
        = train_data.iloc[:, config['data_pctr_index']].astype(
        float)
    train_data = train_data.values

    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    td_error, action_loss = 0,0
    eCPC = 50000  # 每次点击花费
    for episode in range(config['train_episodes']):
        e_clks = [0 for i in range(24)] # episode各个时段所获得的点击数，以下类推
        e_cost = [0 for i in range(24)]
        e_actions = [0 for i in range(24)]
        init_action = 0
        next_action = 0

        state_ = np.array([])
        # 状态包括：当前CTR，
        for t in range(24):
            auc_datas = train_data[train_data[:, config['data_hour_index']] == t]
            if t == 0:
                state = np.array([(t + 1) / 24, 1, 0, 0, 0, 0]) # current_time_slot, budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t
                action = RL.choose_action(state)
                action = action + ou_noise()[0]
                init_action = action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
                win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            else:
                state = state_
                action = next_action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
                win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
            e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']])
            if np.sum(e_cost) >= budget:
                # print('早停时段{}'.format(t))
                temp_cost = 0
                temp_win_auctions = 0
                for i in range(len(auc_datas)):
                    if temp_cost >= (budget - np.sum(e_cost[:t])):
                        break
                    current_data = auc_datas[i, :]
                    temp_market_price = current_data[config['data_marketprice_index']]
                    if t == 0:
                        temp_action = init_action
                    else:
                        temp_action = next_action
                    bid = current_data[config['data_pctr_index']] * eCPC / (1 + temp_action)
                    if bid > temp_market_price:
                        e_clks[t] += current_data[config['data_clk_index']]
                        temp_cost += temp_market_price
                        temp_win_auctions += 1
                e_cost[t] = temp_cost
                ctr_t = e_clks[t] / temp_win_auctions if temp_win_auctions > 0 else 0
                win_rate_t = temp_win_auctions / len(auc_datas)
            else:
                ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions) if len(win_auctions) > 0 else 0
                win_rate_t = len(win_auctions) / len(auc_datas)
            budget_left_ratio = (budget - np.sum(e_cost[:t + 1])) / budget
            budget_left_ratio = budget_left_ratio if budget_left_ratio >= 0 else 0
            cost_t_ratio = e_cost[t] / budget
            next_time_slot = (t + 2) / 24
            if t == 0:
                state_ = np.array([next_time_slot, budget_left_ratio, cost_t_ratio, 1, ctr_t, win_rate_t])
            else:
                budget_spent_speed = (e_cost[t] - e_cost[t-1]) / e_cost[t-1] if e_cost[t-1] > 0 else 1
                state_ = np.array([next_time_slot, budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t])
            action_ = RL.choose_action(state_)
            action_ = action_ + ou_noise()[0]
            next_action = action_
            if t == 0:
                e_actions[0] = init_action
            else:
                e_actions[t] = action_
            reward = e_clks[t]
            transition = np.hstack((state.tolist(), action, reward, state_.tolist()))
            RL.store_transition(transition)
            if np.sum(e_cost) >= budget:
                break
        if RL.memory_counter >= config['batch_size']:
            td_e, a_loss = RL.learn()
            td_error, action_loss = td_e, a_loss
            RL.soft_update(RL.Actor, RL.Actor_)
            RL.soft_update(RL.Critic, RL.Critic_)
        if episode % 100 == 0:
            print('episode {}, budget={}, cost={}, clks={}, td_error={}, action_loss={}\n'.format(episode, budget, np.sum(e_cost), int(np.sum(e_clks)), td_error, action_loss))
            test_env(config['test_budget'] * budget_para, budget_para)

def test_env(budget, budget_para):
    test_data = pd.read_csv("../../data/test_data.csv", header=None).drop([0])
    test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    test_data.iloc[:, config['data_pctr_index']] \
        = test_data.iloc[:, config['data_pctr_index']].astype(
        float)
    test_data = test_data.values
    eCPC = 50000  # 每次点击花费
    e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
    e_cost = [0 for i in range(24)]
    e_actions = [0 for i in range(24)]
    init_action = 0
    next_action = 0

    actions = []
    state_ = np.array([])
    # 状态包括：当前CTR，
    for t in range(24):
        auc_datas = test_data[test_data[:, config['data_hour_index']] == t]
        if t == 0:
            state = np.array([(t + 1) / 24, 1, 0, 0, 0,
                              0])  # current_time_slot, budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t
            action = RL.choose_action(state)
            action = action
            init_action = action
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
        else:
            state = state_
            action = next_action
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
        e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
        e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']])
        if np.sum(e_cost) >= budget:
            # print('早停时段{}'.format(t))
            temp_cost = 0
            temp_win_auctions = 0
            for i in range(len(auc_datas)):
                if temp_cost >= (budget - np.sum(e_cost[:t])):
                    break
                current_data = auc_datas[i, :]
                temp_market_price = current_data[config['data_marketprice_index']]
                if t == 0:
                    temp_action = init_action
                else:
                    temp_action = next_action
                bid = current_data[config['data_pctr_index']] * eCPC / (1 + temp_action)
                if bid > temp_market_price:
                    e_clks[t] += current_data[config['data_clk_index']]
                    temp_cost += temp_market_price
                    temp_win_auctions += 1
            e_cost[t] = temp_cost
            ctr_t = e_clks[t] / temp_win_auctions if temp_win_auctions > 0 else 0
            win_rate_t = temp_win_auctions / len(auc_datas)
        else:
            ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions) if len(
                win_auctions) > 0 else 0
            win_rate_t = len(win_auctions) / len(auc_datas)
        budget_left_ratio = (budget - np.sum(e_cost[:t + 1])) / budget
        budget_left_ratio = budget_left_ratio if budget_left_ratio >= 0 else 0
        cost_t_ratio = e_cost[t] / budget
        next_time_slot = (t + 2) / 24
        if t == 0:
            state_ = np.array([next_time_slot, budget_left_ratio, cost_t_ratio, 1, ctr_t, win_rate_t])
        else:
            budget_spent_speed = (e_cost[t] - e_cost[t - 1]) / e_cost[t - 1] if e_cost[t - 1] > 0 else 1
            state_ = np.array([next_time_slot, budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t])
        action_ = RL.choose_action(state_)
        next_action = action_
        if t == 0:
            e_actions[0] = init_action
        else:
            e_actions[t] = action_
        actions.append(action)
        if np.sum(e_cost) >= budget:
            break
    print('-----------测试结果-----------\n')
    print('budget={}, cost={}, clks={}\n'.format(budget, np.sum(e_cost), int(np.sum(e_clks))))

    actions_df = pd.DataFrame(data=actions)
    actions_df.to_csv('result/test_action_' + str(budget_para) + '.csv')

if __name__ == '__main__':
    RL = DDPG(
        feature_nums = config['feature_num'],
        action_nums = 1,
        lr_A = config['learning_rate_a'],
        lr_C = config['learning_rate_c'],
        reward_decay = config['reward_decay'],
        memory_size = config['memory_size'],
        batch_size = config['batch_size'],
        tau = config['tau'],  # for target network soft update
    )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        train_budget = config['train_budget'] * budget_para[i]
        run_env(train_budget, budget_para[i])