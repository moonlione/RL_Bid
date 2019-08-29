from src.Temp_seg_action.env import AD_env
from src.Temp_seg_action.RL_brain import PolicyGradient
from src.Temp_seg_action.RL_brain import store_para
# import src.PG.No_threshold_no_reward.run_this_for_test as r_test
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config

def run_env(budget, auc_num, budget_para):
    env.build_env(budget, auc_num)  # 参数为训练集的(预算， 预期展示次数)
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

    eCPC = 30000  # 每次点击花费
    for episode in range(config['train_episodes']):
        e_clks = [0 for i in range(24)] # episode各个时段所获得的点击数，以下类推
        e_cost = [0 for i in range(24)]
        e_aucs = [0 for i in range(24)]
        e_actions = [0 for i in range(24)]
        init_action = 0
        next_action = 0

        state_ = np.array([])
        # 状态包括：当前CTR，
        for t in range(24):
            auc_datas = train_data[train_data[:, config['data_hour_index']] == t]
            if t == 0:
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
                win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
                state = np.array([1, 1, 0])
                action = RL.choose_action(state)
            else:
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + next_action)
                win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
                state = state_
                action = next_action
            e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
            e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']])
            e_aucs[t] = len(auc_datas)
            if np.sum(e_cost) >= budget:
                # print('早停时段{}'.format(t))
                temp_cost = 0
                temp_aucs = 0
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
                        temp_aucs += 1
                e_cost[t] = temp_cost
                e_aucs[t] = temp_aucs
                break

            ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions)
            state_ = np.array([(budget - np.sum(e_cost[:t+1])) / budget, (auc_num - np.sum(e_aucs[:t+1])) / auc_num, ctr_t])
            action_ = RL.choose_action(state_)
            next_action = action_
            if t == 0:
                e_actions[0] = init_action
            else:
                e_actions[t] = action_
            reward = e_clks[t]
            RL.store_transition(state, action, reward)
        loss, vt = RL.learn()
        if episode % 100 == 0:
            print('episode {}, budget-{}, cost-{}, clks-{}, loss-{}\n'.format(episode, budget, np.sum(e_cost), int(np.sum(e_clks)), loss))
        if episode % 100 == 0:
            test_env(config['test_budget'] * budget_para, int(config['test_auc_num']), budget_para)

def test_env(budget, auc_num, budget_para):
    env.build_env(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../data/test_data.csv", header=None).drop([0])
    test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    test_data.iloc[:, config['data_pctr_index']] \
        = test_data.iloc[:, config['data_pctr_index']].astype(
        float)
    test_data = test_data.values
    eCPC = 30000  # 每次点击花费
    e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
    e_cost = [0 for i in range(24)]
    e_aucs = [0 for i in range(24)]
    e_actions = [0 for i in range(24)]
    init_action = 0
    next_action = 0

    actions = []
    state_ = np.array([])
    # 状态包括：当前CTR，
    for t in range(24):
        auc_datas = test_data[test_data[:, config['data_hour_index']] == t]

        if t == 0:
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            state = np.array([1,1, 0])
            action = RL.choose_best_action(state)
        else:
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + next_action)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            state = state_
            action = next_action
        e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
        e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']])
        if np.sum(e_cost) >= budget:
            # print('早停时段{}'.format(t))
            temp_cost = 0
            temp_aucs = 0
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
                    temp_aucs += 1
            e_cost[t] = temp_cost
            e_aucs[t] = temp_aucs
            break
        ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions)
        state_ = np.array([(budget - np.sum(e_cost[:t+1])) / budget, (auc_num - np.sum(e_aucs[:t+1])) / auc_num, ctr_t])
        action_ = RL.choose_best_action(state_)
        next_action = action_
        if t == 0:
            e_actions[0] = init_action
        else:
            e_actions[t] = action_
        reward = e_clks[t]
        actions.append(action)
        RL.store_transition(state, action, reward)
    print('-----------测试结果-----------\n')
    print('budget-{}, cost-{}, clks-{}, loss-{}\n'.format(budget, np.sum(e_cost), int(np.sum(e_clks)), loss))

    actions_df = pd.DataFrame(data=actions)
    actions_df.to_csv('result/test_action_' + str(budget_para) + '.csv')

if __name__ == '__main__':
    env = AD_env()
    RL = PolicyGradient(
        action_nums=env.action_numbers,
        feature_nums=3,
        learning_rate=config['pg_learning_rate'],
        reward_decay=config['reward_decay'],
    )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        train_budget = config['train_budget'] * budget_para[i]
        run_env(train_budget, int(config['train_auc_num']), budget_para[i])
        # print('########测试结果########\n')
        # r_test.to_test(budget_para[i])
