from src.DDPG_BN.RL_brain import DDPG
from src.DDPG_BN.config import config
from src.DDPG_BN.RL_brain import OrnsteinUhlenbeckNoise
import pandas as pd
import numpy as np
import datetime


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

    total_clks = np.sum(train_data[:, config['data_clk_index']])
    real_hour_clks = []
    for i in range(24):
        real_hour_clks.append(
            np.sum(train_data[train_data[:, config['data_hour_index']] == i][:, config['data_clk_index']]))

    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    td_error, action_loss = 0, 0
    eCPC = budget / budget_para / total_clks  # 每次点击花费，由历史数据得到

    e_results = []
    test_records = []

    is_learn = False
    for episode in range(config['train_episodes']):
        e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
        e_profits = [0 for i in range(24)]
        e_cost = [0 for i in range(24)]
        actions = [0 for i in range(24)]
        init_action = 0
        next_action = 0

        state_ = np.array([])

        break_time_slot = 0
        real_clks = [0 for i in range(24)]
        bid_nums = [0 for i in range(24)]
        imps = [0 for i in range(24)]

        # 状态包括：当前CTR，
        for t in range(24):
            auc_datas = train_data[train_data[:, config['data_hour_index']] == t]
            if t == 0:
                state = np.array([1, 0, 0, 0, 0])  # current_time_slot, budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t
                action = RL.choose_action(state)
                action = np.clip(action + ou_noise()[0], -0.99, 0.99)
                init_action = action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
                bids = np.where(bids >= 300, 300, bids)
                win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            else:
                state = state_
                action = next_action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
                bids = np.where(bids >= 300, 300, bids)
                win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
            e_profits[t] = np.sum(bids[bids >= auc_datas[:, config['data_marketprice_index']]] - win_auctions[:, config[
                                                                                                                     'data_marketprice_index']])
            e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']], dtype=int)
            imps[t] = len(win_auctions)
            real_clks[t] = np.sum(auc_datas[:, config['data_clk_index']], dtype=int)
            bid_nums[t] = len(auc_datas)

            if np.sum(e_cost) >= budget:
                # print('早停时段{}'.format(t))
                break_time_slot = t
                temp_cost = 0
                temp_win_auctions = 0
                e_clks[t] = 0
                e_profits[t] = 0
                real_clks[t] = 0
                imps[t] = 0
                bid_nums[t] = 0
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
                    bid = bid if bid <= 300 else 300
                    real_clks[t] += int(current_data[config['data_clk_index']])
                    bid_nums[t] += 1

                    if bid > temp_market_price:
                        e_profits[t] += (bid - temp_market_price)
                        e_clks[t] += int(current_data[config['data_clk_index']])
                        imps[t] += 1
                        temp_cost += temp_market_price
                        temp_win_auctions += 1
                e_cost[t] = temp_cost
                ctr_t = e_clks[t] / temp_win_auctions if temp_win_auctions > 0 else 0
                win_rate_t = temp_win_auctions / bid_nums[t]
            else:
                ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions) if len(
                    win_auctions) > 0 else 0
                win_rate_t = len(win_auctions) / len(auc_datas)
            budget_left_ratio = (budget - np.sum(e_cost[:t + 1])) / budget
            budget_left_ratio = budget_left_ratio if budget_left_ratio >= 0 else 0
            cost_t_ratio = e_cost[t] / budget
            if t == 0:
                state_ = np.array([budget_left_ratio, cost_t_ratio, 1, ctr_t, win_rate_t])
            else:
                budget_spent_speed = (e_cost[t] - e_cost[t - 1]) / e_cost[t - 1] if e_cost[t - 1] > 0 else 1
                state_ = np.array(
                    [budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t])
            action_ = RL.choose_action(state_)
            action_ = np.clip(action_ + ou_noise()[0], -0.99, 0.99)
            next_action = action_
            if t == 0:
                actions[0] = init_action
            else:
                actions[t] = action_
            reward = e_clks[t]
            transition = np.hstack((state.tolist(), action, reward, state_.tolist()))
            RL.store_transition(transition)

            if RL.memory_counter % config['observation_size'] == 0:
                is_learn = True
            if is_learn: # after observing config['observation_size'] times, for config['learn_iter'] learning time
                for m in range(config['learn_iter']):
                    td_e, a_loss = RL.learn()
                    td_error, action_loss = td_e, a_loss
                    RL.soft_update(RL.Actor, RL.Actor_)
                    RL.soft_update(RL.Critic, RL.Critic_)
                    if m == config['learn_iter'] - 1:
                        is_learn = False
            if np.sum(e_cost) >= budget:
                break

        e_result = [budget, np.sum(e_profits), np.sum(e_cost), int(np.sum(e_clks)), int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
                    np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, td_error, action_loss]
        e_results.append(e_result)

        if (episode > 0) and (episode % 100 == 0):
            actions_df = pd.DataFrame(data=actions)
            actions_df.to_csv('result/train_actions_' + str(budget_para) + '.csv')

            hour_clks = {'clks': e_clks, 'no_bid_clks': np.subtract(real_hour_clks, e_clks).tolist(),
                         'real_clks': real_hour_clks}
            hour_clks_df = pd.DataFrame(data=hour_clks)
            hour_clks_df.to_csv('result/train_hour_clks_' + str(budget_para) + '.csv')
            print(
                'episode {}, profits={}, budget={}, cost={}, clks={}, real_clks={}, bids={}, imps={}, cpm={}, break_time_slot={}, td_error={}, action_loss={}\n'.format(
                    episode, np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)),
                    int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
                    np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, td_error, action_loss))
            test_result, test_actions, test_hour_clks = test_env(config['test_budget'] * budget_para, budget_para, eCPC)
            test_records.append(test_result)

            max = RL.para_store_iter(test_records)
            if max == test_records[len(test_records) - 1:len(test_records)][0][3]:
                # print('最优参数已存储')
                results = []
                results.append(test_result)
                result_df = pd.DataFrame(data=results,
                                         columns=['profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                                  'break_time_slot'])
                result_df.to_csv('result/best_test_result_' + str(budget_para) + '.csv')

                test_actions_df = pd.DataFrame(data=test_actions)
                test_actions_df.to_csv('result/best_test_action_' + str(budget_para) + '.csv')

                test_hour_clks_df = pd.DataFrame(data=test_hour_clks)
                test_hour_clks_df.to_csv('result/best_test_hour_clks_' + str(budget_para) + '.csv')

    e_results_df = pd.DataFrame(data=e_results, columns=['profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                                         'break_time_slot', 'td_error', 'action_loss'])
    e_results_df.to_csv('result/train_epsiode_results_' + str(budget_para) + '.csv')

    test_records_df = pd.DataFrame(data=test_records,
                                   columns=['profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                            'break_time_slot'])
    test_records_df.to_csv('result/test_epsiode_results_' + str(budget_para) + '.csv')


def test_env(budget, budget_para, eCPC):
    test_data = pd.read_csv("../../data/test_data.csv", header=None).drop([0])
    test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    test_data.iloc[:, config['data_pctr_index']] \
        = test_data.iloc[:, config['data_pctr_index']].astype(
        float)
    test_data = test_data.values

    real_hour_clks = []
    for i in range(24):
        real_hour_clks.append(
            np.sum(test_data[test_data[:, config['data_hour_index']] == i][:, config['data_clk_index']]))

    e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
    e_cost = [0 for i in range(24)]
    e_profits = [0 for i in range(24)]
    init_action = 0
    next_action = 0
    actions = [0 for i in range(24)]
    state_ = np.array([])

    break_time_slot = 0
    real_clks = [0 for i in range(24)]
    bid_nums = [0 for i in range(24)]
    imps = [0 for i in range(24)]

    results = []
    # 状态包括：当前CTR，
    for t in range(24):
        auc_datas = test_data[test_data[:, config['data_hour_index']] == t]
        if t == 0:
            state = np.array([1, 0, 0, 0, 0])  # current_time_slot, budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t
            action = RL.choose_action(state)
            action = np.clip(action, -0.99, 0.99)
            init_action = action
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
            bids = np.where(bids >= 300, 300, bids)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
        else:
            state = state_
            action = next_action
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
            bids = np.where(bids >= 300, 300, bids)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
        e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
        e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']], dtype=int)
        e_profits[t] = np.sum(bids[bids >= auc_datas[:, config['data_marketprice_index']]] - win_auctions[:, config['data_marketprice_index']])
        imps[t] = len(win_auctions)
        real_clks[t] = np.sum(auc_datas[:, config['data_clk_index']], dtype=int)
        bid_nums[t] = len(auc_datas)
        if np.sum(e_cost) >= budget:
            # print('早停时段{}'.format(t))
            break_time_slot = t
            temp_cost = 0
            temp_win_auctions = 0
            e_clks[t] = 0
            real_clks[t] = 0
            imps[t] = 0
            bid_nums[t] = 0
            e_profits[t] = 0
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
                bid = bid if bid <= 300 else 300
                real_clks[t] += int(current_data[config['data_clk_index']])
                bid_nums[t] += 1
                if bid > temp_market_price:
                    e_profits[t] += (bid - temp_market_price)
                    e_clks[t] += int(current_data[config['data_clk_index']])
                    imps[t] += 1
                    temp_cost += temp_market_price
                    temp_win_auctions += 1
            e_cost[t] = temp_cost
            ctr_t = e_clks[t] / temp_win_auctions if temp_win_auctions > 0 else 0
            win_rate_t = temp_win_auctions / bid_nums[t]
        else:
            ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions) if len(
                win_auctions) > 0 else 0
            win_rate_t = len(win_auctions) / len(auc_datas)
        budget_left_ratio = (budget - np.sum(e_cost[:t + 1])) / budget
        budget_left_ratio = budget_left_ratio if budget_left_ratio >= 0 else 0
        cost_t_ratio = e_cost[t] / budget
        if t == 0:
            state_ = np.array([budget_left_ratio, cost_t_ratio, 1, ctr_t, win_rate_t])
        else:
            budget_spent_speed = (e_cost[t] - e_cost[t - 1]) / e_cost[t - 1] if e_cost[t - 1] > 0 else 1
            state_ = np.array([budget_left_ratio, cost_t_ratio, budget_spent_speed, ctr_t, win_rate_t])
        action_ = RL.choose_action(state_)
        action_ = np.clip(action_, -0.99, 0.99)
        next_action = action_
        if t == 0:
            actions[0] = init_action
        else:
            actions[t] = action_
        if np.sum(e_cost) >= budget:
            break
    print('-----------测试结果-----------\n')
    result = [np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)), int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
              np.sum(e_cost) / np.sum(imps), break_time_slot]
    hour_clks = {'clks': e_clks, 'no_bid_clks': np.subtract(real_hour_clks, e_clks).tolist(),
                 'real_clks': real_hour_clks}

    results.append(result)
    result_df = pd.DataFrame(data=results,
                             columns=['profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                      'break_time_slot'])
    result_df.to_csv('result/test_result_' + str(budget_para) + '.csv')

    test_actions_df = pd.DataFrame(data=actions)
    test_actions_df.to_csv('result/test_action_' + str(budget_para) + '.csv')

    test_hour_clks_df = pd.DataFrame(data=hour_clks)
    test_hour_clks_df.to_csv('result/test_hour_clks_' + str(budget_para) + '.csv')
    print('profits={}, budget={}, cost={}, clks={}, real_clks={}, bids={}, imps={}, cpm={}, break_time_slot={}, {}\n'.format(
        np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)),
        int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
        np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, datetime.datetime.now()))

    return result, actions, hour_clks


if __name__ == '__main__':
    RL = DDPG(
        feature_nums=config['feature_num'],
        action_nums=1,
        lr_A=config['learning_rate_a'],
        lr_C=config['learning_rate_c'],
        reward_decay=config['reward_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        tau=config['tau'],  # for target network soft update
    )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        train_budget = config['train_budget'] * budget_para[i]
        run_env(train_budget, budget_para[i])