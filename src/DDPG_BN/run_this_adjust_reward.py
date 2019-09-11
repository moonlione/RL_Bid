from src.DDPG_BN.RL_brain import DDPG
from src.DDPG_BN.config import config
from src.DDPG_BN.RL_brain import OrnsteinUhlenbeckNoise
import pandas as pd
import numpy as np
import datetime

def adjust_reward(e_true_value, e_miss_true_value, bids_t, market_prices_t, e_win_imp_with_clk_value, e_cost, e_win_imp_without_clk_cost, real_clks,
                  e_lose_imp_with_clk_value,
                  e_clk_aucs,
                  e_clk_no_win_aucs, e_lose_imp_without_clk_cost, e_no_clk_aucs, e_no_clk_no_win_aucs, budget, total_clks, t):
    reward_degree = 1 - np.square(np.mean(np.true_divide(np.subtract(bids_t, market_prices_t), bids_t)))
    reward_win_imp_with_clk = e_win_imp_with_clk_value[t] / e_true_value[t] * reward_degree
    reward_win_imp_with_clk = reward_win_imp_with_clk if reward_degree > 0 else 0

    remain_budget = (budget - np.sum(e_cost[:t+1])) / budget
    remain_budget = remain_budget if remain_budget > 0 else 1e-1 # 1e-1防止出现除0错误
    remain_clks = (total_clks - np.sum(real_clks[:t+1])) / total_clks
    punish_win_rate = remain_clks / remain_budget
    reward_win_imp_without_clk = - np.sum(e_win_imp_without_clk_cost[t]) * punish_win_rate / e_cost[t]

    temp_rate = (e_clk_no_win_aucs[t] / e_clk_aucs[t]) if e_clk_aucs[t] > 0 else 1
    punish_no_win_rate = 1 - temp_rate if temp_rate != 1 else 1
    base_punishment = e_lose_imp_with_clk_value[t] / e_miss_true_value[t] if e_miss_true_value[t] > 0 else 0
    reward_lose_imp_with_clk = - base_punishment / punish_no_win_rate

    base_encourage = np.sum(e_lose_imp_without_clk_cost[t]) / e_cost[t]
    encourage_rate = 1 - (e_no_clk_no_win_aucs[t] / e_no_clk_aucs[t])
    reward_lose_imp_without_clk = base_encourage / encourage_rate if encourage_rate > 0 else 1
    reward_t = reward_win_imp_with_clk + reward_win_imp_without_clk + reward_lose_imp_with_clk + reward_lose_imp_without_clk
    return reward_t / 1e3

def run_env(budget, budget_para):
    # 训练
    print('data loading')
    test_data = pd.read_csv("../../data/test_data.csv", header=None).drop([0])
    test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    test_data.iloc[:, config['data_pctr_index']] \
        = test_data.iloc[:, config['data_pctr_index']].astype(
        float)
    test_data = test_data.values

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
    eCPC = 50000  # 每次点击花费

    e_results = []
    test_records = []
    for episode in range(config['train_episodes']):
        e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
        e_profits = [0 for i in range(24)]
        e_reward = [0 for i in range(24)]
        e_cost = [0 for i in range(24)]

        e_true_value = [0 for i in range(24)]
        e_miss_true_value = [0 for i in range(24)]
        e_win_imp_with_clk_value = [0 for i in range(24)]
        e_win_imp_without_clk_cost = [0 for i in range(24)] # 各个时段浪费在没有点击的曝光上的预算
        e_lose_imp_with_clk_value = [0 for i in range(24)]
        e_clk_aucs = [0 for i in range(24)]
        e_clk_no_win_aucs = [0 for i in range(24)]
        e_lose_imp_without_clk_cost = [0 for i in range(24)]
        e_no_clk_aucs = [0 for i in range(24)]
        e_no_clk_no_win_aucs = [0 for i in range(24)]

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
            else:
                state = state_
                action = next_action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
                bids = np.where(bids >= 300, 300, bids)
            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            no_win_auctions = auc_datas[bids <= auc_datas[:, config['data_marketprice_index']]]
            e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
            e_profits[t] = np.sum(win_auctions[:, config['data_pctr_index']] * eCPC - win_auctions[:, config['data_marketprice_index']])

            e_true_value[t] = np.sum(win_auctions[:, config['data_pctr_index']] * eCPC)
            e_miss_true_value[t] = np.sum(no_win_auctions[:, config['data_pctr_index']] * eCPC)
            with_clk_win_auctions = win_auctions[win_auctions[:, config['data_clk_index']] == 1]
            e_win_imp_with_clk_value[t] = np.sum(with_clk_win_auctions[:, config['data_pctr_index']] * eCPC)
            e_win_imp_without_clk_cost[t] = np.sum(win_auctions[win_auctions[:, config['data_clk_index']] == 0][:, config['data_marketprice_index']])
            with_clk_no_win_auctions = no_win_auctions[no_win_auctions[:, config['data_clk_index']] == 1]
            e_lose_imp_with_clk_value[t] = np.sum(with_clk_no_win_auctions[:, config['data_pctr_index']] * eCPC)

            e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']], dtype=int)
            imps[t] = len(win_auctions)
            real_clks[t] = np.sum(auc_datas[:, config['data_clk_index']], dtype=int)
            bid_nums[t] = len(auc_datas)
            
            e_clk_aucs[t] = len(auc_datas[auc_datas[:, config['data_clk_index']] == 1])
            e_clk_no_win_aucs[t] = len(with_clk_no_win_auctions)

            e_no_clk_aucs[t] = len(auc_datas[auc_datas[:, config['data_clk_index']] == 0])

            without_clk_no_win_auctions = no_win_auctions[no_win_auctions[:, config['data_clk_index']] == 0]
            e_lose_imp_without_clk_cost[t] = np.sum(without_clk_no_win_auctions[:, config['data_marketprice_index']])
            e_no_clk_no_win_aucs[t] = len(without_clk_no_win_auctions)

            bids_t = bids
            market_prices_t = auc_datas[:, config['data_marketprice_index']]
            if np.sum(e_cost) >= budget:
                # print('早停时段{}'.format(t))
                break_time_slot = t
                temp_cost = 0
                temp_win_auctions = 0
                e_clks[t] = 0
                e_profits[t] = 0

                e_true_value[t] = 0
                e_miss_true_value[t] = 0

                e_win_imp_without_clk_cost[t] = 0
                e_lose_imp_with_clk_value[t] = 0
                real_clks[t] = 0
                imps[t] = 0
                bid_nums[t] = 0

                e_win_imp_with_clk_value[t] = 0
                e_clk_aucs[t] = 0
                e_lose_imp_without_clk_cost[t] = 0
                e_no_clk_aucs[t] = 0
                e_no_clk_no_win_aucs[t] = 0

                bids_t = []
                market_prices_t = []
                for i in range(len(auc_datas)):
                    if temp_cost >= (budget - np.sum(e_cost[:t])):
                        break
                    current_data = auc_datas[i, :]
                    temp_clk = int(current_data[config['data_clk_index']])
                    temp_market_price = current_data[config['data_marketprice_index']]
                    if t == 0:
                        temp_action = init_action
                    else:
                        temp_action = next_action
                    bid = current_data[config['data_pctr_index']] * eCPC / (1 + temp_action)
                    bid = bid if bid <= 300 else 300
                    real_clks[t] += temp_clk
                    bid_nums[t] += 1

                    if temp_clk == 1:
                        e_clk_aucs[t] += temp_clk
                    else:
                        e_no_clk_aucs[t] += 1
                    bids_t.append(bid)
                    market_prices_t.append(temp_market_price)
                    if bid > temp_market_price:
                        if temp_clk == 0:
                            e_win_imp_without_clk_cost[t] += temp_market_price
                        else:
                            e_win_imp_with_clk_value[t] += (current_data[config['data_pctr_index']] * eCPC - temp_market_price)
                        e_profits[t] += (current_data[config['data_pctr_index']] * eCPC - temp_market_price)
                        e_true_value[t] += current_data[config['data_pctr_index']] * eCPC
                        e_clks[t] += temp_clk
                        imps[t] += 1
                        temp_cost += temp_market_price
                        temp_win_auctions += 1
                    else:
                        e_miss_true_value[t] += current_data[config['data_pctr_index']] * eCPC
                        if temp_clk == 1:
                            e_clk_no_win_aucs[t] += 1
                            e_lose_imp_with_clk_value[t] += (current_data[config['data_pctr_index']] * eCPC - temp_market_price)
                        else:
                            e_no_clk_no_win_aucs[t] += 1
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
            reward_t = adjust_reward(e_true_value, e_miss_true_value, bids_t, market_prices_t, e_win_imp_with_clk_value, e_cost, e_win_imp_without_clk_cost, real_clks,
                  e_lose_imp_with_clk_value,
                  e_clk_aucs,
                  e_clk_no_win_aucs, e_lose_imp_without_clk_cost, e_no_clk_aucs, e_no_clk_no_win_aucs, budget, total_clks, t)
            reward = reward_t
            e_reward[t] = reward
            transition = np.hstack((state.tolist(), action, reward, state_.tolist()))
            RL.store_transition(transition)

            if RL.memory_counter >= config['batch_size']:
                td_e, a_loss = RL.learn()
                td_error, action_loss = td_e, a_loss
                RL.soft_update(RL.Actor, RL.Actor_)
                RL.soft_update(RL.Critic, RL.Critic_)
            if np.sum(e_cost) >= budget:
                break

        e_result = [np.sum(e_reward), np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)), int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
                    np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, td_error, action_loss]
        e_results.append(e_result)

        if (episode > 0) and (episode % 100 == 0):
            actions_df = pd.DataFrame(data=actions)
            actions_df.to_csv('result_adjust_reward/train_actions_' + str(budget_para) + '.csv')

            hour_clks = {'clks': e_clks, 'no_bid_clks': np.subtract(real_hour_clks, e_clks).tolist(),
                         'real_clks': real_hour_clks}
            hour_clks_df = pd.DataFrame(data=hour_clks)
            hour_clks_df.to_csv('result_adjust_reward/train_hour_clks_' + str(budget_para) + '.csv')
            print(
                'episode {}, reward={}, profits={}, budget={}, cost={}, clks={}, real_clks={}, bids={}, imps={}, cpm={}, break_time_slot={}, td_error={}, action_loss={}\n'.format(
                    episode, np.sum(e_reward), np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)),
                    int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
                    np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, td_error, action_loss))
            test_result, test_actions, test_hour_clks = test_env(config['test_budget'] * budget_para, budget_para, test_data)
            test_records.append(test_result)

            max = RL.para_store_iter(test_records)
            if max == test_records[len(test_records) - 1:len(test_records)][0][3]:
                # print('最优参数已存储')
                results = []
                results.append(test_result)
                result_df = pd.DataFrame(data=results,
                                         columns=['profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                                  'break_time_slot'])
                result_df.to_csv('result_adjust_reward/best_test_result_' + str(budget_para) + '.csv')

                test_actions_df = pd.DataFrame(data=test_actions)
                test_actions_df.to_csv('result_adjust_reward/best_test_action_' + str(budget_para) + '.csv')

                test_hour_clks_df = pd.DataFrame(data=test_hour_clks)
                test_hour_clks_df.to_csv('result_adjust_reward/best_test_hour_clks_' + str(budget_para) + '.csv')

    e_results_df = pd.DataFrame(data=e_results, columns=['reward', 'profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                                         'break_time_slot', 'td_error', 'action_loss'])
    e_results_df.to_csv('result_adjust_reward/train_epsiode_results_' + str(budget_para) + '.csv')

    test_records_df = pd.DataFrame(data=test_records,
                                   columns=['profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                            'break_time_slot'])
    test_records_df.to_csv('result_adjust_reward/test_epsiode_results_' + str(budget_para) + '.csv')


def test_env(budget, budget_para, test_data):
    real_hour_clks = []
    for i in range(24):
        real_hour_clks.append(
            np.sum(test_data[test_data[:, config['data_hour_index']] == i][:, config['data_clk_index']]))

    eCPC = 50000  # 每次点击花费
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
        e_profits[t] = np.sum(win_auctions[:, config['data_pctr_index']] * eCPC - win_auctions[:, config['data_marketprice_index']])
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
                temp_clk = int(current_data[config['data_clk_index']])
                temp_market_price = current_data[config['data_marketprice_index']]
                if t == 0:
                    temp_action = init_action
                else:
                    temp_action = next_action
                bid = current_data[config['data_pctr_index']] * eCPC / (1 + temp_action)
                bid = bid if bid <= 300 else 300
                real_clks[t] += temp_clk
                bid_nums[t] += 1
                if bid > temp_market_price:
                    e_profits[t] += (current_data[config['data_pctr_index']] * eCPC - temp_market_price)
                    e_clks[t] += temp_clk
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
    result_df.to_csv('result_adjust_reward/test_result_' + str(budget_para) + '.csv')

    test_actions_df = pd.DataFrame(data=actions)
    test_actions_df.to_csv('result_adjust_reward/test_action_' + str(budget_para) + '.csv')

    test_hour_clks_df = pd.DataFrame(data=hour_clks)
    test_hour_clks_df.to_csv('result_adjust_reward/test_hour_clks_' + str(budget_para) + '.csv')
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