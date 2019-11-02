import pandas as pd
import numpy as np
from src.DDPG_BN.config import config

def test_env(budget, test_data, eCPC, actions):

    e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
    e_cost = [0 for i in range(24)]
    init_action = 0
    next_action = 0

    break_time_slot = 0
    bid_nums = [0 for i in range(24)]
    imps = [0 for i in range(24)]

    e_bids = np.array([])
    e_market_prices = np.array([])
    # 状态包括：当前CTR，
    for t in range(24):
        auc_datas = test_data[test_data[:, config['data_hour_index']] == t]

        bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + actions[t])

        bids = np.where(bids >= 300, 300, bids)

        market_prices = auc_datas[:, config['data_marketprice_index']]
        win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]

        e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
        e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']], dtype=int)

        imps[t] = len(win_auctions)
        bid_nums[t] = len(auc_datas)

        if np.sum(e_cost) >= budget:
            # print('早停时段{}'.format(t))
            break_time_slot = t
            temp_cost = 0
            temp_win_auctions = 0
            e_clks[t] = 0
            imps[t] = 0
            bid_nums[t] = 0

            bids = []
            market_prices = []
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
                bid_nums[t] += 1

                bids.append(bid)
                market_prices.append(temp_market_price)
                if bid > temp_market_price:
                    e_clks[t] += temp_clk
                    imps[t] += 1
                    temp_cost += temp_market_price
                    temp_win_auctions += 1
            e_cost[t] = temp_cost
            imps[t] = temp_win_auctions

        e_bids = np.hstack((e_bids, bids))
        e_market_prices = np.hstack((e_market_prices, market_prices))
        if np.sum(e_cost) >= budget:
            break

    print(e_market_prices.tolist())
    print(e_bids.tolist(), e_clks, imps, bid_nums)

budget_para = 0.5
directory = 'result_adjust_reward'
action_file = directory + '/test_episode_actions_' + str(budget_para) + '.csv'
actions = pd.read_csv(action_file, header=None).drop([0]).iloc[0, 1:].tolist() # 后面应该更新为选择最优的对应的actions

test_data = pd.read_csv('../../data/' + config['campaign_id'] + '/test_data.csv', header=None).drop([0])
test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
    = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
    int)
test_data.iloc[:, config['data_pctr_index']] \
    = test_data.iloc[:, config['data_pctr_index']].astype(
    float)
test_data = test_data.values

budget = config['test_budget'] * budget_para

eCPC = 60920.22773088766 # 每次点击花费
    # 由启发式算法得到最优eCPC 1458-60920.22773088766,38767.41764692851,33229.21512593873, 22152.81008395915‬
    # 3386-77901.22125145316‬,47939.21307781733,35954.409808363,23969.60653890866‬

test_env(budget, test_data, eCPC, actions)