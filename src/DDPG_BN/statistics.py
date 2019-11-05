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
    e_hours = np.array([])
    e_real_labels = np.array([]) # 是否被点击
    # 状态包括：当前CTR，
    for t in range(24):
        auc_datas = test_data[test_data[:, config['data_hour_index']] == t]

        bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + actions[t])

        bids = np.where(bids >= 300, 300, bids)

        market_prices = auc_datas[:, config['data_marketprice_index']]
        hours = auc_datas[:, config['data_hour_index']]
        real_labels = auc_datas[:, config['data_clk_index']]
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
            hours = []
            real_labels = []
            for i in range(len(auc_datas)):
                if temp_cost >= (budget - np.sum(e_cost[:t])):
                    break
                current_data = auc_datas[i, :]
                temp_clk = int(current_data[config['data_clk_index']])
                temp_market_price = current_data[config['data_marketprice_index']]

                bid = current_data[config['data_pctr_index']] * eCPC / (1 + actions[t])
                bid = bid if bid <= 300 else 300
                bid_nums[t] += 1

                bids.append(bid)
                market_prices.append(temp_market_price)
                hours.append(current_data[config['data_hour_index']])
                real_labels.append(temp_clk)
                if bid >= temp_market_price:
                    e_clks[t] += temp_clk
                    imps[t] += 1
                    temp_cost += temp_market_price
                    temp_win_auctions += 1
            e_cost[t] = temp_cost
            imps[t] = temp_win_auctions

        e_bids = np.hstack((e_bids, bids))
        e_market_prices = np.hstack((e_market_prices, market_prices))
        e_hours = np.hstack((e_hours, hours))
        e_real_labels = np.hstack((e_real_labels, real_labels))
        if np.sum(e_cost) >= budget:
            break

    print(imps)
    print(bid_nums)
    print(e_clks, np.sum(e_clks))

    records = {'bids': e_bids.tolist(), 'market_prices':e_market_prices.tolist(), 'clks': e_real_labels, 'hours': e_hours}
    records_df = pd.DataFrame(data=records)
    records_df.to_csv(directory + '/bids_' + str(config['budget_para'][0]) + '.csv', index=None)

def max_train_index(directory, para):
    train_results = pd.read_csv(directory + '/train_episode_results_' + str(para) + '.csv')
    train_clks = train_results.values[:, [0, 5]]

    test_results = pd.read_csv(directory + '/test_episode_results_' + str(para) + '.csv').values
    test_clks = test_results[:, [0, 4]]

    new_test_clks = []
    new_test_results = []
    for i in range(len(test_clks)):
        test_clk_temp = [test_clks[i, 1] for k in range(10)]
        new_test_clks.append(test_clk_temp)

        test_temp = [test_results[i, [0, 3, 4, 6, 7, 8]].tolist() for m in range(10)]
        new_test_results.append(test_temp)

    new_test_results = np.array(new_test_results).reshape(50000, 6)
    extend_test_clks = np.array(new_test_clks).flatten()

    max_value = train_clks[train_clks[:, 1].argsort()][-1, 1]
    max_value_indexs = train_clks[train_clks[:, 1] == max_value]

    max_test_value = []
    max_test_value_index = []
    for index in max_value_indexs:
        max_test_value.append(extend_test_clks[int(index[0])])
        max_test_value_index.append(int(index[0]))

    test_value_max_index = np.argmax(max_test_value)  # max_test_value 最大值的索引

    max_test_result = new_test_results[max_test_value_index[test_value_max_index], :].tolist()
    print(max_test_result)
    return int(max_test_result[0])

# 由启发式算法得到最优eCPC 1458-60920.22773088766,38767.41764692851,33229.21512593873, 22152.81008395915‬
# 3386-77901.22125145316‬,47939.21307781733,35954.409808363,23969.60653890866‬

if config['campaign_id'] == '1458':
    if config['budget_para'][0] == 0.5:
        eCPC = 60920.22773088766
    elif config['budget_para'][0] == 0.25:
        eCPC = 38767.41764692851
    elif config['budget_para'][0] == 0.125:
        eCPC = 33229.21512593873
    else:
        eCPC = 22152.81008395915
    directory = 'result_adjust_reward'
else:
    if config['budget_para'][0] == 0.5:
        eCPC = 77901.22125145316
    elif config['budget_para'][0] == 0.25:
        eCPC = 47939.21307781733
    elif config['budget_para'][0] == 0.125:
        eCPC = 35954.409808363
    else:
        eCPC = 23969.60653890866
    directory = '3386-result_adjust_reward'

action_file = directory + '/test_episode_actions_' + str(config['budget_para'][0]) + '.csv'
actions_df = pd.read_csv(action_file, header=None).drop([0])

max_result_index = max_train_index(directory, config['budget_para'][0])
print(max_result_index)
actions = actions_df.iloc[max_result_index, 1:].tolist()

test_data = pd.read_csv('../../data/' + config['campaign_id'] + '/test_data.csv', header=None).drop([0])
test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
    = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
    int)
test_data.iloc[:, config['data_pctr_index']] \
    = test_data.iloc[:, config['data_pctr_index']].astype(
    float)
test_data = test_data.values

budget = config['test_budget'] * config['budget_para'][0]

test_env(budget, test_data, eCPC, actions)