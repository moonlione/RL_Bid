import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from src.DDPG_BN.config import config

def test_env(directory, budget, budget_para, test_data, eCPC, actions):

    e_clks = [0 for i in range(24)]  # episode各个时段所获得的点击数，以下类推
    e_cost = [0 for i in range(24)]

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
    records_df.to_csv(directory + '/bids_' + str(budget_para) + '.csv', index=None)

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

def to_bids(budget_para, campaign_id, reward_directory):
    if campaign_id == '1458':
        if budget_para == 0.5:
            eCPC = 60920.22773088766
        elif budget_para == 0.25:
            eCPC = 38767.41764692851
        elif budget_para == 0.125:
            eCPC = 33229.21512593873
        else:
            eCPC = 22152.81008395915
        directory = reward_directory
    else:
        if budget_para == 0.5:
            eCPC = 77901.22125145316
        elif budget_para == 0.25:
            eCPC = 47939.21307781733
        elif budget_para == 0.125:
            eCPC = 35954.409808363
        else:
            eCPC = 23969.60653890866
        directory = '3386-' + reward_directory

    action_file = directory + '/test_episode_actions_' + str(budget_para) + '.csv'
    actions_df = pd.read_csv(action_file, header=None).drop([0])

    max_result_index = max_train_index(directory, budget_para)
    print(max_result_index)
    actions = actions_df.iloc[max_result_index, 1:].tolist()

    test_data = pd.read_csv('../../data/' + campaign_id + '/test_data.csv', header=None).drop([0])
    test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    test_data.iloc[:, config['data_pctr_index']] \
        = test_data.iloc[:, config['data_pctr_index']].astype(
        float)
    pd_test_data = test_data
    test_data = test_data.values

    budget = config['test_budget'] * budget_para

    test_env(directory, budget, budget_para, test_data, eCPC, actions)

    return directory,pd_test_data

def list_metrics(budget_para, directory):
    bid_records = pd.read_csv(directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 1:] = bid_records.iloc[:, 1:].astype(int)

    hour_clk_records = []
    hour_cost_records = []
    hour_cpc_records = []
    hour_imp_records = []
    for hour_clip in range(24):
        hour_records = bid_records[bid_records.iloc[:, 3].isin([hour_clip])]
        hour_records = hour_records.values

        win_records = hour_records[hour_records[:, 0] >= hour_records[:, 1]]

        hour_clks = np.sum(win_records[:, 2])
        hour_costs = np.sum(win_records[:, 1])
        hour_cpc = hour_costs / hour_clks if hour_clks > 0 else 0
        hour_imps = len(win_records)

        hour_clk_records.append(hour_clks)
        hour_cost_records.append(hour_costs)
        hour_cpc_records.append(hour_cpc)
        hour_imp_records.append(hour_imps)

    print(hour_clk_records)
    print(hour_cost_records)
    print(hour_cpc_records)
    print(hour_imp_records)

    records = [hour_clk_records, hour_cost_records, hour_cpc_records, hour_imp_records]

    for k in range(4):
        current_str = ''
        for m in range(len(hour_clk_records)):
            current_str = current_str + str(records[k][m]) + '\t'

        print(current_str)

def action_distribution(budget_para, directory):
    bid_records = pd.read_csv(directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 1:] = bid_records.iloc[:, 1:].astype(int)

    # 平均价格
    avg_actions = np.average(bid_records.iloc[:, 0])
    avg_market_prices = np.average(bid_records.iloc[:, 1])
    print(avg_actions, avg_market_prices)

    # 0-300价格区间的每个价格的个数
    action_dicts = {}
    market_price_dicts = {}
    for i in range(301):
        action_dicts.setdefault(i, 0)
        market_price_dicts.setdefault(i, 0)

    action_records = bid_records.iloc[:, 0].values
    market_price_records = bid_records.iloc[:, 1].values
    for i in range(len(action_records)):
        action_dicts[math.floor(action_records[i])] += 1 # math.ceil(x) 向上取整， math.floor(x)向上取整
        market_price_dicts[market_price_records[i]] += 1
    print(action_dicts)
    print(market_price_dicts)

    x_axis = action_dicts.keys()
    action_y_axis = list(action_dicts.values())
    market_price_y_axis = list(market_price_dicts.values())

    plt.plot(x_axis, action_y_axis, 'r')
    plt.plot(x_axis, market_price_y_axis, 'b')
    plt.legend()
    plt.show()

    action_i_sum = [0 for k in range(301)]
    market_price_i_sum = [0 for m in range(301)]

    # 0-300各个价格的累加
    for i in range(301):
        action_i_sum[i] = np.sum(list(action_y_axis)[:i])
        market_price_i_sum[i] = np.sum(list(market_price_y_axis)[:i])

    plt.plot(x_axis, action_i_sum, 'r')
    plt.plot(x_axis, market_price_i_sum, 'b')
    plt.legend()
    plt.show()

    price_nums = {'action_nums':action_y_axis, 'market_price_nums': market_price_y_axis, 'action_cumulative_nums': action_i_sum, 'market_price_cumulative_nums': market_price_i_sum}
    price_nums_df = pd.DataFrame(data=price_nums)
    price_nums_df.to_csv('price_nums.csv')

    f_i = open('price_nums.csv')
    for line in f_i:
        print(line.replace(',', '\t').strip())

def clk_frequency(test_data, budget_para, directory):
    real_labels = []
    for i in range(24):
        real_labels.append(
            np.sum(test_data[test_data.iloc[:, config['data_hour_index']].isin([i])].iloc[:, config['data_clk_index']]))
    print(real_labels)

    bid_records = pd.read_csv(directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 0] = np.ceil(bid_records.iloc[:, 0].values) # 向下取整
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(int)
    bid_records.iloc[:, 1:] = bid_records.iloc[:, 1:].astype(int)

    record_clk_indexs = [300]
    hour_appear_arrays = bid_records[bid_records.iloc[:, 0].isin(record_clk_indexs)].iloc[:, 3]
    is_in_indexs = bid_records[bid_records.iloc[:, 0].isin(record_clk_indexs)].iloc[:, 3].unique() # 有哪些时段出现了record_clk_index
    index_dicts = {}
    for index in is_in_indexs:
        index_dicts.setdefault(index, 0)

    for hour_appear in hour_appear_arrays.values:
        index_dicts[hour_appear] += 1

    print('点击最多的时段排序')
    sort_clk_index = np.argsort(-np.array(real_labels))  # 降序排列
    print(sort_clk_index) # 点击次数最多的时段排序

    print('出价价格300出现次数最多时段排序')
    sort_300_index = np.argsort(-np.array(list(index_dicts.values())))
    dict_keys = []
    for i in range(len(sort_300_index)):
        dict_keys.append(list(index_dicts.keys())[sort_300_index[i]])
    print(dict_keys) # 出价价格300出现次数最多时段排序
    print(index_dicts)


# 由启发式算法得到最优eCPC 1458-60920.22773088766,38767.41764692851,33229.21512593873, 22152.81008395915‬
# 3386-77901.22125145316‬,47939.21307781733,35954.409808363,23969.60653890866‬

budget_paras = [0.5, 0.25, 0.125, 0.0625]
campaign_id = '1458'
reward_type = 1 # 1-result_adjust_reward, 2-result_profit, 3-result
if reward_type == 1:
    reward_directory = 'result_adjust_reward'
elif reward_type == 2:
    reward_directory = 'result_profit'
else:
    reward_directory = 'result'

file_directory = ''
print('\n##########To Bids.csv files##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    directory, pd_test_data = to_bids(budget_para, campaign_id, reward_directory)
    file_directory = directory
    clk_frequency(pd_test_data, budget_para, directory)

# 各个时段的平均市场价格
for i in range(24):
    print(np.sum(pd_test_data[pd_test_data.iloc[:, 3].isin([i])].iloc[:, 2])/len(pd_test_data[pd_test_data.iloc[:, 3].isin([i])]))

print('\n##########List Metrics##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    list_metrics(budget_para, file_directory)

print('\n##########Action Distribution##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    action_distribution(budget_para, file_directory)

