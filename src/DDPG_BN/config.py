config = {
    'e_greedy': 0.9,
    'learning_rate_a': 1e-4,
    'learning_rate_c': 1e-3,
    'reward_decay': 1,
    'tau': 0.001,
    'feature_num': 5, # 153,3
    'data_pctr_index': 4, # 0
    'data_hour_index': 3, # 17:train-fm,3
    'data_clk_index': 1, # 15:train-fm,1
    'data_marketprice_index': 2, # 16:train-fm,2
    'data_feature_index': 1, # 15:train-fm,1
    'state_feature_num': 1, #,1
    'campaign_id': '3386',
    'train_date': str(20130606), # sample 328481 328 22067108
    'test_date': str(20130607), # sample 307176 307 19441889
    # 'train_budget': 30096630, # 30096630, 30608307
    # 'train_auc_num': 448164, # 1448164, 448164, 435900
    # 'test_budget': 30228554, # 130228554, 30228554, 30231716
    # 'test_auc_num': 444191, # 478109, 444191
    'train_budget': 30608307, # 30096630, 30608307
    'train_auc_num': 435900, # 1448164, 448164, 435900
    'test_budget': 30231716, # 130228554, 30228554, 30231716
    'test_auc_num': 444191, # 478109, 444191
    'budget_para': [1/4],
    'train_episodes': 50000,
    'neuron_nums_c_1': 50,
    'neuron_nums_c_2': 40,
    'neuron_nums_a_1': 30,
    'neuron_nums_a_2': 20,
    'GPU_fraction': 1,
    'learn_iter': 24,
    'observation_size': 240,
    'memory_size': 500000,
    'batch_size': 32, # GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128...时往往要比设置为整10、整100的倍数时表现更优
}