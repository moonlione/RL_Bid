config = {
    'e_greedy': 0.9,
    'learning_rate_a': 1e-4,
    'learning_rate_c': 1e-3,
    'reward_decay': 1,
    'tau': 0.005,
    'feature_num': 6, # 153,3
    'data_pctr_index': 4, # 0
    'data_hour_index': 3, # 17:train-fm,3
    'data_clk_index': 1, # 15:train-fm,1
    'data_marketprice_index': 2, # 16:train-fm,2
    'data_feature_index': 1, # 15:train-fm,1
    'state_feature_num': 1, #,1
    'train_date': str(20130606), # sample 328481 328 22067108
    'test_date': str(20130607), # sample 307176 307 19441889
    'train_budget': 30096630, # 22067108
    'train_auc_num': 448164, # 155444, 127594, 173710
    'test_budget': 30228554, # 14560732
    'test_auc_num': 478109, # 68244
    'budget_para': [1/2, 1/4, 1/8, 1/16],
    'train_episodes': 30000,
    'neuron_nums': 50,
    'GPU_fraction': 1,
    'relace_target_iter': 100,
    'observation_size': 5000,
    'memory_size': 10000,
    'batch_size': 32, # GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128...时往往要比设置为整10、整100的倍数时表现更优
}