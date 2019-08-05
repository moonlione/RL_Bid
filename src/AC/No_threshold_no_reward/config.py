config = {
    'e_greedy': 1,
    'learning_rate': 0.1,
    'reward_decay': 0.2,
    'feature_num': 150,
    'train_episodes': 1000,
    'neuron_nums': 50,
    'GPU_fraction': 1,
    'relace_target_iter': 100,
    'memory_size': 100000,
    'batch_size': 32, # GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128...时往往要比设置为整10、整100的倍数时表现更优
}