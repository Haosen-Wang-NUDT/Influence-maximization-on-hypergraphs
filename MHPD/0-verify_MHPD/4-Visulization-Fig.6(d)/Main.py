import numpy as np
import pandas as pd
import os
import MHPD
from tqdm import tqdm
import time
import Hyperspreading

if __name__ == '__main__':
    target_time_cost = pd.DataFrame(0, index = ['MHPD', 'Monte Karlo'], columns = range(8))
    
    # 0、数据准备
    path = 'random_seeds.txt'
    seeds_list = np.loadtxt(path, dtype=int)
    k = 50 # 总共预测多少阶
    beta = 0.025 # 传播的概率
    repeat = 20
    model = 'CP'

    # 1、先计算MHPD的时间开销
    datasets_name = os.listdir('../0-Syn_hypergraphs/HGs/')
    for i, dataset in tqdm(enumerate(datasets_name), desc='MHPD_Datasets'):
        path = '../0-Syn_hypergraphs/HGs/' + dataset
        matrix = np.loadtxt(path)
        i_cost_time_list = []
        for r in tqdm(range(repeat), desc = 'repeat'):
            start_time = time.time()
            _, _ = MHPD.MHPD(matrix, seeds_list[i, :], k, beta, model)
            end_time = time.time()
            cost_time = end_time - start_time
            i_cost_time_list.append(cost_time)
        i_avg_cost_time = sum(i_cost_time_list) / repeat
        target_time_cost.iloc[0, i] = i_avg_cost_time

    
    # 2、再计算MonteKarlo的时间开销
    R = 20000 # 仿真实验的次数
    for i, dataset in tqdm(enumerate(datasets_name), desc='MT_Datasets'):
        path = '../0-Syn_hypergraphs/HGs/' + dataset
        matrix = np.loadtxt(path)
        start_time = time.time()
        for r in tqdm(range(R), 'R'):
            _, _ = Hyperspreading.Hyperspreading().hyperSI_List(matrix, seeds_list[i, :], k, beta)
        end_time = time.time()
        cost_time = end_time - start_time
        target_time_cost.iloc[1, i] = cost_time
        
target_time_cost.to_csv('cost_time.csv')