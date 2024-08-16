import numpy as np
import os
import MHPD
from tqdm import tqdm
import time

if __name__ == '__main__':
    # 1、读取在8个超图数据集上选择的测试种子
    path = '../1-Random_seeds/random_seeds.txt'
    seeds_list = np.loadtxt(path, dtype=int)
    
    Scale_prediction = [] # 用于存放所有预测出来的传播规模
    k = 50 # 总共预测多少阶
    beta = 0.025 # 传播的概率
    model = 'CP' # 使用CP传播模型
    
    # 2、首先读取所有的超图数据集
    datasets_name = os.listdir('../0-Syn_hypergraphs/HGs/')
    for i, dataset in tqdm(enumerate(datasets_name), desc='Datasets'):
        path = '../0-Syn_hypergraphs/HGs/' + dataset
        matrix = np.loadtxt(path)
        
        start_time = time.time()
        _, dataset_MHPD = MHPD.MHPD(matrix, seeds_list[i, :], k, beta, model)
        end_time = time.time()
        print(dataset,end_time - start_time)
        Scale_prediction.append(dataset_MHPD)
        
    save_text = np.array(Scale_prediction)
    # np.savetxt('./Scale_prediction_MHPD/Scale_prediction_%s.txt'%beta, save_text, fmt="%f")