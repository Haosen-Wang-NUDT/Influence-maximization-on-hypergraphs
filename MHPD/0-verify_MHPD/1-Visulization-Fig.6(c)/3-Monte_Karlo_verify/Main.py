from Simulation_Experiments import  simulation_experiments
import os
import numpy as np

if __name__ == '__main__':
    # 1、读取在8个超图数据集上选择的测试种子
    path = '../1-Random_seeds/random_seeds.txt'
    seeds_list = np.loadtxt(path, dtype=int)
    
    t = 70 # 总共预测多少阶
    beta = 0.05 # 传播的概率
    R = 20000 # 仿真实验的次数

    n = 7
    datasets_name = os.listdir('../0-Syn_hypergraphs/HGs/')
    for dataset in datasets_name[n:n+1]:
        path = '../0-Syn_hypergraphs/HGs/' + dataset
        # print('-------------------------Simulation Experiment %s-------------------------'%dataset)
        # 2、调用dataloader类的dataload方法，获取必要信息
        matrix = np.loadtxt(path)
        
        seeds = seeds_list[n, :]

        # 5、传播仿真模拟测试验证效果
        simulation_experiments.conduct_all_information(dataset, matrix, seeds, R, t, beta)
        # result.to_csv('./beta = %s/'%beta + dataset[:-4] + '_%s.csv'%R, sheet_name="Spread_Scale")
        print('-------------------------Finished-------------------------\n')  