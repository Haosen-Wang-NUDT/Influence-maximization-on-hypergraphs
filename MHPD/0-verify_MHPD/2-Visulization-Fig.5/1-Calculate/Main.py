import pandas as pd
import numpy as np
from Dataloader import dataloader
import os
from tqdm import tqdm
from Adjacency_matrix import Adjacency_matrix_CP
from MHPD import MHPD

def get_scale(dataset, beta, T, Algorithms):
    seeds_size = range(1,31)
    result = pd.DataFrame(index = seeds_size, columns = Algorithms)
    for col, algo in enumerate(result.columns):
        for row, size in enumerate(result.index):
            inf_scale = []
            data_path = '../3-simulation_experiment/beta = %s, T = %s/%s/%s_%s.txt'%(beta, T, dataset, algo, size)
            with open(data_path, 'r') as file:
                for line in file:
                    line_strip = line.strip()
                    inf_scale.append([int(x) for x in line_strip.split()])
            result.iloc[row, col] = np.array(inf_scale)
    return result

if __name__ == '__main__':
    # 1、获取路径下面所有数据集的名称
    datasets_name = os.listdir('../0-hypergraph_datasets/')
    n = 0
    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索
    
    for dataset in datasets_name[n:n+1]:
        print('-------------------------Searching %s-------------------------'%dataset)   
        path = '../0-hypergraph_datasets/' + dataset
        # 2.1 调用dataloader类的dataload方法，获取必要信息
        dl = dataloader(path)
        dl.dataload()
        df_hyper_matrix = dl.hyper_matrix # 从数据中获取的节点超边矩阵
        
        N, M = df_hyper_matrix.shape
        
        data = pd.DataFrame(columns=range(60), index=range(N))
        
        # 1、先读取所有Monte Karlo数据
        for i in tqdm(range(N), desc = 'Read_Monte_Karlo'):
            inf_scale = []
            data_path = './%s/%s.txt'%(dataset[:-4], i)
            with open(data_path, 'r') as file:
                for line in file:
                    line_strip = line.strip()
                    inf_scale.append([int(x) for x in line_strip.split()])
            data_i = np.array(inf_scale)
            data.iloc[i, 30:] = data_i.mean(axis=0)
            
        # 2、计算所有MHPD的值
        beta = 0.01
        inf_mx = Adjacency_matrix_CP(df_hyper_matrix.values) * beta
        for i in tqdm(range(N), desc = 'Calculate_MHPD'):
            data.iloc[i, :30] = np.sum(MHPD(df_hyper_matrix, [i], 30, beta, 'CP', inf_mx), axis=1)[1:]
            
        data.to_excel('%s.xlsx'%dataset[:-4])
            