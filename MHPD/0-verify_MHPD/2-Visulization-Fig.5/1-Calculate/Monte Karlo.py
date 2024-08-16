import pandas as pd
import numpy as np
import os
import Hyperspreading
from Dataloader import dataloader
from tqdm import tqdm

def conduct_all_information(dataname, df_hyper_matrix, seed_index, R, T, beta):
    if not os.path.exists('./%s/'%(dataname)):
        # 如果不存在，则新建一个名为A的文件夹
        os.makedirs('./%s/'%(dataname))
    else:
        pass
    
    work_path = './%s/'%(dataname)
    # 开始仿真并记录
    # 1、新建一个txt文件，存储仿真的数据
    file = open(work_path+'%s.txt'%(seed_index), 'w')
    inf_spread_matrix = [] # 一个K的结果，装2000个向量       
    seeds = [seed_index]
    for r in range(R):
        scale_list, _ = Hyperspreading.Hyperspreading().hyperSI_T_Scale_list(df_hyper_matrix, seeds, T, beta)
        inf_spread_matrix.append(scale_list)
    # 2、写入数据
    # print(inf_spread_matrix)
    for i in inf_spread_matrix:
        # print(i)
        file.write(' '.join([str(x) for x in i]))
        file.write('\n')
    file.close()


if __name__ == '__main__':
    R = 10000
    T = 30
    beta = 0.01
    # 1、获取路径下面所有数据集的名称
    datasets_name = os.listdir('../0-hypergraph_datasets/')
    n = 0
    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索
    
    for dataset in datasets_name[n:n+1]:
        print('-------------------------Searching %s-------------------------'%dataset)   
        path = '../0-hypergraph_datasets/' + dataset
        # 2.1 调用dataloader类的dataload方法，获取必要信息
        df_hyper_matrix = pd.DataFrame(np.loadtxt(path))
        # break
        N, M = df_hyper_matrix.shape
        for i in tqdm(range(N), desc = '%s'%dataset[:-4], position=0,leave=True):
            conduct_all_information(dataset[:-4], df_hyper_matrix, i, R, T, beta)
        
        