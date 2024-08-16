from Dataloader import dataloader
from Algorithms import algorithms

import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 1、获取路径下面所有数据集的名称
    datasets_name = os.listdir('../0-hypergraph_datasets/')
    n = 0
    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索
    for dataset in datasets_name[n:n+1]:
        path = '../0-hypergraph_datasets/' + dataset
        print('-------------------------Searching %s-------------------------'%dataset)   
        # 2.1 调用dataloader类的dataload方法，获取必要信息
        dl = dataloader(path)
        dl.dataload()
        df_hyper_matrix = dl.hyper_matrix # 从数据中获取的节点超边矩阵
        
        # 3、设置K的值
        K = 30 # 目标种子集合的大小
        beta = 0.01
        
        # 4、使用每一种已有的算法获取种子节点
        # 4.1 创建容器
        seeds_list = []
        cost_time_list = []
        # 4.2 使用每一种算法求解
        '''
        顺序依次为
        HEDV-greedy, HADP, HSDP, H-RIS, H-CI(I=1), H-CI(I=2), H-Degree, Degree, General-greedy
        '''
        # methods = ['MHPD-Heuristic', 'MHPD-Greedy', 'HADP', 'HSDP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','Greedy']
        methods = ['MHPD-Heuristic', 'MHPD-Greedy']
        
        seeds_list_MHPD_H , cost_time_MHPD_H  = algorithms.MHPD_herusitc(df_hyper_matrix, K, beta, k = 1)
        seeds_list_MHPD_G , cost_time_MHPD_G  = algorithms.MHPD_greedy(df_hyper_matrix, K, beta, k = 1)
        # seeds_list_HADP   , cost_time_HADP    = algorithms.HADP(df_hyper_matrix, K)
        # seeds_list_HSDP   , cost_time_HSDP    = algorithms.HSDP(df_hyper_matrix, K)
        # seeds_list_CI2    , cost_time_CI2     = algorithms.CI(df_hyper_matrix, K, 2)
        # seeds_list_HDegree, cost_time_HDegree = algorithms.HDegree(df_hyper_matrix, K)
        # seeds_list_Degree , cost_time_Degree  = algorithms.degreemax(df_hyper_matrix, K)
        # seeds_list_RIS    , cost_time_RIS     = algorithms.RIS(df_hyper_matrix, K, 0.01, 200)
        # seeds_list_Greedy , cost_time_Greedy  = algorithms.generalGreedy(df_hyper_matrix, K, mtkl = 50)
        # 4.3 全部添加到列表中，统一输出
        seeds_list.append(seeds_list_MHPD_H)
        seeds_list.append(seeds_list_MHPD_G)
        # seeds_list.append(seeds_list_HADP)
        # seeds_list.append(seeds_list_HSDP)
        # seeds_list.append(seeds_list_CI2)
        # seeds_list.append(seeds_list_HDegree)
        # seeds_list.append(seeds_list_Degree)
        # seeds_list.append(seeds_list_RIS)
        # seeds_list.append(seeds_list_Greedy)
        
        cost_time_list.append(cost_time_MHPD_H)
        cost_time_list.append(cost_time_MHPD_G)
        # cost_time_list.append(cost_time_HADP)
        # cost_time_list.append(cost_time_HSDP)
        # cost_time_list.append(cost_time_CI2)
        # cost_time_list.append(cost_time_HDegree)
        # cost_time_list.append(cost_time_Degree)
        # cost_time_list.append(cost_time_RIS)
        # cost_time_list.append(cost_time_Greedy)
        # 5、保存搜索到的种子结果
        seeds_result = pd.DataFrame(seeds_list).T
        seeds_result.loc[len(seeds_result.index)] = np.array(cost_time_list)
        seeds_result.index = list(np.arange(1,K+1)) + ['Time_Cost']
        seeds_result.columns = methods
        seeds_result.to_excel('./seeds_result/' + dataset[:-4] + '.xlsx', sheet_name="Seeds_List")
        print('-------------------------Finished-------------------------\n')  



    
