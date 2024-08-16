import pandas as pd
import Hyperspreading
from tqdm import tqdm
import numpy as np
import os

class simulation_experiments:
    def conduct(df_hyper_matrix, seeds_list, R):
        # seeds_list是一个dataframe
        scale_result = seeds_list.copy()
        M, N = seeds_list.shape
        for col in range(N):
            for row in tqdm(range(M), desc="method %s: Loading..."%(seeds_list.columns[col])):
                inf_spread_matrix = [] # 存放每一次模拟的结果
                # seeds = simulation_experiments.string_to_list(seeds_list.iloc[row,col])
                seeds = eval(seeds_list.iloc[row,col])
                for r in range(R):
                    scale, _ = Hyperspreading.Hyperspreading().hyperSI(df_hyper_matrix, seeds)
                    inf_spread_matrix.append(scale)
                scale_result.iloc[row,col] = np.array(inf_spread_matrix).mean()
        return scale_result
    
    def conduct_list(df_hyper_matrix, seeds_list, R, t, beta):
        # seeds_list是一个dataframe
        scale_result = []
        M, N = seeds_list.shape
        for col in range(N):
            col_scale = []
            for row in tqdm(range(M), desc="method %s: Loading..."%(seeds_list.columns[col])):
                inf_spread_matrix = [] # 存放每一次模拟的结果
                seeds = eval(seeds_list.iloc[row,col])
                for r in range(R):
                    scale, _ = Hyperspreading.Hyperspreading().hyperSI_List(df_hyper_matrix, seeds, t, beta)
                    inf_spread_matrix.append(scale)
                col_scale.append(list(np.array(inf_spread_matrix).mean(axis = 0)))
            scale_result.append(col_scale)
        result = pd.DataFrame(scale_result).T
        result.index = seeds_list.index
        result.columns = seeds_list.columns  
        return result
    
    def conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta):
        if not os.path.exists('./beta = %s/'%beta + dataset[:-4]):
            # 如果不存在，则新建一个名为A的文件夹
            os.makedirs('./beta = %s/'%beta + dataset[:-4])
        else:
            pass
        
        work_path = './beta = %s/%s/'%(beta, dataset[:-4])
        # 开始仿真并记录
        M, N = seeds_list.shape
        for col in range(N):
            for row in tqdm(range(M), desc="method %s: Loading..."%(seeds_list.columns[col])):
                # 1、新建一个txt文件，存储仿真的数据
                file = open(work_path+'%s_%s.txt'%(seeds_list.columns[col], seeds_list.index[row]), 'w')
                inf_spread_matrix = [] # 一个K的结果，装2000个向量       
                seeds = eval(seeds_list.iloc[row,col])
                for r in range(R):
                    scale, _ = Hyperspreading.Hyperspreading().hyperSI_List(df_hyper_matrix, seeds, t, beta)
                    inf_spread_matrix.append(scale)
                # 2、写入数据
                for i in inf_spread_matrix:
                    # print(i)
                    file.write(' '.join([str(x) for x in i]))
                    file.write('\n')
                file.close()
                

