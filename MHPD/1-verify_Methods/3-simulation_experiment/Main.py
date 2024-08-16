from Dataloader import dataloader
from Simulation_Experiments import  simulation_experiments
import os
import pandas as pd

if __name__ == '__main__':
    # 1、获取路径下面所有数据集的名称
    datasets_name = os.listdir('../0-hypergraph_datasets/')
    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索
    n = 7
    for dataset in datasets_name[n:n+1]:
        path = '../0-hypergraph_datasets/' + dataset
        print('-------------------------Simulation Experiment %s-------------------------'%dataset)
        # 2、调用dataloader类的dataload方法，获取必要信息
        dl = dataloader(path)
        dl.dataload()
        df_hyper_matrix = dl.hyper_matrix # 从数据中获取的节点超边矩阵 
        
        # 3、设置R的值
        R = 500 # 仿真实验的次数
        t = 50
        beta = 0.01
        
        # 4、读取已搜索到的种子节点数据
        seeds_list = pd.read_excel('../2-search_seeds/seeds_result/' + dataset[:-4] + '.xlsx', \
                                    sheet_name="Seeds_List", \
                                    index_col=0, \
                                    skipfooter=1)

        # 5、传播仿真模拟测试验证效果
        simulation_experiments.conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta)
        # result.to_csv('./beta = %s/'%beta + dataset[:-4] + '_%s.csv'%R, sheet_name="Spread_Scale")
        print('-------------------------Finished-------------------------\n')  