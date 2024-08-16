import numpy as np
import pandas as pd
from Adjacency_matrix import Adjacency_matrix_CP
import MHPD
import Hyperspreading
from tqdm import tqdm

R = 20000
T = 40
beta = 0.025

seeds = [797, 399, 756, 27, 796]
data = pd.DataFrame(np.loadtxt('../0-hypergraph/SFH_1000_4000_7_0.7.txt'))

# 1、计算Monte Karlo
Monte_Karlo = pd.DataFrame(None, index = range(R), columns = range(T+1))
for r in tqdm(range(R), desc = 'Monte_Karlo'):
    T_I_list, T_num_i_list = Hyperspreading.Hyperspreading().hyperSI_T_infected_nodes_list(data, seeds, T, beta)
    for t in range(T+1):
        Monte_Karlo.iloc[r, t] = np.array(T_I_list[t])

n, m = data.shape
target_MT = pd.DataFrame(0, index = range(n), columns = range(T+1))
for t in range(T+1):
    all_information = Monte_Karlo.iloc[:, t]
    all_infected_nodes = []
    for row in all_information:
        for r in row:
            all_infected_nodes.append(r)
    count_list = np.array([all_infected_nodes.count(node) for node in range(n)])
    count_list = count_list/R
    # target_MT.iloc[:, t] = count_list
    target_MT[target_MT.columns[t]] = count_list
    
target_MT.to_csv('MT_.csv', index = False)  

# 2、计算MHPD
adjacency_matrix = Adjacency_matrix_CP(data.values)
infect_matrix = adjacency_matrix * beta
node_prob_list = MHPD.MHPD(data, seeds, T, beta, infect_matrix)

# 3、记录两者的结果
target_data = np.zeros((1000, 10))
# 分别选择的代数
select_T = [0, 10, 20, 25, 35]

# 1、记录MHPD的实验结果
for i in range(5):
    target_data[:, i] = node_prob_list[select_T[i]]
for i in range(5):
    target_data[:, i+5] = target_MT.iloc[:, select_T[i]]
    
target_data = pd.DataFrame(target_data, columns = ['MHPD_0', 'MHPD_10', 'MHPD_20', 'MHPD_25', 'MHPD_35',\
                                                   'MT_0', 'MT_10', 'MT_20', 'MT_25', 'MT_35'])
target_data = target_data.rename_axis('ID')
target_data.to_csv('../node.csv')

