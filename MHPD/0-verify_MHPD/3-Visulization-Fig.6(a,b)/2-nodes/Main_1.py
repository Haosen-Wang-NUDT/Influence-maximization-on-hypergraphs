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

target_MT = pd.read_csv('MT_.csv')

# 2、计算MHPD
adjacency_matrix = Adjacency_matrix_CP(data.values)
infect_matrix = adjacency_matrix * beta
node_prob_list = MHPD.MHPD(data, seeds, T, beta, infect_matrix)

# 3、记录两者的结果
target_data = np.zeros((1000, 10))
# 分别选择的代数
select_T = [0, 15, 20, 25, 30]

# 1、记录MHPD的实验结果
for i in range(5):
    target_data[:, i] = node_prob_list[select_T[i]]
for i in range(5):
    target_data[:, i+5] = target_MT.iloc[:, select_T[i]]
    
target_data = pd.DataFrame(target_data, columns = ['MHPD_0', 'MHPD_15', 'MHPD_20', 'MHPD_25', 'MHPD_30',\
                                                   'MT_0', 'MT_15', 'MT_20', 'MT_25', 'MT_30'])
target_data = target_data.rename_axis('ID')
target_data.to_csv('../node.csv')