import pandas as pd
import numpy as np
import os
import Hyperspreading
from tqdm import tqdm

R = 5
T = 50
beta = 0.01

seeds = [797, 399, 756, 27, 796]
data = pd.DataFrame(np.loadtxt('../0-hypergraph/SFH_1000_4000_7_0.7.txt'))
Monte_Karlo = pd.DataFrame(None, index = range(R), columns = range(T))
for r in tqdm(range(R), desc = 'R'):
    T_I_list, T_num_i_list = Hyperspreading.Hyperspreading().hyperSI_T_infected_nodes_list(data, seeds, T, beta)
    for t in range(T):
        Monte_Karlo.iloc[r, t] = np.array(T_I_list[t])

n, m = data.shape
target = pd.DataFrame(0, index = range(n), columns = range(T))
for t in range(T):
    all_information = Monte_Karlo.iloc[:, t]
    all_infected_nodes = []
    for row in all_information:
        for r in row:
            all_infected_nodes.append(r)
    count_list = np.array([all_infected_nodes.count(node) for node in range(n)])
    count_list = count_list/R
    target.iloc[:, t] = count_list
        
        