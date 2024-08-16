import pandas as pd
import numpy as np
import os

Datasets = os.listdir('../0-hypergraph_datasets/')
Datasets_name = [x[:-4] for x in Datasets]
Algorithms = ['MHPD-heristic','MHPD-greedy']
result = pd.DataFrame(index = Datasets_name, columns = Algorithms)

for filename in Datasets_name:
    data = pd.read_excel('../2-search_seeds/seeds_result/' + filename + '.xlsx', index_col=0)
    time_cost = data.loc['Time_Cost']
    # print(time_cost)
    # print(result.loc[filename])
    result.loc[filename] = time_cost.values
    
result.to_excel('time_cost_all.xlsx')