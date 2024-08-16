import numpy as np
import pandas as pd
from scipy.integrate import trapz
import os
from dataload import dataload
from tqdm import tqdm

Datasets = os.listdir('../0-hypergraph_datasets/')
Datasets_name = [x[:-4] for x in Datasets]

Algorithms = ['MHPD-Greedy', 'General-greedy']
    
beta_list = [0.01 ,0.02 ,0.015 ,0.005]
T_list = [25, 20, 30, 35]
AUV_all = pd.DataFrame(columns = Algorithms, index = Datasets_name)
writer = pd.ExcelWriter('AUV_all_beta.xlsx')

for q in range(4): # 四种参数设置
    beta = beta_list[q]
    t    = T_list[q]
    for i, filename in tqdm(enumerate(Datasets_name)): 
        data = dataload.get_scale(filename, beta, Algorithms)
        x_index_for_k = data.index
        AUV_list = []
        for j, name in enumerate(Algorithms):
            y_index_for_scale = [scale_list.mean(axis = 0)[t] for scale_list in data[name]]
            auv = trapz(y_index_for_scale, x_index_for_k)
            AUV_list.append(auv)
        AUV_list = np.array(AUV_list)
        AUV_list_normalized = AUV_list/sum(AUV_list)
        # print(AUV_all.loc[:, name])
        AUV_all.loc[filename, :] = np.around(AUV_list_normalized, 4)

    AUV_all.to_excel(writer, sheet_name='beta = %s, t = %s'%(beta,t))

writer.close()
        