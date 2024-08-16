import pandas as pd
import Hyperspreading
from tqdm import tqdm
import numpy as np
import os

class simulation_experiments:
    def conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta):
        if not os.path.exists('./beta = %s'%beta):
            # 如果不存在，则新建一个名为A的文件夹
            os.makedirs('./beta = %s'%beta)
        else:
            pass
        
        save_path = './beta = %s/%s.txt'%(beta, dataset[:-4])
        inf_spread_matrix = []
        for r in tqdm(range(R), desc='%s'%dataset):
            scale, _ = Hyperspreading.Hyperspreading().hyperSI_List(df_hyper_matrix, seeds_list, t, beta)
            inf_spread_matrix.append(scale)
        
        save_text = np.array(inf_spread_matrix)
        # print(save_text)
        np.savetxt(save_path, save_text, fmt="%d")
                

