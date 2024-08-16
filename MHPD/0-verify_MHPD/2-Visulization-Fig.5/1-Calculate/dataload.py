import pandas as pd
import numpy as np

class dataload:
    def get_scale(dataset, beta, T, Algorithms):
        seeds_size = range(1,31)
        result = pd.DataFrame(index = seeds_size, columns = Algorithms)
        for col, algo in enumerate(result.columns):
            for row, size in enumerate(result.index):
                inf_scale = []
                data_path = '../3-simulation_experiment/beta = %s, T = %s/%s/%s_%s.txt'%(beta, T, dataset, algo, size)
                with open(data_path, 'r') as file:
                    for line in file:
                        line_strip = line.strip()
                        inf_scale.append([int(x) for x in line_strip.split()])
                result.iloc[row, col] = np.array(inf_scale)
        return result
    
# data = dataload.get_scale('Algebra', 0.01)
                        
                
                
                