import numpy as np
import pandas as pd
from Adjacency_matrix import Adjacency_matrix_CP
from tqdm import tqdm

beta = 0.01
data = np.loadtxt('../0-hypergraph/SFH_1000_4000_7_0.7.txt')
adjacency_matrix = Adjacency_matrix_CP(data)
infect_matrix = adjacency_matrix * beta
n, m = data.shape

edges = []
for i in tqdm(range(n), desc = 'Node'):
    for j in range(i, n):
        element = infect_matrix[i, j]
        if not pd.isnull(element) and element != 0: # 假如不为空，且概率不为0
            edge = [i, j, element]
            edges.append(edge)

target_csv = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
target_csv.to_csv('../edge.csv', index = False)
