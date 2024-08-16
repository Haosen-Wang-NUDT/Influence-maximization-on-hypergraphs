import numpy as np
import pandas as pd
import os

def get_topo_hypergraph(all_path):
    matrix = np.loadtxt(all_path)
    n, m = matrix.shape
    k = sum(matrix[:,0])
    avg_d = avg_degree(matrix)
    avg_hyperdegree = avg_HD(matrix)
    return np.array([n,m,k,avg_d,avg_hyperdegree])

def avg_degree(matrix):
    n, m = matrix.shape
    nodes_degree = []
    for inode in range(n):
        inode_edges = np.where(matrix[inode, :] == 1)[0]
        inode_neibors = []
        for jedge in inode_edges:
            nodes = np.where(matrix[:, jedge] == 1)[0]
            for x in nodes:
                if x not in inode_neibors and x != inode:
                    inode_neibors.append(x)
        nodes_degree.append(len(inode_neibors))
    avg_degree = sum(nodes_degree) / n
    return avg_degree

def avg_HD(matrix):
    n, m = matrix.shape
    nodes_HD = [len(np.where(matrix[x, :] == 1)[0]) for x in range(n)]             
    avg_HD = sum(nodes_HD) / n
    return avg_HD

if __name__ == '__main__':
    paths = os.listdir('./HGs')
    values = ['n','m','k', 'deg','dH']
    datasets = [i[:-4] for i in paths]
    result = pd.DataFrame(index = datasets, columns= values)
    
    for i, path in enumerate(paths):
        all_path = './HGs/' + path
        informations = get_topo_hypergraph(all_path)
        result.loc[datasets[i]] = informations
        
    result.to_excel('topo.xlsx')