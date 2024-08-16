import numpy as np
import os

np.random.seed(0)

def get_random_seeds(hypermatrix, seeds_num):
    m, n = hypermatrix.shape
    nodes_list = list(range(m))
    random_nodes = np.random.choice(nodes_list, seeds_num, replace = False)
    return random_nodes

if __name__ == '__main__':
    datasets_name = os.listdir('../0-Syn_hypergraphs/HGs/')
    seeds_list = []
    seeds_num = 5 # 测试用的种子节点的数量，暂定用5个
    for dataset in datasets_name:
        path = '../0-Syn_hypergraphs/HGs/' + dataset
        # print('-------------------------Searching %s-------------------------'%dataset)   
        # 2.1 调用dataloader类的dataload方法，获取必要信息
        matrix = np.loadtxt(path)
        dataset_seeds = get_random_seeds(matrix, seeds_num)
        seeds_list.append(dataset_seeds)
    save_text = np.array(seeds_list)
    np.savetxt('random_seeds.txt', save_text, fmt="%d")
        

        