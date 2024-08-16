'''
该函数用于生成随机超图
'''
import numpy as np
np.random.seed(0)

def ERH(N, M, k):
    '''
    Parameters
    ----------
    N  : int —— 超图中节点的数量
    M  : int —— 超边的数量
    k  : int —— 超边的大小，即一条超边中包含节点的数量
    Returns —— 节点超边矩阵
    -------
    过程解释：
    随机生成M条不重复的超边构成超图
    '''
    edges = []
    num_edge = 0
    while num_edge < M:
        potential_edge = list(np.random.choice(N, size = k, replace = False))
        if potential_edge not in edges:
            edges.append(potential_edge)
            num_edge += 1
        else:
            pass
    matrix = np.zeros((N,M))
    for col in range(M):
        for node in edges[col]:
            matrix[node, col] = 1
    return edges, matrix

def SFH(N, M, k, miu):
    '''
    Parameters
    ----------
    N  : int —— 超图中节点的数量
    M  : int —— 超边的数量
    k  : int —— 超边的大小，即一条超边中包含节点的数量
    miu: float ~ (0,1) —— 控制度分布的参数
    Returns —— 节点超边矩阵
    -------
    过程解释：
    类似ERH，但选择超边的时候每个节点被选择的概率正比于x**(-miu)
    miu越小，度分布越集中
    '''
    edges = []
    num_edge = 0
    p_list = np.array([x**(-miu) for x in range(1, N+1)])
    p_select_list = p_list / p_list.sum()

    while num_edge < M:
        potential_edge = list(np.random.choice(N, size = k, replace = False, p = p_select_list))
        if potential_edge not in edges:
            edges.append(potential_edge)
            num_edge += 1
        else:
            pass
    matrix = np.zeros((N,M))
    for col in range(M):
        for node in edges[col]:
            matrix[node, col] = 1
    return edges, matrix

# 1、生成ERH
# N = 1000
# M = 7000
# k = 4
# _, matrix = ERH(N, M, k)
# np.savetxt('./HGs/ERH_%s_%s_%s.txt'%(N,M,k), matrix)

# 2、生成SFH
N = 1000
M = 5000
k = 6
miu = 0.6
_, matrix = SFH(N, M, k, miu)
np.savetxt('./HGs/SFH_%s_%s_%s_%s.txt'%(N,M,k,miu), matrix)
    
    
    
    
    
    
    
    