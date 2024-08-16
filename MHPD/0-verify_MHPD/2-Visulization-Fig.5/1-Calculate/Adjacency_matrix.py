import numpy as np

def K_hop_neighbors(df_hyper_matrix, seeds, iters):
    RP_matrix = Adjacency_matrix_RP(df_hyper_matrix)
    involved_nodes = list(seeds)
    for t in range(iters):
        t_involved_nodes = involved_nodes.copy()
        for node in t_involved_nodes:
            node_neighbors = np.where(RP_matrix[node, :] > 0)[0]
            # 上述列表不包含node自身，且没有重复节点
            # 若不再involed列表中，则全部添加进去
            involved_nodes.extend([x for x in node_neighbors if x not in involved_nodes])
    Neighbors = [x for x in involved_nodes if x not in list(seeds)]
    return Neighbors

def Adjacency_matrix_RP(df_hyper_matrix):
    # 1、根据节点超图矩阵获得RP模式下的节点之间的邻接矩阵
    # 其中每个元素代表两个节点之间的共有超边数
    m, n = df_hyper_matrix.shape
    Matrix = np.zeros((m, m))
    for i in range(m):
        i_neighbors = []
        i_edges = np.where(df_hyper_matrix[i, :] == 1)[0]
        for edge in i_edges:
            nodes_of_edge = np.where(df_hyper_matrix[:, edge] == 1)[0]
            i_neighbors.extend([x for x in nodes_of_edge if x != i])
        for j in range(m):
            if i == j:
                Matrix[i, j] = np.nan
            else:
                Matrix[i, j] = i_neighbors.count(j)
    return Matrix

def Adjacency_matrix_CP(df_hyper_matrix):
    # 1、根据节点超图矩阵获得CP模式下的节点之间的邻接矩阵
    # 其中每个元素代表共有超边数除以该节点自己的超度
    m, n = df_hyper_matrix.shape
    RP_matrix = Adjacency_matrix_RP(df_hyper_matrix)
    HEdges = [len(np.where(df_hyper_matrix[node, :] == 1)[0]) for node in range(m)]
    Matrix = RP_matrix
    for i in range(m):
        if HEdges[i] > 0:
            Matrix[i, :] /= HEdges[i]
    return Matrix