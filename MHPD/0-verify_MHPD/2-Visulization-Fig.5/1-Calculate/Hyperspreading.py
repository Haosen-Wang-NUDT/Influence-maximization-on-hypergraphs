import numpy as np
import random

class Hyperspreading:
    def getHpe(inode, matrix):
        """
        获取与给定节点相连的节点集合
        参数:
        - inode: int, 节点编号
        - matrix: array-like, 节点邻接矩阵
        返回:
        - array-like, 与给定节点相连的节点集合
        """
        return np.where(matrix[inode, :] == 1)[0]
    def chooseHpe(hpe_set):
        """
        从给定节点集合中选择一个节点
        参数:
        - hpe_set: set, 节点集合
        返回:
        - int or list, 选择的节点
        """
        if len(hpe_set) > 0:
            return random.sample(list(hpe_set), 1)[0]
        else:
            return []
    def getNodesofHpe(hpe, matrix):
        """
        获取与给定节点相连的节点集合
        参数:
        - hpe: int, 节点编号
        - matrix: array-like, 节点邻接矩阵
        返回:
        - array-like, 与给定节点相连的节点集合
        """
        return np.where(matrix[:, hpe] == 1)[0]
    def getNodesofHpeSet(hpe_set, matrix):
        """
        获取与给定节点集合中的节点相连的节点集合
        参数:
        - hpe_set: set, 节点集合
        - matrix: array-like, 节点邻接矩阵
        返回:
        - array-like, 与给定节点集合中的节点相连的节点集合
        """
        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(Hyperspreading.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)
    def findAdjNode_CP(inode, df_hyper_matrix):
        """
        根据给定节点和邻接矩阵找到相邻的节点集合
        参数:
        - inode: int, 节点编号
        - df_hyper_matrix: DataFrame, 节点邻接矩阵
        返回:
        - array-like, 相邻的节点集合
        """
        edges_set = Hyperspreading.getHpe(inode, df_hyper_matrix.values)
        edge = Hyperspreading.chooseHpe(edges_set)
        adj_nodes = np.array(Hyperspreading.getNodesofHpe(edge, df_hyper_matrix.values))
        return adj_nodes
    def formatInfectedList(I_list, infected_list, infected_T):
        """
        格式化感染节点集合
        参数:
        - I_list: list, 已感染节点集合
        - infected_list: array-like, 新感染节点集合
        - infected_T: list, 当前时刻感染节点集合
        返回:
        - generator, 格式化后的感染节点集合
        """
        return (x for x in infected_list if x not in I_list and x not in infected_T)
    def getTrueStateNode(self, adj_nodes, I_list, R_list):
        """
        获取未感染和未恢复节点集合
        参数:
        - adj_nodes: array-like, 相邻的节点集合
        - I_list: list, 已感染节点集合
        - R_list: list, 已恢复节点集合
        返回:
        - array-like, 未感染和未恢复节点集合
        """
        adj_list = list(adj_nodes)
        for i in range(0, len(adj_nodes)):
            if adj_nodes[i] in I_list or adj_nodes[i] in R_list:
                adj_list.remove(adj_nodes[i])
        return np.array(adj_list)
    def spreadAdj(adj_nodes, I_list, infected_T, beta):
        """
        根据给定节点集合进行传播
        参数:
        - adj_nodes: array-like, 相邻的节点集合
        - I_list: list, 已感染节点集合
        - infected_T: list, 当前时刻感染节点集合
        - beta: float, 传播概率
        返回:
        - array-like, 新感染节点集合
        """
        random_list = np.random.random(size=len(adj_nodes))
        infected_list = adj_nodes[np.where(random_list < beta)[0]]
        infected_list_unique = Hyperspreading.formatInfectedList(I_list, infected_list, infected_T)
        return infected_list_unique
    def hyperSI(self, df_hyper_matrix, seeds):
        """
        超传播模型
        参数:
        - df_hyper_matrix: DataFrame, 节点邻接矩阵
        - seeds: list, 初始感染节点集合
        返回:
        - int, 最后一个时刻感染节点数
        - list, 所有感染节点集合
        """
        I_list = list(seeds)
        beta = 0.01
        iters = 25
        I_total_list = [1]
        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            I_total_list.append(len(I_list))
        # plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
        # plt.show()
        return I_total_list[-1:][0], I_list
    
    def hyperSI_List(self, df_hyper_matrix, seeds, t, b):
        """
        超传播模型
        参数:
        - df_hyper_matrix: DataFrame, 节点邻接矩阵
        - seeds: list, 初始感染节点集合
        返回:
        - int, 最后一个时刻感染节点数
        - list, 所有感染节点集合
        """
        I_list = list(seeds)
        beta = b
        iters = t
        I_total_list = [len(I_list)]
        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            I_total_list.append(len(I_list))
        # plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
        # plt.show()
        return I_total_list[-1:][0], I_list
    
    def hyperSI_T_Scale_list(self, df_hyper_matrix, seeds, t, b):
        """
        超传播模型
        参数:
        - df_hyper_matrix: DataFrame, 节点邻接矩阵
        - seeds: list, 初始感染节点集合
        返回:
        - int, 最后一个时刻感染节点数
        - list, 所有感染节点集合
        """
        I_list = list(seeds)
        beta = b
        iters = t
        I_total_list = []
        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            I_total_list.append(len(I_list))
        # plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
        # plt.show()
        return I_total_list, I_list