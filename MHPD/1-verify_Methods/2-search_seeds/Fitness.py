'''
用于估计选定种子的好坏
'''
import numpy as np

class fitness:
    # 方法一：HEDV, Hypergragh EDV
    def HEDV(df_hyper_matrix, seed_set, beta=0.01):
        # 只考虑一跳范围内传播节点数量的期望
        # 1、首先确定初始种子集的一跳区域（除开种子外的）
        one_hop_nodes = fitness.get_one_hop_Nodes_of_Nodes(seed_set, df_hyper_matrix) # 获取所有一跳区域节点
        one_hop_nodes_set = [x for x in one_hop_nodes if x not in seed_set] # 除去种子节点部分
        
        # 2、遍历每一个邻居节点i，计算其被感染的概率，求和
        EDV = len(seed_set)
        for i,inode in enumerate(one_hop_nodes_set):
            adj_inodes_list = fitness.get_one_hop_Nodes_of_Node(inode, df_hyper_matrix) # 该邻居节点的邻居
            seeds_involved = [x for x in adj_inodes_list if x in seed_set] # 与该节点相连的种子节点
            pro_not_i = 1
            for j in seeds_involved:
                pro_not_i *= (1-beta*fitness.cal_select_probability(inode, j, df_hyper_matrix))
            pro_i = 1-pro_not_i
            EDV += pro_i
        return EDV
     
    # 辅助函数：
    def getHpe(inode, matrix):
        """
        获取给定节点所在的超边列表
        """
        return np.where(matrix[inode, :] == 1)[0]


    def getNodesofHpe(hpe, matrix):
        """
        获取给定超边中的所有节点列表
        """
        return np.where(matrix[:, hpe] == 1)[0]
    
    def getNodesofHpeSet(hpe_set, matrix):
        """
        获取超边集合中涉及到的所有节点
        """
        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(fitness.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)
    
    def get_one_hop_Nodes_of_Node(node, matrix):
        """
        获取节点集合的所有一跳区域节点
        """
        edge_set = fitness.getHpe(node, matrix) # 获得该节点所在的所有超边
        one_hop_nodes = fitness.getNodesofHpeSet(edge_set, matrix) # 获得超边集中所有节点
        one_hop_nodes_set = fitness.drop_duplicates(one_hop_nodes) # 去除重复
        return one_hop_nodes_set
    
    def get_one_hop_Nodes_of_Nodes(nodes, matrix):
        """
        获取节点集合的所有一跳区域节点
        """
        one_hop_nodes_set = [] # 初始化
        for inode in nodes:
            edge_set = fitness.getHpe(inode, matrix) # 获得该节点所在的所有超边
            one_hop_nodes = fitness.getNodesofHpeSet(edge_set, matrix) # 获得超边集中所有节点
            one_hop_nodes_set.extend(one_hop_nodes) # 将得到的所有节点添加到列表中
        one_hop_nodes_set = fitness.drop_duplicates(one_hop_nodes_set) # 去除重复
        return one_hop_nodes_set

    def drop_duplicates(nodes_list):
        """
        去除列表重复
        """
        nodes_list = list(set(nodes_list)) # 去重
        return nodes_list

    def cal_select_probability(adj_node, seed, df_hyper_matrix):
        j_edges = fitness.getHpe(seed, df_hyper_matrix)
        j_num_edges = len(j_edges)
        if j_num_edges == 0:
            return 0
        else:
            j_include_i_num_edges = 0
            for edge in j_edges:
                if adj_node in fitness.getNodesofHpe(edge, df_hyper_matrix):
                    j_include_i_num_edges += 1
            return j_include_i_num_edges/j_num_edges
#######################################################################################################
