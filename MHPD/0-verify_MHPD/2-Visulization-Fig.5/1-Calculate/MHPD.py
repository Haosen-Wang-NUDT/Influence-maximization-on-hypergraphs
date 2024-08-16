"""
让概率在网络中传播的方法
"""
import numpy as np
import Adjacency_matrix
# from Dataloader import dataloader

def MHPD(hyper_matrix, seeds, k, beta, model, infect_matrix):
    """
    输入：
    超图、种子节点集合、评估阶段数量k、节点之间传播概率beta、传播模型可选CP或RP
    """
    m, n = hyper_matrix.shape
    # spread_scale = []
    nodes_prob = [] # 存储不同k情况下节点被感染的概率
    # 根据传播模型和超图结构确定节点之间在一次传播过程中的感染概率
    
    nodes_data = np.zeros(m) # 存储节点当前被感染的概率
    for i in seeds:
        nodes_data[i] = 1 # 种子的data为1
    nodes_prob.append(nodes_data.copy()) # 保存初始数据
    
    for i in range(k): # 一共要更新k次
        # print(nodes_data-1)
        
        nodes_data = nodes_prob_update(infect_matrix, nodes_data)
        nodes_prob.append(nodes_data.copy())
        
    nodes_prob_k = nodes_prob[-1] # 返回最后一次的节点状态列表
        
    return nodes_prob
     
def cal_infect_matrix(model, hyper_matrix, beta):
    if model == 'RP':
        infect_matrix = 1-(1-beta)**Adjacency_matrix.Adjacency_matrix_RP(hyper_matrix)
    elif model == 'CP':
        infect_matrix = Adjacency_matrix.Adjacency_matrix_CP(hyper_matrix) * beta
    return infect_matrix

def nodes_prob_update(infect_matrix, nodes_data):
    raw_nodes_data = nodes_data.copy() # 备份，用这个来计算
    for i in range(len(nodes_data)): # 第i个节点，非种子节点才需要更新
        raw_prob = raw_nodes_data[i]
        neighbors = np.where(infect_matrix[i, :] > 0)[0]
        new_p_list_infect = raw_nodes_data[neighbors]
        new_p_list_spread = np.array([infect_matrix[x, i] for x in neighbors])
        new_p_list = list(new_p_list_infect * new_p_list_spread)
        nodes_data[i] = prob_meanwhile(raw_prob, new_p_list)
    return nodes_data

def prob_meanwhile(raw_prob, new_p_list):
    # 输入一串感染概率P_list输出整体感染概率
    new_p_list.append(raw_prob)
    result = 1 - np.prod(1-np.array(new_p_list))
    return result

# path = 'test.txt'
# dl = dataloader(path)
# dl.dataload()
# df_hyper_matrix = dl.hyper_matrix

# k = 3
# beta = 0.5
# model = 'CP'
# infect_matrix = cal_infect_matrix(model, df_hyper_matrix.values, beta)
# prob_sum = MHPD(df_hyper_matrix.values, [0], k, beta, model, infect_matrix = infect_matrix)
# print('\n', prob_sum)
