'''
本程序记载所有搜索算法，可调用后求解影响力最大化问题
'''
import numpy as np
import pandas as pd
import random
import copy
import Hyperspreading
import networkx as nx
from tqdm import tqdm
import time
import MHPD

class algorithms:
    # 张子柯文章提出方法：
    ################################################################################################
    # 方法一：度最大化：'Degree'方法
    def degreemax(df_hyper_matrix, K):
        """
        度最大化方法
        """
        begin_time = time.time()
        seed_list_degreemax = []
        degree = algorithms.getTotalAdj(df_hyper_matrix) # 计算总度数
        for i in tqdm(range(0, K), desc='Degree finished'): # 对于每个种子节点数量
            seeds = algorithms.getSeeds_sta(degree, i) # 获取种子节点
            seed_list_degreemax.append(seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_degreemax, cost_time
       
       
    # 辅助函数：
    # 1、计算所有节点的度数
    def getTotalAdj(df_hyper_matrix):
        """
        计算所有节点的度数列表：
        度数：与该节点相连接，即在同一个超边下的节点的数量
        """
        deg_list = [] # 初始化度数列表
        N,M = df_hyper_matrix.shape
        nodes_arr = np.arange(N) # 生成节点索引数组
        for node in nodes_arr: # 对于每个节点
            node_list = [] # 初始化节点列表
            edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0] # 找到与节点相连的边集合
            for edge in edge_set: # 对于每条边
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0])) # 找到与边相连的节点，并添加到节点列表中
            node_set = np.unique(np.array(node_list)) # 去除重复的节点
            deg_list.append(len(list(node_set)) - 1) # 计算节点的度数，并添加到度数列表中
        return np.array(deg_list) # 返回度数列表
    # 2、根据节点读书列表，选择前i个度数最大的节点作为种子集合
    def getSeeds_sta(degree, i): 
        """
        根据节点度数选择目标种子集合
        不做其他处理，而是直接选择度数最靠前的几个节点，可能导致影响力重复严重
        """
        # 构建节点和度数矩阵
        matrix = []
        matrix.append(np.arange(len(degree)))
        matrix.append(degree)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.index = ['node_index', 'node_degree']
        # 根据节点度数降序排序
        df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
        # 获取排序后的度数和节点列表
        degree_list = list(df_sort_matrix.loc['node_degree'])
        nodes_list = list(df_sort_matrix.loc['node_index'])
        # 选择前i个节点作为种子节点
        chosed_arr = list(df_sort_matrix.loc['node_index'][:i])
        # 如果度数相同的节点有多个，则随机选择一个
        index = np.where(np.array(degree_list) == degree_list[i])[0]
        nodes_set = list(np.array(nodes_list)[index])
        while 1:
            node = random.sample(nodes_set, 1)[0]
            # 如果该节点不在已选择的节点列表中，则加入选择列表中
            if node not in chosed_arr:
                chosed_arr.append(node)
                break
            else:
                # 如果该节点已经在已选择的节点列表中，则从节点集合中删除，并继续选择
                nodes_set.remove(node)
                continue
        return chosed_arr
    ###################################################################################################
    # 2、方法二：超度最大化：'H-Degree'方法
    def HDegree(df_hyper_matrix, K):
        """
        超度最大化方法
        """
        begin_time = time.time()
        seed_list_HDegree = []
        degree = df_hyper_matrix.sum(axis=1) # 计算节点的超度数
        for i in tqdm(range(0, K), desc='H-Degree finished'): # 对于每个种子节点数量
            seeds = algorithms.getSeeds_sta(degree, i) # 获取种子节点
            seed_list_HDegree.append(seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HDegree, cost_time
    ###################################################################################################
    # 3、方法三：HeuristicDegreeDiscount，基于度折扣的启发式方法
    def HADP(df_hyper_matrix, K):
        """
        HeuristicDegreeDiscount算法是一种基于度数的启发式算法，用于选择种子节点。以下是该算法的伪代码：
        1、初始化一个空的种子节点列表seeds
        2、初始化一个节点度数列表degree，调用函数getTotalAdj(df_hyper_matrix, N)得到每个节点的度数
        3、循环K次：
            a. 找到度数最大的节点，调用函数getMaxDegreeNode(degree, seeds)得到最大度数节点
            b. 将最大度数节点添加到种子节点列表seeds中
            c. 更新度数列表degree，调用函数updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
        输出：选择的种子节点列表seeds
        """
        begin_time = time.time()
        seed_list_HUR = []
        seeds = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)
        for j in tqdm(range(1, K+1), desc="HADP finished"):
            chosenNode = algorithms.getMaxDegreeNode(degree, seeds)
            seeds.append(chosenNode)
            seed_list_HUR.append(seeds.copy())
            algorithms.updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HUR, cost_time
    # 辅助函数：
    # 1、获取当前度数最大的未被选择节点
    def getMaxDegreeNode(degree, seeds):
        """
        获取度数最大的未选择节点
        """
        degree_copy = copy.deepcopy(degree)
        global chosedNode
        while 1:
            flag = 0
            degree_matrix = algorithms.getDegreeList(degree_copy)
            node_index = degree_matrix.loc['node_index']
            for node in node_index:
                if node not in seeds:
                    chosedNode = node
                    flag = 1
                    break
            if flag == 1:
                break
        return chosedNode
    # 2、获取按照度数由大到小重新排序后的节点序列
    def getDegreeList(degree):
        """
        获取按照度数由大到小重新排序后的节点序列
        """
        matrix = []
        matrix.append(np.arange(len(degree)))
        matrix.append(degree)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.index = ['node_index', 'node_degree']
        return df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
    # 3、使用HUR方法更新节点的度数，主要用于支持HUR节点选择方法
    def updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds):
        """
        使用HUR方法更新节点的度数，主要用于支持HUR节点选择方法
        """
        edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]  # 获取选择节点所连接的超边索引集合
        adj_set = []
        for edge in edge_set:
            adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))  # 获取与选择节点相连的节点集合
        adj_set_unique = np.unique(np.array(adj_set))  # 去除重复的节点
        for adj in adj_set_unique:  # 遍历与选择节点相连的节点
            adj_edge_set = np.where(df_hyper_matrix.loc[adj] == 1)[0]  # 获取与相连节点相连的超边索引集合
            adj_adj_set = []
            for each in adj_edge_set:
                adj_adj_set.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))  # 获取与相连节点相连的节点集合
            if adj in adj_adj_set:
                adj_adj_set.remove(adj)  # 去除相连节点自身
            sum = 0
            for adj_adj in adj_adj_set:
                if adj_adj in seeds:
                    sum = sum + 1  # 统计相连节点相连的节点中已选择的种子节点的个数
            degree[adj] = degree[adj] - sum  # 更新相连节点的度数
            
    ###################################################################################################
    # 4、方法四：HeuristicSingleDiscount：简单的度折扣方法
    def HSDP(df_hyper_matrix, K):
        """
        HuresticSingleDiscount algorithm
        """
        begin_time = time.time()
        seed_list_HSD = []
        seeds = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)
        for j in tqdm(range(1, K+1), desc="HSDP finished"):
            chosenNode = algorithms.getMaxDegreeNode(degree, seeds)
            seeds.append(chosenNode)
            seed_list_HSD.append(seeds.copy())
            algorithms.updateDeg_hsd(degree, chosenNode, df_hyper_matrix)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HSD, cost_time
    # 辅助函数：
    # 1、使用HSD方法更新节点的度数，主要用于支持HSD节点选择方法
    def updateDeg_hsd(degree, chosenNode, df_hyper_matrix):
        """
        使用HSD方法更新节点的度数，主要用于支持HSD点选择方法：
        """
        edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]  # 获取选择节点所连接的超边索引集合
        for edge in edge_set:
            node_set = np.where(df_hyper_matrix[edge] == 1)[0]  # 获取与选择节点相连的节点集合
            for node in node_set:
                degree[node] = degree[node] - 1  # 更新相连节点的度数
    ###################################################################################################
    # 5、方法五：'greedy'方法
    def generalGreedy(df_hyper_matrix, K, mtkl=1):
        """
        GeneralGreedy algorithm
        """
        begin_time = time.time()
        degree = df_hyper_matrix.sum(axis=1)
        seed_list_Greedy = []
        seeds = []
        for i in tqdm(range(0, K), desc="General-greedy finished"):
            scale_list_temp = []
            maxNode = 0
            maxScale = 0
            for inode in range(0, len(degree)):
                if inode not in seeds:
                    seeds.append(inode)
                    scale_avg = []
                    for i in range(mtkl):
                        scale_temp, _ = Hyperspreading.Hyperspreading().hyperSI(df_hyper_matrix, seeds)
                        scale_avg.append(scale_temp)
                    scale = np.array(scale_avg).mean()
                    seeds.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale
            seeds.append(maxNode)
            seed_list_Greedy.append(seeds.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_Greedy, cost_time
    ###################################################################################################
    # 6、方法六：'CI'方法
    def CI(df_hyper_matrix, K, l):
        begin_time = time.time()
        # df_hyper_matrix是超边的邻接矩阵，K是要选择的节点个数，l是计算CI值时的参数，表示使用几阶邻居
        seed_list_CI = []
        seeds = []  # 保存选中的节点
        N, M = df_hyper_matrix.shape  # 获取节点和超边的个数
        n = np.ones(N)  # 初始化一个长度为N的数组，表示节点是否被选中，初始都为1表示未选中
        CI_list = algorithms.computeCI(df_hyper_matrix, l)  # 调用computeCI函数计算所有节点的CI值
        CI_arr = np.array(CI_list)  # 将CI_list转换为numpy数组
        for j in range(0, K):
            # 循环K次，每次选择一个具有最大CI值的节点
            CI_chosed_val = CI_arr[np.where(n == 1)[0]]  # 获取未被选中的节点的CI值
            CI_chosed_index = np.where(n == 1)[0]  # 获取未被选中的节点的索引
            index = np.where(CI_chosed_val == np.max(CI_chosed_val))[0][0]  # 找到最大CI值对应的索引
            node = CI_chosed_index[index]  # 获取对应的节点
            n[node] = 0  # 将选中的节点标记为已选中
            seeds.append(node)  # 将选中的节点添加到seeds列表中
            seed_list_CI.append(seeds.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_CI, cost_time
    # 辅助函数：
    # 1、计算每个节点的CI值
    def computeCI(df_hyper_matrix, l):
        CI_list = []  # 用于存储每个节点的CI值
        degree = df_hyper_matrix.sum(axis=1)  # 每个节点的度数
        N,M = df_hyper_matrix.shape  # 节点和超边的个数
        for i in tqdm(range(0, N), desc = "CI (l=%d) finished"%l):  # 遍历每个节点
            edge_set = np.where(df_hyper_matrix.loc[i] == 1)[0]  # 找到与当前节点相连的超边索引
            if l == 1:  # 如果l=1，只考虑一阶邻居
                node_list = []
                for edge in edge_set:
                    node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
                if i in node_list:
                    node_list.remove(i)
                node_set = np.unique(np.array(node_list))  # 找到节点集合，去除重复节点
            elif l == 2:  # 如果l=2，考虑二阶邻居
                node_list = []
                for edge in edge_set:
                    node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
                if i in node_list:
                    node_list.remove(i)
                node_set1 = np.unique(np.array(node_list))  # 找到一阶邻居节点集合
                node_list2 = []
                edge_matrix = np.dot(df_hyper_matrix.T, df_hyper_matrix)
                edge_matrix[np.eye(M, dtype=np.bool_)] = 0
                df_edge_matrix = pd.DataFrame(edge_matrix)
                adj_edge_list = []
                for edge in edge_set:
                    adj_edge_list.extend(list(np.where(df_edge_matrix[edge] != 0)[0]))
                adj_edge_set = np.unique(np.array(adj_edge_list))  # 找到与一阶邻居有共同邻居的超边集合
                for each in adj_edge_set:
                    node_list2.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
                node_set2 = list(np.unique(np.array(node_list2)))  # 找到二阶邻居节点集合
                for node in node_set2:
                    if node in list(node_set1):  # 去除二阶邻居中已经包含在一阶邻居中的节点
                        node_set2.remove(node)
                node_set = np.array(node_set2)  # 最终得到节点集合
            ki = degree[i]  # 当前节点的度数
            sum = 0
            for u in node_set:  # 遍历节点集合中的每个节点
                sum = sum + (degree[u] - 1)
            CI_i = (ki - 1) * sum  # 计算CI值
            CI_list.append(CI_i)
        return CI_list  # 返回每个节点的CI值列表
    ###################################################################################################
    # 7、方法七：'RIS'方法
    def RIS(df_hyper_matrix, K, lamda, theta):
        begin_time = time.time()
        seed_list_RIS = []
        S = []  # 存储选定的种子节点
        U = []  # 存储每次迭代生成的子图的节点
        N, M = df_hyper_matrix.shape  # 获取节点和超边的个数
        # 迭代theta次
        for theta_iter in tqdm(range(0, theta), desc = "RIS finished"):
            df_matrix = copy.deepcopy(df_hyper_matrix)  # 深拷贝超图的邻接矩阵
            # 随机选择一个节点
            selected_node = random.sample(list(np.arange(len(df_hyper_matrix.index.values))), 1)[0]
            # 以1-λ的比例删除边，构成子超图
            all_edges = np.arange(len(df_hyper_matrix.columns.values))  # 所有边的索引
            prob = np.random.random(len(all_edges))  # 随机生成概率
            index = np.where(prob > lamda)[0]  # 概率大于lamda的边的索引
            for edge in index:
                df_matrix[edge] = 0  # 删除边
            # 将子超图映射到普通图
            adj_matrix = np.dot(df_matrix, df_matrix.T)  # 子超图的邻接矩阵
            adj_matrix[np.eye(N, dtype=np.bool_)] = 0  # 将对角线元素置为0
            df_adj_matrix = pd.DataFrame(adj_matrix)
            df_adj_matrix[df_adj_matrix > 0] = 1  # 大于0的元素置为1
            G = nx.from_numpy_array(df_adj_matrix.values)  # 将邻接矩阵转换为图
            shortest_path = nx.shortest_path(G, target=selected_node)  # 得到从随机选择的节点到其他节点的最短路径
            RR = []
            for each in shortest_path:
                RR.append(each)
            U.append(list(np.unique(np.array(RR))))  # 将每次迭代生成的节点加入U
        # 重复K次
        for k in range(0, K):
            U_list = []
            for each in U:
                U_list.extend(each)
            dict = {}  # 存储节点和出现次数的字典
            for each in U_list:
                if each in dict.keys():
                    dict[each] = dict[each] + 1
                else:
                    dict[each] = 1
            candidate_list = sorted(dict.items(), key=lambda item: item[1], reverse=True)  # 按节点出现次数降序排序
            chosed_node = candidate_list[0][0]  # 选择出现次数最多的节点
            S.append(chosed_node)  # 将选定的节点加入S
            seed_list_RIS.append(S.copy())
            for each in U:
                if chosed_node in each:
                    U.remove(each)  # 从U中移除包含选定节点的子图
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_RIS, cost_time
    ###################################################################################################
    # 自提方法
    # 1、基于KPD的启发式方法（其中k的数量可选）
    def MHPD_herusitc(df_hyper_matrix, K, beta, k, model = 'CP'):
        begin_time = time.time()
        m, n = df_hyper_matrix.shape
        infect_matrix = MHPD.cal_infect_matrix(model, df_hyper_matrix.values, beta)
        nodes_values = [MHPD.MHPD(df_hyper_matrix.values, [x], k, beta, model, infect_matrix = infect_matrix).sum() for x in tqdm(range(m), desc='MHPD-heuristic')]
        sorted_nodes = sorted(list(range(m)), key=lambda x: nodes_values[x], reverse=True)
        seeds = [sorted_nodes[:k] for k in range(1, K+1)]
        end_time = time.time()
        cost_time = end_time - begin_time
        return seeds, cost_time
    
    def MHPD_greedy(df_hyper_matrix, K, beta, k, model = 'CP'):
        """
        基于目标函数的贪婪策略构建初始解
        """
        begin_time = time.time()
        seed_list_MHPD = []
        num_nodes = df_hyper_matrix.shape[0]
        seeds_Greedy = []
        infect_matrix = MHPD.cal_infect_matrix(model, df_hyper_matrix.values, beta)
        for i in tqdm(range(K),desc='MHPD-greedy'): # 一共要添加k个节点
            maxNode = 0
            maxfitness = 0
            for inode in range(num_nodes):
                if inode not in seeds_Greedy:
                    seeds_Greedy.append(inode)
                    fitness = MHPD.MHPD(df_hyper_matrix.values, seeds_Greedy, k, beta, model, infect_matrix = infect_matrix).sum()
                    seeds_Greedy.remove(inode)
                    if fitness > maxfitness:
                        maxNode = inode
                        maxfitness = fitness
            seeds_Greedy.append(maxNode)
            seed_list_MHPD.append(seeds_Greedy.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_MHPD, cost_time

    
    

    
    
    
    
    
    
    
    
    
    
    
