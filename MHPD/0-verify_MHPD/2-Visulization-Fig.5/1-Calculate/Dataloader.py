# 用于加载数据集

import pandas as pd
import numpy as np

class dataloader:
    node_dict = {}
    # 1、设置初始化方法，实例化类时需要传入目标数据的path
    def __init__(self,path):
        self.path = path
    # 2、读取数据集中的数据，将结果存储在类中
    def dataload(self):
        # 2.1 读取数据，获得节点列表字典
        df = pd.read_csv(self.path, index_col=False, header=None) # 读取指定路径的CSV文件，不指定列名和索引列
        arr = df.values    # 将DataFrame转换为二维数组
        node_list = [] # 创建一个空列表，用于存储节点
        for each in arr:
            node_list.extend(list(map(int, each[0].split(" ")))) # 将每行的字符串转换为整数列表，并添加到node_list列表中
        node_arr = np.unique(np.array(node_list)) # 使用numpy库的unique方法去除重复的节点，并转换为数组
        # 遍历node_arr数组的每个元素的索引和值
        for i in range(0, len(node_arr)):
            self.node_dict[node_arr[i]] = i # 将节点和对应的索引添加到node_dict字典中
        self.node_num = len(list(node_arr)) # 数据中涉及的节点的数量
        self.hp_edge_num = len(arr)
        # 2.2 读取数据，将数据转换成超边矩阵
        matrix = np.random.randint(0, 1, size=(self.node_num, self.hp_edge_num))  # 创建一个N行M列的随机矩阵，表示某节点是否在某超边中
        index = 0
        for each in arr:
            # 将字符串转换为整数列表
            edge_list = list(map(int, each[0].split(" ")))
            for edge in edge_list:
                # 将矩阵中对应位置的元素设为1
                matrix[self.node_dict[edge]][index] = 1
            index = index + 1
        self.hyper_matrix = pd.DataFrame(matrix)
