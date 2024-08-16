import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from dataload import dataload
from tqdm import tqdm

class visualization:
    def k_scale_curve():
        # 1、外观设置
        # 1.1 颜色
        # color_name = 'Paired'
        # colors_plot = list(matplotlib.colormaps.get_cmap(color_name).colors)
        # element = colors_plot.pop(1)  # 移除第6个元素，并将其保存到变量element中
        # colors_plot.insert(0, element)  # 将保存的元素插入到列表的开头
        # element = colors_plot.pop(5)  # 移除第6个元素，并将其保存到变量element中
        # colors_plot.insert(0, element)  # 将保存的元素插入到列表的开头
        
        colors_plot = ['#D62728','#1F77B4','#9467BD','#2CA02C','#FF7F0E','#8C564B','#BCBD22','#7F7F7F','#E377C2']
        alpha_list = [0.8, 1, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        # element = colors_plot.pop(1)  # 移除第6个元素，并将其保存到变量element中
        # colors_plot.insert(0, element)  # 将保存的元素插入到列表的开头
        # element = colors_plot.pop(5)  # 移除第6个元素，并将其保存到变量element中
        # colors_plot.insert(0, element)  # 将保存的元素插入到列表的开头
        # 1.2 形状
        marker_symbols = ['d-', '^-', 's-', 'v-', '*-', 'H-', '+-', 'x-', '--']
        # 1.3 字体
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 11
        
        # 2、数据读取
        Datasets = os.listdir('../0-hypergraph_datasets/')
        
        # Algorithms = ['MHPD-Greedy', 'MHPD-Heuristic', 'HADP', 'HSDP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','General-greedy']
            
        # labels = [ 'MHPD-Greedy', 'MHPD-Heuristic','HADP', 'HSDP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','Greedy']
        
        Algorithms = ['MHPD-Greedy','MHPD-HP', 'MHPD-Heuristic', 'HADP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','General-greedy']
            
        labels = ['MHPD-Greedy','MHPD-HP', 'MHPD-Heuristic','HADP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','Greedy']
        
        base_size = 0.9
        fig = plt.figure(figsize=(20*base_size, 10*base_size))
        axes = fig.subplots(nrows=2, ncols=4)
        
        beta = 0.01
        t = 25
        for i, filename in tqdm(enumerate(Datasets), desc='Datasets'): 
            data = dataload.get_scale(filename[:-4], beta, Algorithms)
            x_index_for_k = data.index
        
            fig.axes[i].set_xlabel('K')
            fig.axes[i].set_ylabel('Influence Spread')
            
            for j, name in enumerate(Algorithms):
                y_index_for_scale = [scale_list.mean(axis = 0)[t] for scale_list in data[name]]
                fig.axes[i].plot(x_index_for_k, y_index_for_scale , 
                                 marker_symbols[j],
                                 color = colors_plot[j], 
                                 label=labels[j], 
                                 zorder=len(labels)-j, 
                                 alpha = alpha_list[j])
        
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, ncol=9, bbox_to_anchor=(0.51, 0.94),loc = 'upper center',\
                   prop = {'size':13.3}, frameon=False) 
        # plt.savefig('K_scale_curve.svg', dpi = 400, bbox_inches = 'tight')
        
    def t_scale_curve():
        colors_plot = ['#D62728','#1F77B4','#9467BD','#2CA02C','#FF7F0E','#8C564B','#BCBD22','#7F7F7F','#E377C2']
        alpha_list = [0.8, 1, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        # 1.2 形状
        marker_symbols = ['d-', '^-', 's-', 'v-', '*-', 'H-', '+-', 'x-', '--']
        # 1.3 字体
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 11
        
        # 2、数据读取
        Datasets = os.listdir('../0-hypergraph_datasets/')
        
        Algorithms = ['MHPD-Greedy', 'MHPD-Heuristic', 'HADP', 'HSDP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','General-greedy']
            
        labels = [ 'MHPD-Greedy', 'MHPD-Heuristic','HADP', 'HSDP', 'H-CI(I=2)', 'H-Degree', 'Degree', 'H-RIS','Greedy']
        
        base_size = 0.9
        fig = plt.figure(figsize=(20*base_size, 10*base_size))
        axes = fig.subplots(nrows=2, ncols=4)
        
        k = 5
        beta = 0.01
        t_max = 25
     
        for i, filename in tqdm(enumerate(Datasets)): 
            data = dataload.get_scale(filename[:-4], beta, Algorithms)
            x_index_for_t = range(t_max)
        
            fig.axes[i].set_xlabel('T')
            fig.axes[i].set_ylabel('Influence Spread')
            # fig.axes[i].set_title (filename[:-4])
            # fig.axes[i].text(0.5,0.5,'(%s)'%(i+1),fontsize=18,ha='center',zorder=100)
            
            for j, name in enumerate(Algorithms):
                y_index_for_scale = data[name].iloc[k-1].mean(axis = 0)[:t_max]
                
                fig.axes[i].plot(x_index_for_t, y_index_for_scale , 
                                 marker_symbols[j],
                                 color = colors_plot[j], 
                                 label=labels[j], 
                                 zorder=len(labels)-j, 
                                 alpha = alpha_list[j])
                
                # plt.plot(x_index_for_k, y_index_for_scale)
        
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, ncol=9, bbox_to_anchor=(0.51, 0.94),loc = 'upper center',\
                   prop = {'size':13.3}, frameon=False) 
        plt.savefig('T_scale_curve.svg', dpi = 400, bbox_inches = 'tight')
        # plt.savefig('T_scale_curve.jpg', dpi = 800, bbox_inches = 'tight')
        # plt.show()
        
    def NB_graph():
        # 用于单独对比8个数据集上Greedy和HEDV-Greedy方法的区别，用两种不同颜色对比
        # 8个数据集画8个图，每个图上使用k=30的时候t=0-30的曲线  
        # 1.1 字体
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 11
        
        # 2、数据读取
        Datasets = os.listdir('../0-hypergraph_datasets/')
            
        labels = ['HEDV-Greedy','Greedy']
        
        base_size = 0.9
        fig = plt.figure(figsize=(20*base_size, 10*base_size))
        axes = fig.subplots(nrows=2, ncols=4)
        
        k = 30
        beta = 0.01
        t_max = 25
        mtkl = 2000
        alpha_mtkl = 0.45
        
        for i, filename in tqdm(enumerate(Datasets)): 
            data = dataload.get_scale(filename[:-4], beta)
            HEDV_all = np.array([x[:t_max] for x in data['HEDV-greedy'].iloc[-1]])[:mtkl]
            HEDV_avg = HEDV_all.mean(axis = 0)
            Greedy_all = np.array([x[:t_max] for x in data['General-greedy'].iloc[-1]])[:mtkl]
            Greedy_avg = Greedy_all.mean(axis = 0)
            HADP_all = np.array([x[:t_max] for x in data['HADP'].iloc[-1]])[:mtkl]
            HADP_avg = HADP_all.mean(axis = 0)
            
            x_index_for_t = np.array(range(t_max))
        
            fig.axes[i].set_xlabel('T')
            fig.axes[i].set_ylabel('Influence Spread')
            
            # 1、先画两个的所有曲线
            HADP_all_A , HADP_all_B = visualization.concat_array(x_index_for_t, HADP_all)
            Greedy_all_A , Greedy_all_B = visualization.concat_array(x_index_for_t, Greedy_all)
            HEDV_all_A , HEDV_all_B = visualization.concat_array(x_index_for_t, HEDV_all)
            

            fig.axes[i].plot(Greedy_all_A, Greedy_all_B, '#91ACE0', alpha = alpha_mtkl, linewidth = 0.2)
            fig.axes[i].plot(HADP_all_A, HADP_all_B, '#ACD78E', alpha = alpha_mtkl, linewidth = 0.2)
            fig.axes[i].plot(HEDV_all_A, HEDV_all_B, '#EC97A5', alpha = alpha_mtkl, linewidth = 0.2)
            
            # 1、再绘制两个的平均曲线
            fig.axes[i].plot(x_index_for_t, Greedy_avg, 'blue', linewidth = 2, label = 'Greedy')
            fig.axes[i].plot(x_index_for_t, HADP_avg, 'green', linewidth = 2, label = 'HADP')
            fig.axes[i].plot(x_index_for_t, HEDV_avg, 'red', linewidth = 3, label = 'HEDV-Greedy')      
 
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines[::-1], labels[::-1], ncol=3, bbox_to_anchor=(0.51, 0.94) ,loc = 'upper center',\
                   prop = {'size':13.3}, frameon=False) 
        plt.savefig('NB_graph.svg', dpi = 400, bbox_inches = 'tight')
        # plt.show()
        
    def array_2d_to_mean(array):
        return np.array(array).mean(axis = 0)
    
    def concat_array(x_index, y_index_list):
        t_max = len(x_index)
        x_index = x_index.astype("float")
        y_index_list = y_index_list.astype("float")
        # 1、处理横坐标
        # 每相隔x_index添加一个np.nan
        A = np.tile(np.append(x_index,[np.nan]),len(y_index_list))
        # 2、处理纵坐标
        # 把y_index_list拉开成一个一维向量
        raveled_y = y_index_list.flatten()
        # 每相隔t_max添加一个np.nan
        # 计算要插入的元素个数
        num_nans = len(y_index_list)
        # 生成要插入的元素，都为np.nan
        nans = np.full(num_nans, np.nan)
        # 使用np.insert()函数在原数组的每隔2个元素的位置插入np.nan
        B = np.insert(raveled_y, np.arange(t_max, len(raveled_y)+1, t_max), nans)
        return A,B
 
if __name__ == '__main__':
    visualization.k_scale_curve()
    # visualization.t_scale_curve()
    # visualization.NB_graph()
