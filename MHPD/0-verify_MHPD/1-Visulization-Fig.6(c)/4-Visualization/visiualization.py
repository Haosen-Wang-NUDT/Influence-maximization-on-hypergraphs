import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import os

line_colors = ['#FF7F0E', '#2CA02C', '#1F77B4', '#D62728']
alpha_list = [0.6, 0.6, 1, 1]
# colors_plot = ['#D62728','#1F77B4','#9467BD','#2CA02C','#FF7F0E','#8C564B','#BCBD22','#7F7F7F','#E377C2']
# alpha_list = [0.8, 1, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]


def get_monte_karlo_result(dataset_name, beta):
    path = '../3-Monte_Karlo_verify/beta = %s/%s.txt'%(beta, dataset_name[:-4])
    data = np.loadtxt(path)
    return data

def get_MHPD_result(beta):
    path = '../2-MHPD_prediction/Scale_prediction_MHPD/Scale_prediction_%s.txt'%beta
    data = np.loadtxt(path)
    return data

def get_max_curve(data):
    max_curve = [max(data[:, x]) for x in range(len(data[0, :]))]
    return np.array(max_curve)

def get_min_curve(data):
    min_curve = [min(data[:, x]) for x in range(len(data[0, :]))]
    return np.array(min_curve)

def get_mean_curve(data):
    result = data.mean(axis = 0)
    return result

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 13

base_size = 0.9
fig = plt.figure(figsize=(20*base_size, 10*base_size))
axes = fig.subplots(nrows=2, ncols=4)

beta_list = [0.025,0.05,0.075,0.1]
# beta_list = [0.025,0.05]
t_list = [70]*8

titles = ['ERH_1',
'ERH_2',
'ERH_3',
'ERH_4',
'SFH_1',
'SFH_2',
'SFH_3',
'SFH_4',
]

Datasets = os.listdir(r'C:/Users/86442/Desktop/MHPD/0-Experiments/0-verify_MHPD/0-Syn_hypergraphs/HGs/')
for i, filename in tqdm(enumerate(Datasets), desc='Datasets'): 
    path = 'C:/Users/86442/Desktop/MHPD/0-Experiments/0-verify_MHPD/0-Syn_hypergraphs/HGs/' + filename
    t = t_list[i]
    matrix = np.loadtxt(path)
    m, n = matrix.shape
    
    Monte_Karlo_mean = [get_monte_karlo_result(filename, x)[:,:t].mean(axis = 0)/m for x in beta_list]
    Monte_Karlo_max = [get_monte_karlo_result(filename, x)[:,:t].max(axis = 0)/m for x in beta_list]
    Monte_Karlo_min = [get_monte_karlo_result(filename, x)[:,:t].min(axis = 0)/m for x in beta_list]
    Monte_Karlo_upper  = [np.quantile(get_monte_karlo_result(filename, x)[:,:t],0.9,axis = 0)/m for x in beta_list]
    Monte_Karlo_lower  = [np.quantile(get_monte_karlo_result(filename, x)[:,:t],0.1,axis = 0)/m for x in beta_list]
    # Monte_Karlo_std = [get_monte_karlo_result(filename, x)[:,:t].std(axis = 0) for x in beta_list]
    
    MHPD_list = [get_MHPD_result(x)[i][:t]/m for x in beta_list]

    x_index_for_t = range(1, t+1)

    fig.axes[i].set_xlabel('T/hop')
    fig.axes[i].set_ylabel('Influence Spread')
    x = 0.96  # x坐标位置（0-1之间，0为左边缘，1为右边缘）
    y = 0.05  # y坐标位置（0-1之间，0为下边缘，1为上边缘）
    fig.axes[i].text(x, y, titles[i], transform=fig.axes[i].transAxes, ha='right', va='bottom')
    # fig.axes[i].set_title(titles[i])

    for j in range(len(beta_list)):
        fig.axes[i].plot(x_index_for_t, Monte_Karlo_mean[j], color = line_colors[j], label = 'Monte Karlo beta = '+ str(beta_list[j]), alpha=alpha_list[j])
        # plt.fill_between(range(iter + 1), senate_res_edge_min, senate_res_edge_max, color='lightgreen', alpha=0.2)
        fig.axes[i].fill_between(x_index_for_t, Monte_Karlo_lower[j], Monte_Karlo_upper[j], color = line_colors[j], alpha=0.2)
        fig.axes[i].scatter(x_index_for_t, MHPD_list[j], color = line_colors[j], label = 'MHPD beta = '+ str(beta_list[j]), marker = 'x', alpha=alpha_list[j])
        
    # break
    


lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, ncol=4, bbox_to_anchor=(0.51, 0.967),loc = 'upper center',\
            prop = {'size':13.3}, frameon=False, columnspacing=7) 
plt.savefig('MHPD vs MTKL.svg', dpi = 400, bbox_inches = 'tight')
# # plt.savefig('K_scale_curve.jpg', dpi = 800, bbox_inches = 'tight')
# plt.show()