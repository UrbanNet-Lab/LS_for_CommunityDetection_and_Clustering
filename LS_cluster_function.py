import numpy as np
import pandas as pd
import networkx as nx
import re
import os
import collections
import heapq
import sys
import community
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mc
import colorsys
# import random

# from pylab import *
from collections import Counter
from scipy.special import comb, perm
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'

font = {'family': 'Times New Roman',
        'style': 'italic',
        'weight': 'normal',
        'size': 22,
        }
font1 = {'family': 'Times New Roman',
        'style': 'italic',
        'weight': 'normal',
        'size': 18,
        }
    
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# 绘制划分结果
def plot_fig(x,y,labels,center_id,filepath='./',dataname='LS_default',save=False):
    cmap = plt.get_cmap('RdYlBu')
    color_number = len(set(labels))
    colors = [cmap(i) for i in np.linspace(0, 1, color_number)]
    text = [str(i) for i in range(len(x))]
    
    fig, ax = plt.subplots(figsize=(8,7))
    for i in range(len(x)):
        if int(labels[i]) != -1:
            edgecolors = adjust_lightness(colors[int(labels[i])], 0.8)
            if i in center_id:
#                 ax.text(x[i], y[i]-0.1, text[i], ha='center', fontsize=14,fontweight='bold',alpha=0.6)
                ax.scatter(x[i], y[i], c=colors[int(labels[i])], s = 1000, marker='*',linewidth=1,edgecolors=edgecolors, alpha = 1,zorder=2)
            else:
                ax.scatter(x[i], y[i], c=colors[int(labels[i])], s = 300, marker='o',linewidth=1,edgecolors=edgecolors, alpha = 0.3,zorder=1)
                
        else:
            ax.scatter(x[i], y[i], c='black', marker='o',s=100,edgecolor='black',alpha=0.3)
    
    ax.grid(True)
    fig.tight_layout()
    plt.xticks([])  # 去x坐标刻度
    plt.yticks([])  # 去y坐标刻度   
    if save == True:
        filename = filepath + str(dataname) + '_LScluster_result.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()

def plot_louvian_fig(x,y,labels,filepath='./',dataname='LS_default',save=False):
    cmap = plt.get_cmap('RdYlBu')
    color_number = len(set(labels))
    colors = [cmap(i) for i in np.linspace(0, 1, color_number)]
    text = [str(i) for i in range(len(x))]
    
    fig, ax = plt.subplots(figsize=(8,7))
    for i in range(len(x)):
        edgecolors = adjust_lightness(colors[int(labels[i])], 0.8)
        if int(labels[i]) != -1:
            ax.scatter(x[i], y[i], c=colors[int(labels[i])], s = 300, marker='o',linewidth=1,edgecolors=edgecolors, alpha = 0.6)
        else:
            ax.scatter(x[i], y[i], c='black', marker='o',s=100,edgecolor='black',alpha=0.3)
            
    ax.grid(True)
    fig.tight_layout()
    plt.xticks([])  # 去x坐标刻度
    plt.yticks([])  # 去y坐标刻度
    if save == True:
        filename = filepath + str(dataname) + '_Louvaincluster_result.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
    

# 计算任意两点之间的切比雪夫距离,并存储为矩阵
def caldistance(v,order):
#     man = np.linalg.norm(vector1-vector2,ord=1)  # Manhattan
#     euc = np.linalg.norm(vector1-vector2)  # Euclidian
#     che = np.linalg.norm(vector1-vector2,ord=np.inf)  # Chebyshev
#     cos = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))) # Cosine Similarity
    distance = np.zeros(shape=(len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            if i > j:
                distance[i][j] = distance[j][i]
            elif i < j:
                if order == 0:
                    distance[i][j] = np.linalg.norm(v[i]-v[j])
                else:
                    distance[i][j] = np.linalg.norm(v[i]-v[j],ord=np.inf)
    return distance

# 选择合适的阈值
def chose_dc(dis, t):
    temp = []
    for i in range(len(dis[0])):
        for j in range(i + 1, len(dis[0])):
            temp.append(dis[i][j])
    #  升序排列
    temp.sort()
    arr_min = np.min(temp)
    arr_max = np.max(temp)
    unit = (arr_max - arr_min) / 99
    dc_list = []
    for i in range(100):
        temp1 = arr_min + unit * i
        dc_list.append(temp1)# 2.4861762041092385e-08
    dc = dc_list[t]
    return dc

def chose_dc_gradual(dis):
    temp = []
    for i in range(len(dis[0])):
        for j in range(i + 1, len(dis[0])):
            temp.append(dis[i][j])
    #  升序排列
    temp.sort()
    arr_min = np.min(temp)
    arr_max = np.max(temp)
    unit = (arr_max - arr_min) / 99
    dc = []
    for i in range(100):
        temp1 = arr_min + unit * i
        dc.append(temp1)# 2.4861762041092385e-08
    return dc

#  计算网络连边
def cal_adge(dis, dc):
    start = []
    end = []
    for i in range(len(dis[0])):
        for j in range(i + 1, len(dis[0])):
            if dis[i][j] <= dc:
                start.append(i)
                end.append(j)
    return start,end

def plot_connect(arr1,arr2,t,filepath='./',dataname='LS_default',save=False):
    # plot 
    fig, ax_f = plt.subplots(figsize=(8, 7))
    ax_c = ax_f.twinx()
    
    colors = ['#3E7C17','#F4A442']
    labels = ['GCC','SGCC']
    left, bottom, width, height = 0.1,0.1,0.8,0.8
    
    f, = ax_f.plot(arr2, color=colors[0],linestyle='-', linewidth=1, marker='o',  markersize=10, label = labels[0],alpha = 0.8)
    c, = ax_c.plot(arr1, color=colors[1],linestyle='-', linewidth=1, marker='o',  markersize=10, label = labels[1],alpha = 0.8)
    for i in range(len(arr1)):
        if i == t:
            ax_f.plot(i,arr2[i], color='red',linestyle='-', linewidth=1, marker='o',  markersize=12, label = labels[0],alpha = 0.8)
            ax_c.plot(i,arr1[i], color='red',linestyle='-', linewidth=1, marker='o',  markersize=12, label = labels[1],alpha = 0.8)
            
    ax_c.set_ylim(-0.05,max(arr1)+0.05)
    ax_f.set_ylim(-0.05,max(arr2)+0.05)
    
    ax_f.legend(handles=[c,f],loc='upper left',fontsize=14)
    ax_f.tick_params(labelsize=18)
    ax_c.tick_params(labelsize=18)
    
    ax_f.set_xlabel(u'$\epsilon$ (% of diameter)',fontsize=18)
    ax_f.set_ylabel(u'GCC',fontsize=18) # cumulative distribution function
    ax_c.set_ylabel(u'SGCC',fontsize=18) # cumulative distribution function
    
    if save == True:
        filename = filepath + str(dataname) + '_jumppoint.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
    
def cal_jumppoint(input_x,t,dataname):
    norm_data = input_x
    distance = caldistance(norm_data,0)  # 制作任意两点之间的距离矩阵
    nodes = [i for i in range(len(input_x))]
    gnode = len(input_x)
    dc_list = []
    dc_list = chose_dc_gradual(distance)
    Gc_list = []
    subGc_list = []
    edge_proportion = []
    sub_edge_proportion = []
    for dc in dc_list:
        start, end = cal_adge(distance, dc)  # 统计每点的密度
        df = pd.DataFrame({ 'from':start, 'to':end})
        G = nx.from_pandas_edgelist(df, source = 'from', target = 'to')
        G.add_nodes_from(nodes)
        
        # 获得第2大连通子图
        count = 0
        connected = list(nx.connected_components(G))
        subgraphs = [G.subgraph(i) for i in connected]
        G_all = sorted(subgraphs,key=len, reverse = True)
        if len(G_all) > 1:
            for i in  G_all:
                if count == 1:
                    subGc = i
                    sub_connectivity = subGc.number_of_edges() / comb(gnode,2)
                    subGc_list.append(subGc.number_of_nodes()/ gnode)
                    sub_edge_proportion.append(sub_connectivity)
                count += 1
        else:
            subGc_list.append(0)
            sub_edge_proportion.append(0)
            
        # 获得最大连通子图
        Gc = max(subgraphs, key=len)
        connectivity = Gc.number_of_edges() / comb(gnode,2)
        Gc_list.append(Gc.number_of_nodes() / gnode)
        edge_proportion.append(connectivity)
    print('Determine jumppoint :',np.array(subGc_list)[:50])
    plot_connect(np.array(subGc_list),np.array(Gc_list),t,filepath='./',dataname=dataname,save=False)
    
def plot_predict_olivetti_img(imgs,targets,predict,filepath='./',dataname='LS_default',save=False):
    cmap = plt.get_cmap('rainbow')
    color_number = len(set(predict))+1
    colors = [cmap(i) for i in np.linspace(0, 1, color_number)]
    figure = plt.figure(figsize=(30, 30))
    plt.subplots_adjust(left=0.15,bottom=0.1,right=0.85,top=0.9,wspace=0.05,hspace=0.05)
    cols, rows = 10, 10
    for t in range(1, cols * rows + 1):
        i = t-1 
        img = imgs[i]
        label = targets[i]
        if predict[i] == -1:
            edge_color = 'black'
        else:
            edge_color = colors[predict[i]]
        ax = figure.add_subplot(rows, cols, t)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_color(edge_color)
        ax.spines["top"].set_linewidth(5)
        ax.spines["left"].set_color(edge_color)
        ax.spines["left"].set_linewidth(5)
        ax.spines["right"].set_color(edge_color)
        ax.spines["right"].set_linewidth(5)
        ax.spines["bottom"].set_color(edge_color)
        ax.spines["bottom"].set_linewidth(5)
        ax.imshow(img.squeeze(), cmap="gray")
        
    if save == True:
        filename = filepath + str(dataname) + '_olivetti_result.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
    

def evaluate_network(new_class,node_number):
    cluster_number = {}
    j = 0
    for i in set(new_class.values()):
        if i != -1:
            cluster_number[i] = j
            j += 1
        else:
            cluster_number[i] = -1
            
    labels = []
    partition = {}
    for i in range(node_number):
        labels.append(cluster_number[new_class[i]])
        partition[i] = cluster_number[new_class[i]]
    return labels,partition