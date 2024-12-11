import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from scipy.special import comb, perm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mc
import colorsys

# 绘图字体设置
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
# 层级社团标签的字体
font2 = {'family': 'Times New Roman',
            'style': 'normal',
            'weight': 'normal',
            'size': 22,
            }

# 1、带有真实标签网络划分的相关函数
# 将有向网络转换为无向网络
def direct_to_undirect_id(G1):
    '''
    input:
        G：networkx中的图结构(数据类型：nx.Graph)
    return:
       G1：networkx中的图结构(数据类型：nx.Graph)
    '''
    G = nx.Graph()
    G.add_nodes_from(list(G1.nodes()))
    G.add_edges_from(list(G1.edges()))
    return G

# 加载带有真实标签的网络和其节点对应的标签
def load_graph(Compound_list,data_id):
    '''
    input:
        Compound_list：数据集的集合(数据类型：list)
        data_id: 用于确定当前加载第几个数据集(数据类型：int)
    return:
       G：networkx中的图结构(数据类型：nx.Graph)
       y_true：每一个节点的标签(数据类型：list)
    '''
    # load graph
    if data_id in [2,3,4,5,6,7]:
        Compound = Compound_list[data_id]
        G = nx.read_gml(Compound,label='id')  # load a default graph
    elif data_id == 0:
        G = nx.karate_club_graph()   # 空手道俱乐部
    elif data_id in [1]:
        Compound = Compound_list[data_id]
        G1 = nx.read_gml(Compound,label='id')
        G = direct_to_undirect_id(G1)
    
    # load true community
    if data_id in [2,3,4,6,7]:
        labels = nx.get_node_attributes(G,'value')
        y_true = list(labels.values())
    elif data_id in [0,1,5]:
        if data_id == 0:
            labels = nx.get_node_attributes(G,'club')
        elif data_id == 1:
            labels = nx.get_node_attributes(G1,'value')
        else:
            labels = nx.get_node_attributes(G,'value')
        y = list(labels.values())
        # 将标签用数字表示，从0开始逐渐增大
        y_true_dict = {}
        k = 0
        for t in set(y):
            if t not in y_true_dict.keys():
                y_true_dict[t] = k
                k += 1
        y_true = [-1 for i in range(len(y))]
        for t in range(len(y)):
            y_true[t] = y_true_dict[y[t]]
            
    # network information
    print('*'*10,'Original network structure','*'*10)
    print('Graph : node %d edges %d community number %d'%(G.number_of_nodes(), G.number_of_edges(),len(set(y_true))))
    print('*'*48)
    return G,y_true

# 绘制网络的划分结果
def draw_graph(G,y_true):
    # draw graph with labels
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 7))
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=300, cmap=plt.cm.RdYlBu, node_color=y_true, linewidths=0.3, alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#BBBBBB', style="solid", alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="#191A19", font_weight="bold")
#     plt.savefig(r'result.pdf',bbox_inches='tight',dpi=300)
    plt.show(G)
    
# 绘制网络中节点度值和最短路径的分布
def plot_degree_shortpath(x,y,text,center_id,filepath='./',dataname='LS_default',save=False):
    '''
    input:
        x：节点的度值(数据类型：list)
        y：节点的最短路径(数据类型：list)
        text：节点的id(数据类型：list)
        center_id: LS算法识别的社团中心节点集合(数据类型：list)
        filepath：需要存储的文件路径(数据类型：str)
        dataname: 当前网络的名称(数据类型：str)
        save：是否需要存储文件(数据类型：boolean)
    return:
       plot
    '''
    fig, ax = plt.subplots(figsize=(8,7))
    basecolor = '#FFA900'
    edgecolor = adjust_lightness(basecolor, amount=1)
    # 绘图
    for i in range(len(x)):
        ax.scatter(x[i], y[i], c=basecolor, marker='o',s=200,edgecolor=edgecolor)
        if text[i] in center_id:
            ax.text(x[i], y[i]-0.03, str(text[i]), ha='center', fontsize=12,fontweight='bold')
    ax.set_xlabel(r'$k_i$',font)
    ax.set_ylabel(r'$l_i$',font)
    ax.tick_params(labelsize=16)
    fig.tight_layout()
    if save == True:
        filename = filepath + str(dataname) + '_degree_shortpath.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()

# # 绘制网络中节点和最短路径乘积的log-log分布
def plot_multi_log(x,y,text,center_id,filepath='./',dataname='LS_default',save=False,number_0 = 10):
    '''
    input:
        x：节点按照乘积~{k_i} * ~{l_i}的rank排序 (数据类型：list)
        y：~{k_i} * ~{l_i}(数据类型：list)
        text：节点的id(数据类型：list)
        filepath：需要存储的文件路径(数据类型：str)
        center_id: LS算法识别的社团中心节点集合(数据类型：list)
        dataname: 当前网络的名称(数据类型：str)
        save：是否需要存储文件(数据类型：boolean)
    return:
       plot
    '''
    # 用于具有层级社团网络的绘制
    center_to_label = {}
    center_label = ['a1','a2','a3','a4','b1','b2','b3','b4','c1','c2','c3','c4','d1','d2','d3','d4']
    for i in range(16):
        center_to_label[i] = center_label[i]
        
    fig, ax = plt.subplots(figsize=(8,7))
    basecolor = '#A73489'
    edgecolor = adjust_lightness(basecolor, amount=1)
        
    
    # 取x,y进行log-log处理，x整体加1,y最小值除以e,避免0值
    x_log = np.log(np.array(x)+1)
    x_new = []
    y_new = []
    y_min = min(filter(lambda x: x > 0, y))
    count_0 = 0
    for i in range(len(y)):
        if count_0 <= number_0:
            x_new.append(x_log[i])
            if y[i] != 0:
                y_new.append(np.log(y[i]))
            else:
                y_new.append(np.log(y_min/np.e))
                count_0 += 1
    # 绘图      
    for i in range(len(x_new)):
        if text[i] in center_id:
            ax.scatter(x_new[i], y_new[i], c=basecolor, marker='^',s=50,edgecolor=edgecolor)
#             ax.text(x_new[i], y_new[i]+0.02, center_to_label[int(text[i])//100], ha='center', fontsize=12)
        else:
            ax.scatter(x_new[i], y_new[i], c=basecolor, marker='o',s=50,edgecolor=edgecolor) 
    ax.set_xlabel(r'$ \ln \ rank $', fontsize=18)
    ax.set_ylabel(r'$\ln \, ( \~{k_i} \times \~{l_i} ) $', fontsize=18)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    if save == True:
        filename = filepath + str(dataname) + '_multiply.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
    
# 绘图中节点边缘颜色设置，比节点自己的颜色偏深一点   
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# 绘制联合图：大图表示度值和最短路径的分布，子图为网络中节点度值和最短路径乘积的分布
def plot_combination(x,y,text,x1,y1,text1,center_id,subplot_location,xlim_start_end,ylim_start_end,font_location,filepath='./',dataname='LS_default',save=False):
    '''
    input:
        x：节点的度值(数据类型：list)k
        y：节点的最短路径(数据类型：list)l
        x1：节点按照乘积~{k_i} * ~{l_i}的rank排序 (数据类型：list)
        y1：~{k_i} * ~{l_i}(数据类型：list)
        text：节点的id(数据类型：list)
        filepath：需要存储的文件路径(数据类型：str)
        center_id: LS算法识别的社团中心节点集合(数据类型：list)
        dataname: 当前网络的名称(数据类型：str)
        save：是否需要存储文件(数据类型：boolean)
    return:
       plot
    '''
        
    fig = plt.figure(figsize=(8,7))
    basecolor = '#FFA900'
    edgecolor = adjust_lightness(basecolor, amount=1)
    left, bottom, width, height = 0.1,0.1,0.8,0.8
    ax = fig.add_axes([left,bottom,width,height])
    for i in range(len(x)):
        # ax.scatter(x[i], y[i], c=basecolor, marker='o',s=200,edgecolor=edgecolor)
        if text[i] in center_id:
            ax.text(x[i], y[i]+ font_location, str(text[i]), ha='center', fontsize=12,fontweight='bold')
    ax.scatter(x,y, c=basecolor, marker='o',s=200)
    # for i in center_id:
    #     i = int(i)
    #     ax.text(x[i], y[i] + font_location, str(text[i]), ha='center', fontsize=12, fontweight='bold')

    if np.max(np.array(x))//10 < 1:
        x_unit = 1
    else:
        x_unit = np.max(np.array(x))//10
    if np.max(np.array(y))//10 < 1:
        y_unit = 1
    else:
        y_unit = np.max(np.array(y))//10
    x_major_locator = MultipleLocator(x_unit)
    y_major_locator = MultipleLocator(y_unit)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlim(xlim_start_end[0],max(x)+xlim_start_end[1])
    ax.set_ylim(ylim_start_end[0],max(y)+ylim_start_end[1])
    ax.set_xlabel(r'$k_i$',font)
    ax.set_ylabel(r'$l_i$',font)
    ax.tick_params(labelsize=16)
    
    font1 = {'family': 'Times New Roman',
            'style': 'italic',
            'weight': 'normal',
            'size': 16,
            }
    basecolor = '#A73489'
    edgecolor = adjust_lightness(basecolor, amount=1)
#     left, bottom, width, height = 0.25,0.595,0.35,0.3 # darkar
#     left, bottom, width, height = 0.18,0.55,0.35,0.3 # Abidjan
#     left, bottom, width, height = 0.25,0.55,0.35,0.3 # Beijing
    # 添加子图
    left, bottom, width, height = subplot_location[0],subplot_location[1],subplot_location[2],subplot_location[3]
    ax1 = fig.add_axes([left,bottom,width,height])
    # 对x1,y1进行log-log处理
    x1_new = np.log(np.array(x1)+1)
    y1_new = []
    y1_min = min(filter(lambda x: x > 0, y1))
    for i in range(len(y1)):
        if y1[i] != 0:
            y1_new.append(np.log(y1[i]))
        else:
            y1_new.append(np.log(y1_min/np.e))

    #     for i in range(len(x1_new)):
#         if text1[i] in center_id:
#             ax1.scatter(x1_new[i], y1_new[i], color=basecolor, marker='^',s=20,edgecolor=edgecolor)
#         else:
#             ax1.scatter(x1_new[i], y1_new[i], color=basecolor, marker='o',s=2,edgecolor=edgecolor)
# #             ax1.text(x1_new[i], y1_new[i]-0.1, str(int(text1[i])), ha='center', fontsize=10,fontweight='bold')
    ax1.scatter(x1_new, y1_new, color=basecolor, marker='o',s=2)
    center_x = []; center_y = []
    for i in range(len(x1_new)):
        if text1[i] in center_id:
            center_x.append(x1_new[i])
            center_y.append(y1_new[i])
    ax1.scatter(center_x, center_y, color=basecolor, marker='^', s=20)

    ax1.set_xlabel(r'$\ln \, rank$',font1)
    ax1.set_ylabel(r'$\ln \, ( \~{k_i} \times \~{l_i} ) $',font1)
    ax1.tick_params(labelsize=16)
    fig.tight_layout()
    if save == True:
        filename = filepath + str(dataname) + '_process.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()

# 绘制联合图：大图表示度值和最短路径的分布，子图为网络中节点度值和最短路径乘积的分布
def plot_combination_without_centers(x,y,text,x1,y1,text1,center_id,subplot_location,xlim_start_end,ylim_start_end,font_location,filepath='./',dataname='LS_default',save=False):
    '''
    input:
        x：节点的度值(数据类型：list)
        y：节点的最短路径(数据类型：list)
        x1：节点按照乘积~{k_i} * ~{l_i}的rank排序 (数据类型：list)
        y1：~{k_i} * ~{l_i}(数据类型：list)
        text：节点的id(数据类型：list)
        filepath：需要存储的文件路径(数据类型：str)
        center_id: LS算法识别的社团中心节点集合(数据类型：list)
        dataname: 当前网络的名称(数据类型：str)
        save：是否需要存储文件(数据类型：boolean)
    return:
       plot
    '''
        
    fig = plt.figure(figsize=(8,7))
    basecolor = '#FFA900'
    edgecolor = adjust_lightness(basecolor, amount=1)
    left, bottom, width, height = 0.1,0.1,0.8,0.8
    ax = fig.add_axes([left,bottom,width,height])
    for i in range(len(x)):
        ax.scatter(x[i], y[i], c=basecolor, marker='o',s=200,edgecolor=edgecolor)
    
    if np.max(np.array(x))//10 < 1:
        x_unit = 1
    else:
        x_unit = np.max(np.array(x))//10
    if np.max(np.array(y))//10 < 1:
        y_unit = 1
    else:
        y_unit = np.max(np.array(y))//10
    x_major_locator = MultipleLocator(x_unit)
    y_major_locator = MultipleLocator(y_unit)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlim(xlim_start_end[0],max(x)+xlim_start_end[1])
    ax.set_ylim(ylim_start_end[0],max(y)+ylim_start_end[1])
    ax.set_xlabel(r'$k_i$',font)
    ax.set_ylabel(r'$l_i$',font)
    ax.tick_params(labelsize=16)
    
    font1 = {'family': 'Times New Roman',
            'style': 'italic',
            'weight': 'normal',
            'size': 16,
            }
    basecolor = '#A73489'
    edgecolor = adjust_lightness(basecolor, amount=1)
#     left, bottom, width, height = 0.25,0.595,0.35,0.3 # darkar
#     left, bottom, width, height = 0.18,0.55,0.35,0.3 # Abidjan
#     left, bottom, width, height = 0.25,0.55,0.35,0.3 # Beijing
    # 添加子图
    left, bottom, width, height = subplot_location[0],subplot_location[1],subplot_location[2],subplot_location[3]
    ax1 = fig.add_axes([left,bottom,width,height])
    # 对x1,y1进行log-log处理
    x1_new = np.log(np.array(x1)+1)
    y1_new = []
    y1_min = min(filter(lambda x: x > 0, y1))
    for i in range(len(y1)):
        if y1[i] != 0:
            y1_new.append(np.log(y1[i]))
        else:
            y1_new.append(np.log(y1_min/np.e))
    for i in range(len(x1_new)):
        ax1.scatter(x1_new[i], y1_new[i], c=basecolor, marker='o',s=2,edgecolor=edgecolor)
#             ax1.text(x1_new[i], y1_new[i]-0.1, str(int(text1[i])), ha='center', fontsize=10,fontweight='bold')
    ax1.set_xlabel(r'$\ln \, rank$',font1)
    ax1.set_ylabel(r'$\ln \, ( \~{k_i} \times \~{l_i} ) $',font1)
    ax1.tick_params(labelsize=16)
    fig.tight_layout()
    if save == True:
        filename = filepath + str(dataname) + '_process.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()

# 计算评价指标：precesion,recall,F1
def cal_auc(y_pred,y_true):
    '''
    input:
        y_pred：节点的预测标签(数据类型：list)
        y_true: 节点的真实标签(数据类型：list)
    return:
       precesion,recall,F1：评价指标(数据类型：float,float,float)
    '''
    labels = {}
    for index,label in enumerate(y_true):
        if label not in labels.keys():
            labels[label] = []
            labels[label].append(index)
        else:
            labels[label].append(index)
    TP = 0
    FP = 0
    TP_all = 0
    FP_all = 0
    # check True(同一标签组的比较)
    for label in labels.keys():
        index_list = labels[label]
        for i in range(len(index_list))[:-1]:
            for j in range(len(index_list))[i+1:]:
                TP_all += 1
                if y_pred[index_list[i]] == y_pred[index_list[j]]:
                    TP += 1
    FN = TP_all - TP             
    # check False(不同标签组的比较)
    label_list = list(labels.keys())
    for i in range(len(label_list))[:-1]:
        for j in range(len(label_list))[i+1:]:
            index_list1 = labels[label_list[i]] 
            index_list2 = labels[label_list[j]]
            for k in index_list1:
                for t in index_list2:
                    FP_all += 1
                    if y_pred[k] == y_pred[t]:
                        FP += 1
    TN = FP_all - FP
#     print('TP,FP,FN,TN:',TP,FP,FN,TN)
    if TP+FP == 0:
        precesion = 0
    else:
        precesion = TP/(TP+FP)
    if TP+FN == 0:
        recall = 0
    else:
        recall =  TP/(TP+FN)
    if 2 * TP + FP + FN == 0:
        F1 = 0
    else:
        F1 = 2 * TP /( 2 * TP + FP + FN)
    return precesion,recall,F1

# 2、层级网络相关函数
# 绘制层级网络的邻接矩阵分布图
def plot_matrix(G,filepath='./',dataname='LS_default',save=False):
    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.get_cmap("Greys")
    matrix = nx.adjacency_matrix(G).todense()
    ax.set_xlim(0,1600)
    ax.set_ylim(0,1600)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.invert_yaxis()
    ax.tick_params(labelsize=18)
    
    ax1 = ax.twinx().twiny()
    ax1.set_xlim(0,1600)
    ax1.xaxis.set_major_locator(MultipleLocator(100))
    ax1.yaxis.set_major_locator(MultipleLocator(400))
    ax1.set_xticklabels('')
    ax1.set_xticks([50+100*i for i in range(16)], minor=True)
    ax1.set_xticklabels(['a1','a2','a3','a4','b1','b2','b3','b4','c1','c2','c3','c4','d1','d2','d3','d4'], minor=True, fontdict=font2)
    ax1.tick_params(axis='x',which='minor',labelsize=18,top=False,right=False,bottom=False,left=False)
    ax1.set_yticklabels('')
    
    ax2 = ax.twiny().twinx()
    ax2.set_ylim(1600,0)
    ax2.xaxis.set_major_locator(MultipleLocator(100))
    ax2.yaxis.set_major_locator(MultipleLocator(400))
    ax2.set_yticklabels('')
    ax2.set_yticks([200+400*i for i in range(4)], minor=True)
    ax2.set_yticklabels(['a','b','c','d'], minor=True, fontdict=font)
    ax2.set_xticks([])
    ax2.tick_params(axis='y', which='minor',labelsize=18,top=False,right=False)
    ax2.set_xticklabels('')
    plt.imshow(matrix,cmap=cmap,origin ='upper',aspect='auto')
    if save == True:
        filename = filepath + str(dataname) + '_matrix.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
        
def draw_degree_distrubution(y_init,reference_show=False,t1=0,t2=0,filepath='./',dataname='LS_default',save=False):
#     y_sorted = sorted(y)
    fig, ax = plt.subplots(figsize=(8,7))
    y_mean = np.mean(np.array(y_init))
#     print('degree:',y_init[-50:])
#     print('mean degree:',len(y_init),y_mean)
    y_count = Counter(y_init)
    x = y_count.keys()
    y = y_count.values()
    ax.plot(x,y,color='darkblue',linewidth=0.5, linestyle='-',marker='o', markersize=5, alpha = 0.7)
    ax.set_xlabel(r'Degree', fontsize=18)
    ax.set_ylabel(r'Number', fontsize=18)
    ax.tick_params(labelsize=18)
    if reference_show == True:
        plt.axvline(x=t1,color='red',alpha=1.0) #34 ER 49 BA
        plt.axvline(x=t2,color='blue',alpha=1.0) #36 ER 49 BA
    fig.tight_layout()
    if save == True:
        filename = filepath + str(dataname) + '_degree_distribution.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
