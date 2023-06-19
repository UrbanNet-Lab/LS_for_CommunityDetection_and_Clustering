# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:00:36 2021

Produces Degree Hierarchy Tree from a network

Based on
Hidden directionality unifies community detection and cluster analysis

Fan Shang, Bingsheng Chen, Linyuan Lü, H. Eugene Stanley, Ruiqi Li

Implementation and interpreation by Tim Evans (Imperial)

@author: tseva
"""


#import os
import random
import networkx as nx
import numpy as np
from queue import Queue
from datetime import datetime
import community
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mc
import colorsys
from LS_other_function import plot_combination

    
# 寻找local leader的隐含方向性
def BFS_from_s(G,s,roots):
    '''
    input:
        G：networkx中的图结构(数据类型：nx.Graph)
        s: BFS开始的起始节点(数据类型：int)
        roots： local leader(数据类型：list)
    return:
       w：指向节点的id(数据类型：int),不存在时返回自己
       p：最短路经长度(数据类型：int)，不存在时返回-1
    '''
    queue=[]
    queue.append(s)
    seen=set()#看是否访问过该结点
    seen.add(s)
    path_dict = {} # 记录root到每个节点的距离
    path_dict[s] = 0
    while (len(queue)>0):
        vertex =queue.pop(0)#保存第一结点，并弹出，方便把他下面的子节点接入
        neighbors = [(neighbor,G.degree[neighbor]) for neighbor in list(G.adj[vertex]) if neighbor not in seen]#子节点的数组
        nodes = [node[0] for node in sorted(neighbors, key=lambda k: k[1], reverse=True)]
#         print('nodes',vertex,nodes)
        for w in nodes:
            if w not in seen:#判断是否访问过，使用一个数组
                path_dict[w] = path_dict[vertex] + 1
                queue.append(w)
                seen.add(w)
            if G.degree[w] > G.degree[s] and w in roots:
                return w,path_dict[w]
    return s,-1

# 按照数组的大小返回rank排序的数组
def get_indacator_rank(x):
    set_x = set(x)
    sorted_x = sorted(set_x, reverse=False)
    set_x_dict = {}
    k = 1
    for i in sorted_x:
        if i not in set_x_dict.keys():
            set_x_dict[i] = k
            k += 1
    rank_x = []
    for i in x:
        rank_x.append(set_x_dict[i])
    return rank_x

# 对数组中的每一个值进行平方，返回新数组
def get_square(x):
    square_x = []
    square_x = [np.power(i,2) for i in x]
    return square_x

# min-max归一化
def standard_data(x):
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max - x_min == 0:
        trans_data = np.array([1/len(x) for i in range(len(x))])
    else:
        trans_data = (x - x_min) /  (x_max - x_min)
    return trans_data

# 计算y值之间的差值
def cal_delta(y):
    y_delta = []
    for i in range(len(y))[1:]:
        y_delta.append(abs(y[i] - y[i-1]))
    return np.array(y_delta)

# 根据y之间的差值自动选择聚类中心的个数
def choose_center(multi_sort):
    y = multi_sort[:,1]
    delta = cal_delta(y)
    delta_nozero = [i for i in delta if i!=0]
    delta_std = np.std(delta_nozero)
    center_num = 0
    for i in range(len(delta)):
        if delta[i] > delta_std + np.mean(delta_nozero):
            center_num = i + 1
            break
    return center_num

def full_degree_hierarchy_dag_OLD(G):
    '''
    DEPRECATED
    Create a full degree hierarchy DAG from  networkx graph G
    
    All edges are directed from lower to higher degree nodes, 
    only edges not included are those between equal degree nodes.
    
    Input
    -----
    G -- a simple networkx graph of one component 
     
    Return
    ------
    D -- A directed acyclic graph
    '''
    D = nx.DiGraph()
    D.add_nodes_from(G)
    for e in G.edges:
        ksource_minus_ktarget = G.degree[e[0]] - G.degree[e[1]]
        if ksource_minus_ktarget>0:
            D.add_edge( e[1],e[0] ) # point to larger degree node
        elif ksource_minus_ktarget<0:
            D.add_edge( e[0],e[1] ) # point to larger degree node
    
    print("! With "+str(G.number_of_nodes())+" nodes, from "+str(G.number_of_edges())+" to "+str(D.number_of_edges())+" edges in Full Degree DAG")
    return D        

def full_degree_hierarchy_dag(G):
    '''
    Create a full degree hierarchy DAG from  networkx graph G
    
    All edges are directed from lower to higher degree nodes, 
    only edges not included are those between equal degree nodes.
    
    Input
    -----
    G -- a simple networkx graph of one component 
     
    Return
    ------
    D -- A directed acyclic graph
    '''
    D = nx.DiGraph()
    D.add_nodes_from(G)
    for v in G.nodes:
        kv=G.degree(v)
        e_list = [(v,nn) for nn in G.neighbors(v) if G.degree(nn)>kv]  # point to larger degree node
        D.add_edges_from( e_list ) 
    
    # print("! With "+str(G.number_of_nodes())+" nodes, from "+str(G.number_of_edges())+" to "+str(D.number_of_edges())+" edges in Full Degree DAG")
    
    return D        

def max_degree_hierarchy_dag(G):
    '''
    Create a maximum degree hierarchy DAG from  networkx graph G

    All edges present are from a source node to neighbours which have a larger degree 
    and the degree of these neigthbours is is larger than or equal to than the degree 
    of all the neighbours of the source vertex.
    
    The difference from the full_degree_hierarchy_dag method is that this 
    does not inlcude links to neighbours which have a higher degree than the source node
    but still that neighbouir has a degree which is less than the largest degree 
    of all the neighbours.
    
    Input
    -----
    G -- a simple networkx graph of one component 
     
    Return
    ------
    D -- A directed acyclic graph
    '''
    D = nx.DiGraph()
    D.add_nodes_from(G)
    for v in G.nodes:    
        degree_list = [G.degree(nn) for nn in G.neighbors(v)]
        if len(degree_list) > 0:
            knnmax = max(degree_list)
            if knnmax >= G.degree[v]:
                # has neighbours with the largest degree so add all of the edges to this neighbour
                # here edge points from low degree to high degree, points towards tree root
                e_list = [(v,nn) for nn in G.neighbors(v) if G.degree(nn)==knnmax and (not D.has_edge(nn,v))] 
                D.add_edges_from( e_list ) 
        else:
            continue
    # print("! With "+str(G.number_of_nodes())+" nodes, from "+str(G.number_of_edges())+" to "+str(D.number_of_edges())+" edges in Maximum Degree DAG")
#     print(D.edges())
    return D

def degree_hierarchy_random_tree(G, maximum_tree=True, random_seed=None):
    '''
    Create a degree hierarchy tree from  networkx graph G.
    
    Unless seed=None, this uses random numbers to break ties 
    where neighbours have same (maximum) degree and they are
    both at the same distance from a root node.
    
    Input
    -----
    G -- an simple networkx graph of one component 
    maximum_tree=True -- If true uses maximum dgree DAG as input, otherwise uses full degree DAG 
    random_seed -- an integer if ties to be broken at random, None for no random element
    
    Return
    ------
    D, tree_edge_list 
    
    D --- A directed acyclic graph
    tree_edge_list --- list of edges in terms of node id used in G of a shortest path tree in G
    '''
    if random_seed != None:
        random.seed(random_seed)
    
    if maximum_tree:
        D=max_degree_hierarchy_dag(G)
    else:
        D=full_degree_hierarchy_dag(G)
    
    node_queue = Queue(maxsize=0)
    # start queue for DFS from all the root nodes
    # Each entry in queue is tuple (parent_node, node, root_node,  shortest_distance_to_root)
    parent_node=None
    shortest_distance_to_root=0
    for root_node in D:
        if D.out_degree[root_node]==0:
#             print("Adding root node "+str(root_node))
            node_queue.put( (None, root_node, shortest_distance_to_root) ) 
    number_of_ties=0
    while not node_queue.empty():
        parent_node, next_node , shortest_distance_to_root=node_queue.get()
        if "distancetoroot" in D.nodes[next_node]:
            if D.nodes[next_node]["distancetoroot"] < shortest_distance_to_root: 
                continue  # already found a quicker way from next_node to a root node
            if D.nodes[next_node]["distancetoroot"]== shortest_distance_to_root:
                number_of_ties+=1
                if random.random()<0.5:
                    continue # a simple way to implement randomness where there is a choice of shortest path roots
        if parent_node==None: # Must be a root node
            D.nodes[next_node]["rootnode"]=next_node
        else:    
            D.nodes[next_node]["rootnode"]=D.nodes[parent_node]["rootnode"]
        D.nodes[next_node]["parentnode"]=parent_node
        D.nodes[next_node]["distancetoroot"]=shortest_distance_to_root
#         print(next_node,parent_node,shortest_distance_to_root)
        nn_list = [ (next_node, nn, shortest_distance_to_root+1) for nn in D.predecessors(next_node) ]
        for nn in nn_list:
            node_queue.put(nn)    
    tree_edge_list=[]
    
    for node in D:
        parent_node=D.nodes[node]["parentnode"]
        if parent_node!=None:
            tree_edge_list.append( (parent_node,node) )
    
    # print("! In degree_hierarchy_random_tree broke "+str(number_of_ties)+" ties at random")
    return D, tree_edge_list        
 

def hierarchical_degree_communities(G, center_num=1, auto_choose_centers=False, maximum_tree=True, seed=None):
    '''
    Produces hierarchical degree forest (HDF) of trees and hence communities.
    
    Input
    -----
    G -- simple graph for which communities are required
    maximum_tree=True -- If true uses maximum dgree DAG as input, otherwise uses full degree DAG 
    seed=None -- an integers to use as a seed to break ties at random.  Use None to remove random element
    
    Output
    ------
    On screen statistics of communities
    
    '''
    start_time = datetime.now()
    treename="Hierarchical Maximum Degree Forest"
    treeabv="HMDF"
    if not maximum_tree:
        treename="Hierarchical Full Degree Forest"
        treeabv="HFDF"
        
    # print ("\n===== "+treename+" seed "+str(seed)+" =====")
    print("\n====Local Search Algorithm (random seed " +str(seed) +")==========")
    print("Network: "+str(len(G.nodes))+" nodes,"+str(len(G.edges))+" edges")
    D, tree_edge_list = degree_hierarchy_random_tree(G, maximum_tree=maximum_tree, random_seed=seed)
    # print("With "+str(G.number_of_nodes())+" nodes, now left with "+str(len(tree_edge_list))+" edges in tree" )
    
    # Now find all the nodes with the same root node, i.e. communities
    root_to_node={}
    for node in D:
        if "rootnode" in D.nodes[node]:
            root_node = D.nodes[node]["rootnode"]
        else:
            print("*** ERROR Node "+str(node)+" has no rootnode")
            continue
        if root_node not in root_to_node:
            root_to_node[root_node]=[]
        root_to_node[root_node].append(node)
    
    # determine centers from root_to_node
    # (1). 通过local-BFS计算local leader的指向和最短路径
    Potential_Center = list(root_to_node.keys())
    # print("! Number of Communities (root nodes) found "+str(len(root_to_node)))
#     print("  Root Nodes: ",Potential_Center)
    
    root_number = len(root_to_node)
    root_decision = {}
    avg_l = 0
    # print('Intermediate process of determining the center: ')
    for node in root_to_node.keys():
        e,p = BFS_from_s(G,node,Potential_Center)
        root_decision[node] = [e,p,G.degree[node]]
        

    # 度值最大的节点和噪声节点的最短路径长度设置为所有节点中最短路径长度的最大值
    max_path_temp = max(np.array(list(root_decision.values()))[:,1])
    # print("max_path_temp  == ",max_path_temp," type",type(max_path_temp))
    max_path_temp = int(max_path_temp)
    max_path = max_path_temp if max_path_temp > -1 else 2 
    for node in root_decision:
        if root_decision[node][1] == -1:
            root_decision[node] = [root_decision[node][0],max_path,root_decision[node][2]]
    
    # 如果local leader中节点度值一样，将其最短路径长度从2开始逐渐递增：用于处理football中所有roots度值一样的情况
    degreeequal_len = 2
    root_degree_set = len(set(np.array(list(root_decision.values()))[:,2]))
    if root_degree_set == 1:
        for n in root_decision:
            root_decision[n] = [root_decision[n][0],degreeequal_len,root_decision[n][2]]
            degreeequal_len += 1
 
    # (2).计算所有节点度值和最短路径的分布
    node_plot = root_decision.copy()
    for n in G.nodes():
        if n not in node_plot:
            node_plot[n] = [D.nodes[n]['parentnode'],1,G.degree[n]]
    
    root_array = np.array(list(node_plot.values())) 
#     print('degree, path',root_array[:,2],root_array[:,1])
    degree = get_indacator_rank(root_array[:,2])
    shortest_path = get_square(root_array[:,1])
    degree_standard = standard_data(np.array(degree))
    shortest_path_standard = standard_data(np.array(shortest_path))
    multi = degree_standard * shortest_path_standard
    nodeid = list(node_plot.keys())
    multi_dict = {}
    for i in range(len(nodeid)):
        multi_dict[nodeid[i]] = multi[i]
    multi_sort = np.array(sorted(multi_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse= True)) 
    multi_sort = np.array([[int(i[0]),i[1]] for i in multi_sort])
    multi_x = [i for i in range(len(multi_sort))] #横轴
#     print('Determine centers by muti:',multi_sort[:40])
    
    # 选择聚类中心
    if auto_choose_centers == True:
        auto_centernum = choose_center(multi_sort)
        center_num = auto_centernum if center_num < auto_centernum else center_num
    center_dcd = []
    for i in multi_sort[:center_num]:
        center_dcd.append(int(i[0]))
    
    # 保存绘图需要的数据
    plot_combination_data = [root_array[:,2],root_array[:,1],nodeid, multi_x, multi_sort[:,1],multi_sort[:,0],center_dcd]
    plot_process_degree_shortpath_data = [degree,shortest_path,nodeid]

    local_cnt = 0
    for i in multi_sort[:,1]:
        if i>0:
            local_cnt+=1
    print("The number of local leaders: "+str(local_cnt))
    
    # (3).iteration
    for node in root_to_node.keys():
        D.nodes[node]["parentnode"] = root_decision[node][0]
        D.nodes[node]["rootnode"] =  D.nodes[node]["parentnode"]
    for node in D.nodes():
        recent_node = [] # prevent loop
        recent_node.append(node)
        flag = 0
        if node in center_dcd:
            D.nodes[node]["rootnode"] = node
        else:
            while D.nodes[node]["rootnode"] not in center_dcd and flag == 0:
                j = D.nodes[node]["rootnode"]
                if j not in recent_node and j != None:
                    recent_node.append(j)
                    D.nodes[node]["rootnode"] = D.nodes[j]["rootnode"]
                else:
                    D.nodes[node]["rootnode"] = None
                    flag = 1
                    
    # 4. get the classes and partition
    y_dcd = []
    y_partition = {}
    for node in D.nodes():
        if D.nodes[node]["rootnode"] == None:
            y_dcd.append(-1)
            y_partition[node] = -1
        else:
            y_dcd.append(D.nodes[node]["rootnode"])
            y_partition[node] = D.nodes[node]["rootnode"]

    end_time = datetime.now()
    stamp = (end_time-start_time).total_seconds()*1000
    print('Running Time: %d ms' % stamp)
    # print('The number of community  centers: '+str(center_dcd))
    print('The number of community  centers: ' + str(len(plot_combination_data[6])))
    print('The id of the centers are: '+str(plot_combination_data[6]))
    print('Modularity of the partition by LS: '+str(community.modularity(y_partition, G)))

    print("The decision graph for determining the number of centers "+
          "(centers are nodes with both a large degree k_i and path length l_i; \n"+
          "there might be multi-scale community structure, which is signified by notable gaps):")
    subplot_location = [0.25, 0.55, 0.35, 0.3]
    xlim_start_end = [0.3, 0.7]
    ylim_start_end = [0.7, 0.3]
    font_location = -0.04
    plot_combination(plot_combination_data[0], plot_combination_data[1], plot_combination_data[2],
                     plot_combination_data[3], plot_combination_data[4], plot_combination_data[5],
                     plot_combination_data[6], subplot_location, xlim_start_end, ylim_start_end, font_location,
                     filepath='./', save=False)
    print("Note: The number of communities can be explicitly set according to the results in decision graph, \n"+
          "if multi-scale community structure is of interests")
    return D,center_dcd,y_dcd,y_partition,plot_combination_data


if __name__ == '__main__':
    seed_list=range(1)

    print("  ### Simple (extreme) example of network where this method does not produce a unique community ###")    
    G=nx.Graph()
    G.add_edges_from([ (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5) ])
    for seed in seed_list:
        hierarchical_degree_communities(G, maximum_tree=True, seed=seed)
        hierarchical_degree_communities(G, maximum_tree=False, seed=seed)
    #D=degree_hierarchy_dag(G)

    print("\n\n  ### Karate Club Network ###")    
    G=nx.karate_club_graph()
    for seed in seed_list:
        hierarchical_degree_communities(G, maximum_tree=True, seed=seed)
        hierarchical_degree_communities(G, maximum_tree=False, seed=seed)

        
    # print("\n\n  ### Les Miserables Network ###")
    # G=nx.les_miserables_graph()
    # hierarchical_degree_communities(G,maximum_tree=True)
    # hierarchical_degree_communities(G,maximum_tree=False)
    
    