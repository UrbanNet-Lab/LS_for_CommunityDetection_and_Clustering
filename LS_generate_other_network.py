import random
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mc
import colorsys

# 1 生成层级网络
def combine_2_graph_with_p(G1,G2,p):
    """
    使用概率p将两个图连接起来
    
    连接方法： 
        G1中的n个结点和G2中的m个结点 
        组成n*m个结点对，每个结点对以概率p建立一条连边 
    """
    # 1.将原有结点和连边添加进来
    G = nx.Graph()
    
    G.add_nodes_from(G1)
    G.add_nodes_from(G2)
    
    edges1 = list(G1.edges())
    edges2 = list(G2.edges())
    
    G.add_edges_from(edges1)
    G.add_edges_from(edges2)
    
    G1nodes = G1.nodes()
    G1nodes = list(G1nodes)
    
    G2nodes = G2.nodes()
    G2nodes = list(G2nodes)
    
    #2.G1和G2之间添加连边
    
    for i in range(len(G1nodes)):
        for j in range(len(G2nodes)):
            rand = random.uniform(0, 1)
            if rand < p: 
                G.add_edge(G1nodes[i],G2nodes[j])
    return G 

def generate_scale_free_network(nodes,m):
    """
   N:图中的结点数 
    Number of nodes in graph
        alpha：

    alpha：通过入度随机选择一个结点
    Probability for adding a new node connected to an existing node chosen randomly according to the in-degree distribution.
    根据入度分布随机选择一个新节点并将其连接到现有节点的概率。
    
    beta：  Probability for adding an edge between two existing nodes. One existing node is chosen randomly according the in-degree distribution and the other chosen randomly according to the out-degree distribution.
    在两个现有节点之间添加边的概率。根据入度分布随机选择一个现有节点，根据出度分布随机选择另一个节点。
    
    gamma： Probability for adding a new node connected to an existing node chosen randomly according to the out-degree distribution.
    添加新节点的概率，该新节点连接到根据出度分布随机选择的现有节点
    
    The sum of alpha, beta, and gamma must be 1.
    """
    N = len(nodes)
#     G = nx.scale_free_graph(N)
    #另一种建立scale free的方法
#     edgeNum = int(p*len(nodes)*len(nodes)*0.5)
    G = nx.random_graphs.barabasi_albert_graph(N,m)
#     G = nx.barabasi_albert_graph(len(nodes),edgeNum)

    mapping = dict()
    for i in range(len(nodes)):
        mapping[i]=nodes[i]
    G = nx.relabel_nodes(G,mapping)
#     G.nodes(),G.edges()

    return G 

def combine_n_graph_with_p(GList,p,addP):
    """
    使用概率p将GList中的n个图连接起来
    概率p会有addP幅度的抖动 也就是说最终的概率为p*(1+addP)
    
    连接方法： 
        Gi中的n个结点和 剩余n-1个G中的m个结点 
        组成n*m*(n-1)个结点对，每个结点对以概率p建立一条连边 
        
    输入：要结合的GList，一个图和其他图连接的概率p，p的抖动值addP 区间为[-1,1]
    """
   
    G = nx.Graph()
    for i in range(len(GList)):
        
        # 1.将Gi原有结点和连边添加进来
        G.add_nodes_from(GList[i])
        edges = list(GList[i].edges())
        G.add_edges_from(edges)
    
    for i in range(len(GList)):
        # 2.获取第i个G结点列表
        Ginodes = GList[i].nodes()
        Ginodes = list(Ginodes)
        for j in range(i+1,len(GList)):
            #3. 获取第j个G结点列表
            Gjnodes = GList[j].nodes()
            Gjnodes = list(Gjnodes)
            
            #4. 将第i个和第j个结点进行随机连边
            for ii in range(len(Ginodes)):
                for jj in range(len(Gjnodes)):
                    rand = random.uniform(0, 1)
                    if rand < p*(1+addP): 
                        G.add_edge(Ginodes[ii],Gjnodes[jj])
    
    #5. 将最后的结果返回
    return G

def generate_random_graph(p,nodes,addP):
    """
    生成一个随机图
    
    input：结点列表nodes，每个结点对之间的连接概率是p
    output：随机图G 
    
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = len(nodes)
    
    G = nx.Graph()
    G.add_nodes_from(nodes)

#     N = 8
    m = np.random.random(N*N).reshape(N, N)
    for i in range(N): m[i, i] = 0
        
    for i in range(N):
        for j in range(i+1,N):
            if m[i,j]<p*(1+addP):
                G.add_edge(nodes[i],nodes[j])

    return G


def add_Heightening_pad(G,Biggest = True,p = 0.8):
    """
    Input:
        G:需要进行处理的graph
        Biggest：是否给degree最大的结点加边 
        p：加边的概率是多少 
    
    选择需要加边的结点i：
        Biggest = True：该结点为度最大的结点 
        Biggest = False：结点随机选出 
    
    加边规则：
        结点i
        遍历剩下的N-1个结点
        如果与结点i有连接 则不做处理 
        如果没有则以概率p进行加边 
        
    output：处理过后的Graph 
    """
    nodes = list(G.nodes())
    target = -1 
    maxdegree = -1
    N = len(nodes)
    
    # 1.找到目标节点target
    if Biggest == True:
        for i in range(N):
            if G.degree(nodes[i]) > maxdegree:
                maxdegree = G.degree(nodes[i])
                target = i 
    else:
        target = int(random.uniform(0, N))
#     target = int(len(nodes)/2)
#     print("### target id = = ",target,"target node == ",nodes[target]," nodes == ",nodes[0],"~",nodes[-1])
    # 2. 以p的概率加边
    for i in range(N):
        if G.has_edge(nodes[target],nodes[i] ) == True:
            continue
        rand  = random.uniform(0, 1)
        if rand < p :
            G.add_edge(nodes[i],nodes[target])
    return G

def generate_hierarchy_graph(if_boost,pList,Gsize,groupNum,addP,bottom_graph_random_or_scaleFree = False):
    """
    p0~pn：分别是从最底层到最上面的连接概率
    Gsize:每个最底层图的大小 
    groupNum:将几个图结合在一起
    addP:概率的变化幅度 在[-1，1]之间
    bottom_graph_random_or_scaleFree = True:最底层生成的图是random/ False:scaleFree
    """
#     addP = 0.005
    GList = []
#     centerP = [0.9,0.8,0.7,0.75]
#     centerP = [0.9,0.45,0.45,0.45]
    centerP = [0.9]+[0.35]*(groupNum-1)
    for pindex in range(len(pList)):
        print(" pindex ",pindex)
        if len(GList) == 0:
            num = pow(groupNum,len(pList)-1)
            for k in range(num):
                nodes = list(range(k*Gsize,(k+1)*Gsize))#每一组是Gsize个
                #生成random图
                if bottom_graph_random_or_scaleFree == True:
                    tempG = generate_random_graph(pList[pindex],nodes,addP)
                #生成scale-free网络
                else:
                    tempG = generate_scale_free_network(nodes,int(pList[pindex]))
                if if_boost == True:
                    tempG = add_Heightening_pad(tempG,True,centerP[k%groupNum])
                GList.append(tempG)
        else:
            print(pindex,len(GList))
            GListTmp = []
            for k in range(0,len(GList),groupNum):
                #将一个groupNum内的结合起来
                combList = GList[k:k+groupNum]
#                 for t in range(k,k+groupNum):
#                     combList.append(GList[t])
                
                tmpG = combine_n_graph_with_p(combList,pList[pindex],addP)
#                 3. 加边的函数
#                 if pindex>0 and pindex < len(pList)-1:
#                     tmpG = add_Heightening_pad(tmpG,False,0.6)
                ###############################################################
#                 print(" k == ",k)
# #                 print(tmpG.nodes())
#                 matrix = nx.adjacency_matrix(tmpG).todense()
#                 plt.imshow(matrix)
#                 plt.show()
                ###############################################################
                
                GListTmp.append(tmpG)
#                 k = k+1 
            GList = GListTmp
#     print(len(GList) == 1)
    return GList[0]

# 生成ravasz
def generate_Um():
    edge_list=[]
    Um =nx.Graph()
    nodes=list(range(1,126))
    #生成126个节点
    Um.add_nodes_from(nodes)
    #图加入节点
    for i in range(0,25):
        edge_listx=[(5+5*i,1+5*i),(5+5*i,2+5*i),(5+5*i,3+5*i),(5+5*i,4+5*i),(1+5*i,3+5*i),(1+5*i,4+5*i),(2+5*i,3+5*i),(2+5*i,4+5*i)] 
        edge_list=edge_list+edge_listx
        
    for j in range(0,5):
        for i in range(6,26):
            if((j*25+i)%5!=0):
                edge_list.extend([(5, j*25+i)])
                edge_list.extend([(5+25*j, j*25+i)])           
    Um.add_edges_from(edge_list)
    y_true = []
    for i in range(0,125):
        y_true.append(i//25)
        
    print('Graph : node %d edges %d'%(Um.number_of_nodes(), Um.number_of_edges()))
    return Um,nodes,edge_list,y_true

def plot_Um(Um,nodes,edge_list,y_predict,filepath='./',dataname='LS_default',save=False):
    plt.figure(figsize=(8,8))
    #设置图片的比例
    #生成边           
    coorinates=[[-1,1],[1,-1],[1,1],[-1,-1],[0,0]]
    #生成前5个点的坐标
    x,y =zip(*coorinates)
    newy=[each+6 for each in y]
    coorinates.extend(list(map(list, zip(x, newy))))
    newy=[each-6 for each in y]
    coorinates.extend(list(map(list, zip(x, newy))))
    newx=[each+6 for each in x]
    coorinates.extend(list(map(list, zip(newx, y))))
    newx=[each-6 for each in x]
    coorinates.extend(list(map(list, zip(newx, y))))
    #生成25个点的坐标
    x,y =zip(*coorinates)
    newy=[each+20 for each in y]
    newx=[each+20 for each in x]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newy=[each-20 for each in y]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newx=[each-20 for each in x]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newy=[each+20 for each in y]
    coorinates.extend(list(map(list, zip(newx, newy))))
    
    #生成125个点的坐标
    vnode =np.array(coorinates)
    npos =dict(zip(nodes,vnode))
    nlabels=dict(zip(nodes,nodes))
    #plt.cm.RdYlBu
    nx.draw_networkx_nodes(Um, npos, node_size=100, cmap=plt.cm.rainbow, node_color=y_predict, linewidths=0.3, alpha=0.8)
    nx.draw_networkx_edges(Um, npos, edge_list, width=1.5, edge_color='#BBBBBB', style="solid", alpha=0.3)
#     nx.draw_networkx_labels(Um, npos, font_size=12, font_color="#191A19", font_weight="bold")
    if save == True:
        filename = filepath + str(dataname) + '_ravasz.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
    

# 生成modify-ravasz
def generate_Um_k(k):
    '''
    k：从0开始
    
    '''
    edge_list=[]
    Um =nx.Graph()
#     clique_len = int((1-math.pow(4, k+1))/(-3)) 
    clique_len = 9
    node_len = int(clique_len*25)
    node_clique_len = int(clique_len*5)
#     print(node_len,node_clique_len,clique_len)
    nodes=list(range(1,node_len+1))
    Um.add_nodes_from(nodes)
    
    #图加入节点
    for i in range(0,node_clique_len):
        edge_listx=[(5+5*i,1+5*i),(5+5*i,2+5*i),(5+5*i,3+5*i),(5+5*i,4+5*i),(1+5*i,3+5*i),(1+5*i,4+5*i),(2+5*i,3+5*i),(2+5*i,4+5*i)] 
        edge_list=edge_list+edge_listx
#         print('edge_list:',i,edge_listx)
    
    # 加入每个clique的连边
    for j in range(0,clique_len): # j表示第j个clique
        for i in range(6,26):
            if((j*25+i)%5!=0):
                edge_list.extend([(5+25*j, j*25+i)])
#                 print('edges:',5+25*j, j*25+i)
    
    # 加入每一层的指向
    for j in range(0,clique_len): # j表示第j个clique
        for i in range(6,26):
            if((j*25+i)%5!=0):
                if j < 4:
                    clique_id = 0
                elif j%4 == 0:
                    clique_id = int(j/4-1)
                else:
                    clique_id = int((j-j%4)/4)
                edge_list.extend([(5+clique_id*25, j*25+i)])
#                 print('edges:',5+clique_id*25, j*25+i)
    remove_list = [(55, 66),(55, 67),(55, 68),(55, 69),(55, 61),(55, 62),(55, 63),(55, 64),(80, 86),(80, 87),(80, 88),(80, 89)]
    for n in remove_list:
        edge_list.remove(n)
    Um.add_edges_from(edge_list)
    y_true = []
    for i in range(0,225):
        y_true.append(i//25)
    print('Graph : node %d edges %d'%(Um.number_of_nodes(), Um.number_of_edges()))
    return Um,nodes,edge_list,y_true

def plot_Um_k(Um,nodes,edge_list,y_predict,filepath='./',dataname='LS_default',save=False):
    plt.figure(figsize=(8,8))
    
    #设置图片的比例
    #生成边        
#     coorinates=[[-2,2],[2,-2],[2,2],[-2,-2],[0,0]]
    coorinates=[[-2,2],[2,-2],[2,2],[-2,-2],[0,0]]
    
    #生成前5个点的坐标
    x,y =zip(*coorinates)
    newy=[each+10 for each in y]
    coorinates.extend(list(map(list, zip(x, newy))))
    newy=[each-10 for each in y]
    coorinates.extend(list(map(list, zip(x, newy))))
    newx=[each+10 for each in x]
    coorinates.extend(list(map(list, zip(newx, y))))
    newx=[each-10 for each in x]
    coorinates.extend(list(map(list, zip(newx, y))))
    print(coorinates)
    
    #生成25个点的坐标
    x,y =zip(*coorinates)
    newy=[each+40 for each in y]
    newx=[each+40 for each in x]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newy=[each-40 for each in y]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newx=[each-40 for each in x]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newy=[each+40 for each in y]
    coorinates.extend(list(map(list, zip(newx, newy))))
    
    addpoint_center = [[38, 42], [42, 38], [42, 42], [38, 38], [40, 40], 
                       [38, 52], [42, 48], [42, 52], [38, 48], [40, 50], 
                       [38, 32], [42, 28], [42, 32], [38, 28], [40, 30],
                       [48, 42], [52, 38], [52, 42], [48, 38], [50, 40], 
                       [28, 42], [32, 38], [32, 42], [28, 38], [30, 40]]
    x,y =zip(*addpoint_center)
    newy=[each+20 for each in y]
    newx=[each+20 for each in x]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newy=[each-20 for each in y]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newx=[each-20 for each in x]
    coorinates.extend(list(map(list, zip(newx, newy))))
    newy=[each+20 for each in y]
    coorinates.extend(list(map(list, zip(newx, newy))))
    
    #生成125个点的坐标
    vnode =np.array(coorinates)
    npos =dict(zip(nodes,vnode))
    nlabels=dict(zip(nodes,nodes))
    nx.draw_networkx_nodes(Um, npos, node_size=50, cmap=plt.cm.rainbow, node_color=y_predict, linewidths=0.3, alpha=0.8)
    nx.draw_networkx_edges(Um, npos, edge_list, width=1.5, edge_color='#BBBBBB', style="solid", alpha=0.3)
#     nx.draw_networkx_labels(Um, npos, font_size=12, font_color="#191A19", font_weight="bold")
    if save == True:
        filename = filepath + str(dataname) + '_modify_ravasz.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show()
    

# 生成lattice
def generate_nx_Lattice_circle(n):
    nodes = [i for i in range(0,n)]
    y_true = [1 for i in range(0,n)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = []
    for i in nodes:
        edges.append((i,(i+n+1)%n))
        edges.append((i,(i+n+2)%n))
    G.add_edges_from(edges)
    print('Graph : node %d edges %d'%(G.number_of_nodes(), G.number_of_edges()))
    return G,y_true

# 绘制lattice的结果图
def draw_nx_Lattice_graph_circle(G,y_predict,filepath='./',dataname='LS_default',save=False):
    pos = nx.shell_layout(G)  
    fig, ax = plt.subplots(figsize=(8,7))
    nx.draw_networkx_nodes(G, pos, node_size=300, cmap=plt.cm.RdYlBu, node_color=y_predict, linewidths=0.3, alpha=0.7)
#     nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#BBBBBB', style="solid",connectionstyle="angle3,rad=3", alpha=0.5)
#     nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.8')
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color="#191A19", font_weight="bold")  
    edges = list(G.edges())
    k = 0
    for edge in edges:
        if k not in [2,3,6]:
            ax.annotate("",
                        xy=pos[edge[0]], 
                        xytext=pos[edge[1]], 
                        size=20, va="center", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=0.6",
                                        linewidth=1
                                        ),
                        )
        
        else:
            ax.annotate("",
                        xy=pos[edge[1]], 
                        xytext=pos[edge[0]], 
                        size=20, va="center", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=0.6",
                                        linewidth=1
                                        ),
                        )
        k+=1

    
    plt.axis('off')
    plt.axis('equal')
    if save == True:
        filename = filepath + str(dataname) + '_lattice_circle.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show(G)
    
def generate_ER(n,p):
    G = nx.generators.random_graphs.erdos_renyi_graph(n,p)
    print('Graph : node %d edges %d '%(G.number_of_nodes(), G.number_of_edges()))
    nodelist = [n for n in G.nodes()]
    y_true = [1 for n in G.nodes()]
    return G,y_true

def draw_ER_fixed_position(G,y_true,filepath='./',dataname='LS_default',save=False):
    # draw graph with labels
    pos = nx.shell_layout(G)
    plt.figure(figsize=(8, 7))
    nx.draw_networkx_nodes(G, pos=pos, node_size=300, cmap=plt.cm.RdYlBu, node_color=y_true, linewidths=0.3,alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#BBBBBB', style="solid", alpha=0.3)
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color="#191A19", font_weight="bold")
    plt.axis('off')
    plt.axis('equal')
    if save == True:
        filename = filepath + str(dataname) + '_lattice_circle.pdf'
        plt.savefig(filename, bbox_inches='tight',dpi=300)
    plt.show(G)