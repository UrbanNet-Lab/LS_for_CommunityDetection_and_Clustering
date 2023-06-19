from LS_algorithm import hierarchical_degree_communities
from LS_cluster_function import caldistance, chose_dc, cal_adge, evaluate_network, plot_fig, plot_louvian_fig
from LS_other_function import plot_combination, draw_graph, plot_degree_shortpath, plot_multi_log
import networkx as nx
import pandas as pd
import datetime
import community
import numpy as np


test_cdp = r'./data/2d_datasets/test_cdp.txt'
raw_data = np.loadtxt(test_cdp, delimiter=' ', usecols=[0, 1])
dc_percent = 6 # the threshold is calculated in advance by the function cal_jumppoint in the subroutine LS_cluster_function.py
leaders_num = 2 # the threshold is calculated in advance in the subroutine LS_algorithm.py
x = raw_data[:, 0]
y = raw_data[:, 1]
nodes = [i for i in range(len(raw_data))]
distance = caldistance(raw_data, 1)
dc = chose_dc(distance, dc_percent)
start, end = cal_adge(distance, dc)
df = pd.DataFrame({'from': start, 'to': end})
G = nx.from_pandas_edgelist(df, source='from', target='to')
G.add_nodes_from(nodes)
seed = 1
D, centers, y_ls, y_ls_partition, plot_combination_data = hierarchical_degree_communities(G, leaders_num,
                                                                                                  auto_choose_centers=False,
                                                                                                  maximum_tree=True,
                                                                                                  seed=seed)
y_ls_plot, partition_ls_plot = evaluate_network(y_ls_partition, len(raw_data))
print("The original vector data (stars correspond to identified centers):")
plot_fig(x, y, y_ls_plot, centers)
subplot_location = [0.25,0.55,0.35,0.3]
xlim_start_end = [0.3,0.7]
ylim_start_end = [0.7,0.3]
font_location = -0.04