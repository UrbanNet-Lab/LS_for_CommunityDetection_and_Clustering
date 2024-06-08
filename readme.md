

# Local dominance unveils clusters in networks

Codes developed in our paper "Local dominance unveils clusters in networks, Communications Physics, 2024, 7: 170" [PDF: https://rdcu.be/dJxY0] are stored here. Our "Local Search" algorithm is in **LS_algorithm.py**. And we prepared related visualization and analysis methods in other .py files. 

## Abstract

<p align="justify">
Clusters or communities can provide a coarse-grained description of complex systems at multiple scales, but their detection remains challenging in practice. Community detection methods often define communities as dense subgraphs, or subgraphs with few connections in-between, via concepts such as the cut, conductance, or modularity. Here we consider another perspective built on the notion of local dominance, where low-degree nodes are assigned to the basin of influence of high degree nodes, and design an efficient algorithm based on local information. Local dominance gives rises to community centers, and uncovers local hierarchies in the network. Community centers have a larger degree than their neighbors and are sufficiently distant from other centers. The strength of our framework is demonstrated on synthesized and empirical networks with ground-truth community labels. The notion of local dominance and the associated asymmetric relations between nodes are not restricted to community detection, and can be utilised in clustering problems, as we illustrate on networks derived from vector data.</p>
<img src="fig\abstract.png" alt="abstract" style="zoom:100%;" />

## Requirements

The codebase was implemented in Python 3.9.1. The version of packages used in the codebase are listed below.

```
networkx          2.7
numpy             1.19.5
pandas            1.4.2
scipy             1.8
matplotlib        3.4.3
python-louvain    0.16
```

## Datasets

In our paper, the LS algorithm is tested on typical networks with ground-truth community labels (e.g., Zachary Karate Club, Football, Polbooks, Polblogs, Cora, Citeseers, PubMed) and networks constructed from vector data (which includes typical 2D benchmark vector data: Flame, Spiral, Aggregation, R15, Blobs, Circles, and Moons; High-dimensional vector data: Iris, Wine, MINST, and Olivetti).

## Usage

If you want to use this algorithm for your data, you should format it as a graph type **G** with Networkx in Python, then use the following command:

```
python
>>>from LS_algorithm import hierarchical_degree_communities
>>>hierarchical_degree_communities(G, maximum_tree=True, seed=seed)
```

If you want to set the number of communities to explore the multi-scale community structure, which also can be common in real networks, then you can specify the number of communities at the second input parameter, the upper limit of which equals the number of local leaders (see Fig.3b,e in the main text):

```
python
>>>from LS_algorithm import hierarchical_degree_communities
>>>hierarchical_degree_communities(G, center_num=leaders_num, auto_choose_centers=False, maximum_tree=True, seed=seed)
```

## Example

<p float="left">
<img src="fig\example1.png" alt="example1"  width="500" />

<img src="fig\example3.png" alt="example3"  width="500" height="475" />
</p>

For a multi-scale network, there is a notable gap between first four local leaders and the other twelve ones in the decision graph (see above), setting the number of communities (i.e., the parameter "leaders_num") as four yields the first level partition that comprise four large communities:

```
>>>hierarchical_degree_communities(MSG, 4, auto_choose_centers=False, maximum_tree=True, seed=seed)
```

If specifying the number of communities as sixteen, the obtained second level parititon comprises sixteen small communities:

```
>>>hierarchical_degree_communities(MSG, 16, auto_choose_centers=False, maximum_tree=True, seed=seed)
```


## A Quick Run

To better illustrate the process of clustering vector data, we use the challenging vector data presented in paper as an example. The network **G** feed to our LS algorithm is constructed via $\epsilon$-ball method with help from percolation theory in complex networks (via calling subroutine LS_cluster_function.py), which determines the distance threshold for $\epsilon$. Then, we can call LS algorithm to obtain the decision graph and corresponding clustering results (with centers highlighted as stars); If you want to explore the multi-scale community structure, the number of communities can be hinted in the decision graph (the upper limit of the number of communities is no greater than the number of local leaders). 

For a quick run:  

```
python Example.py
```

you will get:

```
====Local Search Algorithm (random seed 1)==========
Network: 35 nodes,50 edges
The number of local leaders: 5
Running Time: 1 ms
The number of community  centers: 2
The id of the centers are: [30, 18]
Modularity of the partition by LS: 0.4352
The decision graph for determining the number of centers (centers are nodes with both a large degree k_i and path length l_i; 
there might be multi-scale community structure, which is signified by notable gaps):
Note: The number of communities can be explicitly set according to the results in decision graph, 
if multi-scale community structure is of interests
```

<img src="fig\quick_run.png" alt="quick_run"  width="500" />

```
The original vector data (stars correspond to identified centers):
```

<img src="fig\quick_run1.png" alt="quick_run1"  width="500" />



