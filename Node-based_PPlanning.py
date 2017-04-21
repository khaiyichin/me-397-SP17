'''
Path Planning with Node Based ACO (Only forward subprogram based on Jack's proposal)

TO-DOs:
1. General structure of algorithm
    - node based executions have more power in node and links
2. Integrate a node_based.py module
3.
'''
import networkx as nx
import random
import sys
from node_based import initialize_graph, Node, Link

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
nodes = 5 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 4 # number of edges to remove
max_cycles = 1 # total number of cycles

def main():
    ant_graph = initialize_graph(yaml_file=None,space=euc_space,size=nodes,
    num_of_nodes_to_remove=rand_nodes,num_of_edges_to_remove=rand_edges)
    print(ant_graph.nodes[2].node_name)
    for i in ant_graph.links:
        i.ant_mass_ascend = 10
        i.ant_mass_descend = 27
        i.link_function()
    print(ant_graph.nodes[2].fold_function())
    ant_graph.nodes[2].split_fold_function()

if __name__ == '__main__':
    main()
