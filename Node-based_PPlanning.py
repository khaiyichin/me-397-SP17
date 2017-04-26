'''
Path Planning with Node Based ACO (Only forward subprogram based on Jack's proposal)

TO-DOs:
1. General structure of algorithm -- done
    - node based executions have more power in node and links
2. Integrate a node_based.py module -- done
3. Go through the fold function; it seems like something isn't quite right
    - should be good enough, but double check.
    - once okay, clean up code
'''
import networkx as nx
import random
import sys
from node_based import initialize_graph, Node, Link

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
nodes = 4 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 0 # number of edges to remove
max_cycles = 1 # total number of cycles

def main():
    ant_graph = initialize_graph(yaml_file=None,space=euc_space,size=nodes,
    num_of_nodes_to_remove=rand_nodes,num_of_edges_to_remove=rand_edges)
    print(ant_graph.edges(data=True))
    print([(i.node_name,i.ant_mass,i.avg_distance) for i in ant_graph.nodes])
    print('node execution')
    for i in range(3):
        ant_graph.ants_generation_and_activity_cycle()
        print([(i.node_name,i.ant_mass,i.avg_distance) for i in ant_graph.nodes])
        print('node execution')

if __name__ == '__main__':
    main()
