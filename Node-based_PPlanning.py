'''
Path Planning with Node Based ACO (Only forward subprogram based on Jack's proposal)

TO-DOs:
1. General structure of algorithm -- done
    - node based executions have more power in node and links
2. Integrate a node_based.py module -- done
3. Go through the fold function; it seems like something isn't quite right -- pretty much okay
    - should be good enough, but double check. -- looks good
    - once okay, clean up code
        - maybe make ants_generation_and_activity_cycle perform the shooting of ants the for one cycle
'''
import networkx as nx
import random
import sys
from node_based import initialize_graph, Node, Link

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
nodes = 7 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 5 # number of edges to remove
max_cycles = 1 # total number of cycles

def main():
    ant_graph = initialize_graph(yaml_file=None,space=euc_space,size=nodes,
    num_of_nodes_to_remove=rand_nodes,num_of_edges_to_remove=rand_edges)
    print(ant_graph.edges(data=True))
    j = 1
    print('after node execution',j,':')
    for i in range(100):
        ant_graph.ants_generation_and_activity_cycle()
        j += 1
        print('after node execution',j,':')

if __name__ == '__main__':
    main()
