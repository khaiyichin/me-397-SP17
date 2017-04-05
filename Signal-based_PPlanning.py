'''
Path Planning with ACO

TO-DOs:
6. the main question: what kind of information to collect (and conditions to run in)
    to relate ant number, graph size, convergence rate etc

NEW TO-DOs
1. write function to generate random graph
    - assign random coordinates 2-D/3-D -- done
    - include obstacle (node removal) function within, but optional to run -- done
    - include edge removal
2. write function to save into yaml format -- done
    - allow initialization of fresh graph or construct from yaml file -- done
3. consider setting constraints for start and end node
    - maybe set them the furthest apart
    - remove the direct edge between start and end node -- done
    - prohibit removal of them -- done
4. add distances and pheromones on edges -- done
5. make sure ant movement is correct -- done
    - start at the same node -- done
6. record convergence of ants
    - not interested in optimal solution

UPDATES:
    - yaml file name needs changing
    - removal of edges ==> look at subroutine for completeness of graph
    - continue separating application vs library code
'''
import networkx as nx
import sys
from signal_based import initialize_graph, Ant

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
num_of_ants = 10 # total number of ants
nodes = 5 # size of graph (number of nodes)
euc_space = 5 # euclidean space in each dimension (max x,y,z coordinates)
# dim = 2 # dimension of graph (either 2 or 3?)
rand_obstacles = 3 # number of nodes to remove
# decay_rate = 0.5 # trail evaporation
# trail_persistence_rate = 1 - decay_rate # trail persistence
max_cycles = 5 # total number of cycles

class Node(tuple):
    def __new__(cls,node_name,networkx_graph):
        return tuple.__new__(cls,(node_name,networkx_graph)) # the static method __new__ creates and return a new instance of a class from its first argument

    def node_name(self):
        return self[0]

def ACO_metaheuristic():
    list_of_shortest_each_cycle = []
    current_cycle = 0

    # Initialize NetworkX Graph object
    ant_graph = initialize_graph(yaml_file='(5-3)nodes.yaml',space=euc_space,size=nodes)

    while (current_cycle != max_cycles):
        shortest_for_now = ants_generation_and_activity_cycle(ant_graph)
        list_of_shortest_each_cycle.append(shortest_for_now)

        ant_graph.evaporate()
        current_cycle += 1

    dist_and_route = list(zip(*list_of_shortest_each_cycle)) # asterisk means unpacking arguments from a list
    distances = dist_and_route[0]

    optimal_dist = min(distances)
    ind = dist_and_route[0].index(optimal_dist)
    optimal_route = list_of_shortest_each_cycle[ind][1]

    print("Shortest tour consists of these cities:",optimal_route,
    "with a distance of",optimal_dist)
    print("Number of cycles ran =",current_cycle)
    sys.exit()

def ants_generation_and_activity_cycle(ant_graph):
    # Initialize list to store ant objects
    travelled_ants = [];

    for ant in range(num_of_ants):
        travelled_ants.append(Ant(Node(1,ant_graph)))
        travelled_ants[-1].cycle()

    shortest = 100000

    for each_ant in travelled_ants:
        if (shortest > each_ant.travelled):
            shortest = each_ant.travelled
            result = shortest,each_ant.memory
        each_ant.lay_pheromones()

    print(ant_graph.edges(data=True))

    return result

def main():
    ACO_metaheuristic()

if __name__ == '__main__':
    main()
