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
    - removal of edges ==> look at subroutine for completeness of graph -- done
    - continue separating application vs library code
    - prohibit u-turn of ants -- done
'''
import networkx as nx
import random
import sys
from signal_based import initialize_graph, Ant, Node

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
num_of_ants = 3 # total number of ants
nodes = 7 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 4 # number of edges to remove
max_cycles = 20 # total number of cycles

def ACO_metaheuristic():
    # Initialize NetworkX Graph object
    ant_graph = initialize_graph(yaml_file=None,space=euc_space,size=nodes,
    num_of_nodes_to_remove=rand_nodes,num_of_edges_to_remove=rand_edges)

    # Initialize lists to store data for each cycle
    shortest_each_cycle = []

    immutable_dict = {i:j for i,j in ant_graph.ant_counter_per_edge.items()} # exactly the same things, but immutable
    num_of_ants_per_edge_each_cycle = [immutable_dict] # this list updates ALL the elements every cycle if it appends the object attribute

    pheromone_levels_each_cycle =[{(i,j):k for i,j,k in ant_graph.edges(data='phero')}] # creating a dict for the edges; .edges() returns a list

    current_cycle = 0

    # Run until desired number of cycles reached
    while (current_cycle != max_cycles):
        shortest_for_now = ants_generation_and_activity_cycle(ant_graph)
        shortest_each_cycle.append(shortest_for_now)

        immutable_dict = {i:j for i,j in ant_graph.ant_counter_per_edge.items()}
        num_of_ants_per_edge_each_cycle.append(immutable_dict)

        pheromone_levels_each_cycle.append({(i,j):k for i,j,k in ant_graph.edges(data='phero')})

        ant_graph.evaporate()
        ant_graph.initialize()
        current_cycle += 1

    # Process data for shortes route
    dist_and_route = list(zip(*shortest_each_cycle)) # asterisk means unpacking arguments from a list
    distances = dist_and_route[0]

    optimal_dist = min(distances)
    ind = dist_and_route[0].index(optimal_dist)
    optimal_route = shortest_each_cycle[ind][1]

    print("Shortest tour consists of these cities:",optimal_route,
    "with a distance of",optimal_dist)
    print("Number of cycles ran =",current_cycle)

    # Process data for number of ants
    print("Number of ants on each edges on the last cycle:",
    num_of_ants_per_edge_each_cycle[-1])
    print("The shortest tour made on the last cycle:",shortest_for_now[1])

    # Process data for pheromone levels
    print("The pheromone levels on the last cycle:",pheromone_levels_each_cycle[-1])

def ants_generation_and_activity_cycle(ant_graph):
    # Initialize list to store ant objects
    travelled_ants = [];

    # Shoot ants through graph one at a time
    for ant in range(num_of_ants):
        travelled_ants.append(Ant(Node(1,ant_graph)))
        travelled_ants[-1].cycle()

        for i in ant_graph.ant_counter_per_edge:
            ant_graph.ant_counter_per_edge[i] += travelled_ants[-1].travelled_edges.count(i)

    shortest = 100000000

    # Make ants trace back routes and lay pheromones
    for each_ant in travelled_ants:
        if (shortest > each_ant.travelled):
            shortest = each_ant.travelled
            result = shortest,each_ant.memory
        each_ant.lay_pheromones()

    return result

def main():
    ACO_metaheuristic()

if __name__ == '__main__':
    main()
