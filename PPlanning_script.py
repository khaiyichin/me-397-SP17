'''
Path Planning with ACO
5x5 node graph

Rough graph representation:
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O

where the lines (-,|,X) are the connecting links and the circles (O) are the nodes.

TO-DOs:
6. the main question: what kind of information to collect (and conditions to run in)
    to relate ant number, graph size, convergence rate etc

NEW TO-DOs
1. write function to generate random graph
    - assign random coordinates 2-D/3-D -- done
    - include obstacle (node removal) function within, but optional to run -- done
2. write function to save into yaml format -- done
    - allow initialization of fresh graph or construct from yaml file -- done
3. consider setting constraints for start and end node
    - maybe set them the furthest apart
    - prohibit removal of them
4. add distances and pheromones on edges -- done
5. make sure ant movement is correct
'''
import networkx as nx
import sys
from matplotlib import pyplot as plt
import numpy as np
import random

init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 1 # pheromone constant
R = 10 # total number of ants
size = 10 # size of graph (number of nodes)
space = 10 # euclidean space in each dimension (max x,y,z coordinates)
dim = 2 # dimension of graph (either 2 or 3?)
rand_obstacles = 3 # number of nodes to remove
decay_rate = 0.5 # trail evaporation
rho = 1 - decay_rate # trail persistence
max_cycles = 10 # total number of cycles

class Ant(object):
    def __init__(self,starting_node,phero=Q):
        self.memory = [starting_node.node_name()]
        self.current_state = starting_node.node_name()
        self.map = starting_node[1] # Node() class tuple, which the 2nd element is the networkx graph object
        self.first_state = starting_node.node_name()
        self.travelled = 0
        self.phero = phero
        self.routing_table = []

    def generate_routing_table(self):
        alpha = 2
        beta = 0

        avail_options = [i for i in self.map.neighbors(self.current_state)]

        list_of_probs = []
        denom = 0

        for option in avail_options:
            edge_phero = self.map.edge[self.current_state][option]['phero']
            edge_vis = 1/(self.map.edge[self.current_state][option]['dist'])
            prob = (edge_phero**alpha)*(edge_vis**beta)
            denom += prob
            list_of_probs.append(prob)

        for each in list_of_probs:
            each = each/denom
            self.routing_table.append(each) #  WIPE OUT ROUTING TABLE AFTER EACH MOVE

        return avail_options # list of options to move to

    def choices(self):
        target_cities = self.generate_routing_table()
        choice = random.choices(target_cities,weights=self.routing_table)
        choice = choice[0]
        return choice

    def move_to_next_state(self):
        # invoke choices which invokes routing table
        next_node = self.choices()
        self.memory.append(next_node)
        self.travelled += self.map.edge[self.current_state][next_node]['dist']
        self.current_state = next_node

        self.routing_table = []

    def cycle(self): # each cycle
        destination_node = square_grid_number**2
        while self.current_state != destination_node:
            self.move_to_next_state()

    def lay_pheromones(self):
        self.phero_per_unit = self.phero/self.travelled
        backwards_mem = list(reversed(self.memory))

        # Lay pheromone on the networkx graph object
        for step in range(len(backwards_mem)-1):
            head_node = backwards_mem[step]
            tail_node = backwards_mem[step+1]
            distance = self.map.edge[head_node][tail_node]['dist']
            self.map.edge[head_node][tail_node]['phero'] += self.phero_per_unit*distance

class Node(tuple):
    def __new__(cls,node_name,networkx_graph):
        return tuple.__new__(cls,(node_name,networkx_graph)) # the static method __new__ creates and return a new instance of a class from its first argument

    def node_name(self):
        return self[0]

class Ant_Graph(nx.Graph):
    def __new__(cls,networkx_graph_object):
        return nx.Graph.__new__(cls)

    def evaporate(self):
        trail_persistence_rate = rho
        for each in self.edges_iter():
            self.edge[each[0]][each[1]]['phero'] *= trail_persistence_rate

def initialize_graph(name=None,yaml_file=None):
    if yaml_file is None:
        int_coords = []

        for i in range(size):
            x = random.randrange(0,space+1)
            y = random.randrange(0,space+1)

            if dim == 3:
                z = random.randrange(0,space+1)
                int_coords.append((x,y,z))

            else:
                int_coords.append((x,y))

        # Initialize dictionary with coordinates
        pos = dict(zip(range(1,size+1),int_coords))

        temp = nx.complete_graph(size+1)
        temp.remove_node(0) # make it easier/convenient to count

        if dim == 3:
            for n,p in pos.items():
                temp.node[n]['name'] = n
                temp.node[n]['x'] = p[0]
                temp.node[n]['y'] = p[1]
                temp.node[n]['z'] = p[2]

        else:
            for n,p in pos.items():
                temp.node[n]['name'] = n
                temp.node[n]['x'] = p[0]
                temp.node[n]['y'] = p[1]

        rand_node_remover(temp,rand_obstacles) # removing some nodes
        for i in temp.edges_iter():
            distance = distance_calc(temp.node[i[0]],temp.node[i[1]])
            temp.add_edge(i[0],i[1],phero=init_phero,dist=distance)

        if name is None:
            filename = '('+str(size)+'-'+str(rand_obstacles)+')nodes.yaml'

        else:
            filename = name+'.yaml'

        nx.write_yaml(temp,filename)
        graph = Ant_Graph(temp)

    else:
        temp = nx.read_yaml(yaml_file)
        graph = Ant_Graph(temp)

    # Create plot
    fig = plt.figure()
    axes = fig.gca()
    axes.set_xticks(np.arange(0,space+1,1))
    axes.set_yticks(np.arange(0,space+1,1))

    # add_straight_obstacle(graph,'vertical',1,6)
    positions = dict((i,(j['x'],j['y'])) for i,j in graph.nodes_iter(data=True))
    nx.draw_networkx(graph,pos=positions,node_size=175,font_size=9,node_color='w')
    plt.show()

    sys.exit()

    return graph

def rand_node_remover(graph,num_of_nodes_to_remove):
    for i in range(num_of_nodes_to_remove):
        lucky_node = random.randrange(1,len(graph.nodes()))

        while (lucky_node not in graph.nodes()):
            lucky_node = random.randrange(1,len(graph.nodes()))

        graph.remove_node(lucky_node)

def distance_calc(point1,point2):
    x1 = point1['x']
    x2 = point2['x']
    y1 = point1['y']
    y2 = point2['y']

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def ACO_metaheuristic():
    list_of_shortest_each_cycle = []
    current_cycle = 0

    # Initialize NetworkX Graph object
    ant_graph = initialize_graph()

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

def ants_generation_and_activity_cycle(ant_graph):
    # Initialize list to store ant objects
    travelled_ants = [];

    num_of_ants = R

    for ant in range(num_of_ants):
        travelled_ants.append(Ant(Node(1,ant_graph)))
        travelled_ants[-1].cycle()

    shortest = 100000

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
