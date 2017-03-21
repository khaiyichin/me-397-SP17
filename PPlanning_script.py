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
1. develop 5x5 graph -- done
2. create obstacle function -- done
3. create ant objects -- done; movement seems right, but might wanna go through
    some tests to make sure they run okay
4. create tuple class instantiates and reference to networkx object
    don't want to access networkx object globally -- done

5. move ants?

'''
import networkx as nx
import sys
from matplotlib import pyplot as plt
import numpy as np
import random

Q = 1 # pheromone constant
R = 10 # total number of ants
square_grid_number = 3 # size of graph
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
    def __new__(cls,graph_object):
        return nx.Graph.__new__(cls)

    def evaporate(self):
        trail_persistence_rate = rho
        for each in self.edges_iter():
            self.edge[each[0]][each[1]]['phero'] *= trail_persistence_rate

def add_straight_obstacle(ant_graph,type,size,start_node): # goes from bottom (start_node) to top, left to right (horizontal only)
    num_of_nodes = len(ant_graph.nodes())
    row_col_length = np.sqrt(num_of_nodes)
    if type == 'vertical':
        nodes_to_remove = [start_node + row_col_length*i for i in range(size)]
        ant_graph.remove_nodes_from(nodes_to_remove)
    elif type == 'horizontal':
        nodes_to_remove = [start_node + i for i in range(size)]
        ant_graph.remove_nodes_from(nodes_to_remove)
    elif type == 'diagonal/':
        nodes_to_remove = [start_node + row_col_length*i + 1 for i in range(size)]
        ant_graph.remove_nodes_from(nodes_to_remove)
    elif type == 'diagonal"\"':
        nodes_to_remove = [start_node + row_col_length*i - 1 for i in range(size)]
        ant_graph.remove_nodes_from(nodes_to_remove)
    else:
        print("Wrong type of straight obstacle")

    return

def initialize_graph():

    int_coords = []
    for i in range(square_grid_number):
        for j in range(square_grid_number):
            int_coords.append([j,i])

    num_of_nodes = len(int_coords)
    init_phero = 0.001

    # Initialize dictionary with coordinates
    pos = dict(zip(range(1,num_of_nodes+1),int_coords))

    graph = Ant_Graph(nx.Graph())
    graph.add_nodes_from(pos.keys())

    for n,p in pos.items():
        graph.node[n]['name'] = n
        graph.node[n]['x'] = p[0]
        graph.node[n]['y'] = p[1]

    corner = [1,square_grid_number,(square_grid_number**2)-square_grid_number+1,square_grid_number**2]
    for i in range(1,num_of_nodes+1):
        list_of_neighbors = []

        if (i%square_grid_number != 0 and i%square_grid_number != 1 and
        i/square_grid_number < square_grid_number-1 and int(i/square_grid_number) > 0):
            list_of_neighbors.append(i-square_grid_number-1)    # SW
            list_of_neighbors.append(i-square_grid_number)      # S
            list_of_neighbors.append(i-square_grid_number+1)    # SE
            list_of_neighbors.append(i-1)                       # W
            list_of_neighbors.append(i+1)                       # E
            list_of_neighbors.append(i+square_grid_number-1)    # NW
            list_of_neighbors.append(i+square_grid_number)      # N
            list_of_neighbors.append(i+square_grid_number+1)    # NE
        elif i not in corner:
            if i%square_grid_number == 0: # right boundary
                list_of_neighbors.append(i-square_grid_number-1)# SW
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number-1)# NW
            elif i%square_grid_number == 1: # left boundary
                list_of_neighbors.append(i-square_grid_number+1)# SE
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number+1)# NE
            elif i/square_grid_number > square_grid_number-1: # upper boundary
                list_of_neighbors.append(i-square_grid_number-1)# SW
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-square_grid_number+1)# SE
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+1)                   # E
            else: # lower boundary
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i+square_grid_number-1)# NW
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number+1)# NE
        else: # corners
            if i==corner[0]: # bottom left
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number+1)# NE
            elif i==corner[1]: # bottom right
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number-1)# NW
            elif i==corner[2]: # upper left
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-square_grid_number+1)# SE
            else: # upper right
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-square_grid_number-1)# SW

        for j in list_of_neighbors:
            distance = distance_calc(graph.node[i],graph.node[j])
            graph.add_edge(i,j,phero=init_phero,dist=distance)

    # Create plot with gridlines
    fig = plt.figure()
    axes = fig.gca()
    axes.set_xticks(np.arange(0,6,1))
    axes.set_yticks(np.arange(0,6,1))

    # Draw network on figure
    # nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w')
    # plt.grid('on')
    # plt.show()

    add_straight_obstacle(graph,'vertical',1,6)
    nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w')
    plt.show()
    return graph

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
