import networkx as nx
import sys
from matplotlib import pyplot as plt
import numpy as np
import random


class Ant(object):
    def __init__(self,starting_node_object,phero=10):
        self.memory = [starting_node_object.node_name()]
        self.current_state = starting_node_object.node_name()
        self.map = starting_node_object[1] # Node() class tuple, which the 2nd element is the networkx graph object
        self.first_state = starting_node_object.node_name()
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
        destination_node = self.map.nodes()[-1] # the final node
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

class Ant_Graph(nx.Graph):
    def __new__(cls,networkx_graph_object):
        return nx.Graph.__new__(cls)

    def evaporate(self,evaporation_rate=0.5):
        self.leftover = 1 - evaporation_rate
        for each in self.edges_iter():
            self.edge[each[0]][each[1]]['phero'] *= self.leftover

def initialize_graph(name=None,yaml_file=None,space=0,size=0,
    init_phero=0.001):
    if yaml_file is None:
        if (size > (space+1)**2):
            print('The allowed space is too small for the number of nodes.')
            sys.exit()

        int_coords = []

        for i in range(size):
            x = random.randrange(0,space+1)
            y = random.randrange(0,space+1)

            while ((x,y) in int_coords):
                x = random.randrange(0,space+1)
                y = random.randrange(0,space+1)

            else:
                int_coords.append((x,y))

        # Initialize dictionary with coordinates
        pos = dict(zip(range(1,size+1),int_coords))

        temp = nx.complete_graph(size+1)
        temp.remove_node(0) # make it easier/convenient to count
        temp.remove_edge(1,size) # removing the direct link that connects start and end node

        for n,p in pos.items():
            temp.node[n]['x'] = p[0]
            temp.node[n]['y'] = p[1]

        for i in temp.edges_iter():
            distance = distance_calc(temp.node[i[0]],temp.node[i[1]])
            temp.add_edge(i[0],i[1],phero=init_phero,dist=distance)

        if name is None:
            filename = '('+str(size)+'-'+')nodes.yaml'

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

    return graph


class Node(tuple):
    def __new__(cls,node_name,networkx_graph):
        return tuple.__new__(cls,(node_name,networkx_graph)) # the static method __new__ creates and return a new instance of a class from its first argument

    def node_name(self):
        return self[0]

def distance_calc(point1,point2):
    x1 = point1['x']
    x2 = point2['x']
    y1 = point1['y']
    y2 = point2['y']

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def rand_node_remover(graph,num_of_nodes_to_remove):
    for i in range(num_of_nodes_to_remove):
        lucky_node = random.randrange(2,len(graph.nodes())) # lucky_node will never be the last node (i.e. = size)
        # lucky_node will never be the starting or end nodes
        while (lucky_node not in graph.nodes()):
            lucky_node = random.randrange(2,len(graph.nodes()))

        graph.remove_node(lucky_node)
