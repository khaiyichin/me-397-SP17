'''
Module for node based ant colony optimization method

1. should be able to build off the networkx objects

Required classes:
1. Node() object
    - contains n, d_f
    - fold function
    - split function
    - split_fold function
2. ant mass continuum object -- x or y (depending on context) in Jack's paper
    - contains n, d_f
3. Link() object -- w in Jack's paper
    - contains n, d_l, p, d_f (mostly counters or traces of travels)
    - link function

'''

import networkx as nx
import sys
from matplotlib import pyplot as plt
import numpy as np
import random

memory_param = 0.5

class Node(tuple):
    def __new__(cls,node_name,networkx_graph):
        return tuple.__new__(cls,(node_name,networkx_graph)) # the static method __new__ creates and return a new instance of a class from its first argument

    def initialize(self):
        self.node_name = self[0]
        self.graph = self[1]
        self.in_links = [(i,self.node_name) for i in self[1].neighbors(self.node_name)]
        self.out_links = [(self.node_name,i) for i in self[1].neighbors(self.node_name)]
        self.ant_mass = 0

    def add_ant_mass(self,ant_mass):
        self.ant_mass += ant_mass

    # def retain_ant_mass(self): # For the end node
    def random(self):
        x = (1,3)
        print(self.graph.link[i].ant_mass )
    # ADDRESS THIS THE FIRST THING YOU DO! still need to properly obtain the ant_mass; trouble referring to the right index in the ant_graph.links list
    # def fold(self):
    #     list_of_link_tuples = [i[0] for i in self.graph.links]
    #     for i in self.in_links:
    #         if i[0] > i[1]:
    #             i = sorted(i[0],i[1])
    #             self.ant_mass += self.graph.links[i].ant_mass_descend
    #         else:
    #             self.ant_mass += self.graph.links[i].ant_mass_ascend

        for i in self.in_links:
            if i[0] > i[1]:
                self.avg_distance += self.graph.links[i].avg_distance_descend

            else:
                self.avg_distance += self.graph.links[i].avg_distance_descend

        in_distance = [i.average_distance for i in self]
        self.ant_mass = sum(in_ant_mass)
        self.avg_distance = np.multiply(in_distance,in_ant_mass)/self.ant_mass

    def split_fold(self):
        out_phero = [i.phero for i in self.out_links]
        self.total_out_phero = sum(out_phero)

    def split(self,target_link):
        out_distance = self.average_distance
        out_ant_mass = target_link.phero*self.ant_mass/self.total_out_phero
        return out_ant_mass

class Link(tuple):
    def __new__(cls,link_name,networkx_graph):
        return tuple.__new__(cls,(link_name,networkx_graph))

    def initialize(self):
        self.link_name = self[0] # a tuple containing the head and tail node connected
        self.graph = self[1]
        self.link_distance = self[1].edge[self.link_name[0]][self.link_name[1]]['dist'] # d_l
        self.phero = self[1].edge[self.link_name[0]][self.link_name[1]]['phero'] # p
        self.ant_mass_ascend = 0 # n
        self.ant_mass_descend = 0 # n
        self.avg_distance_ascend = 0 # d_f from low to high (eg. 1-2)
        self.avg_distance_descend = 0 # d_f from high to low (eg. 2-1)

    def link_ascend(signal_avg_distance):
        self.avg_distance_ascend = signal_avg_distance + self.link_distance

    def link_descend(signal_avg_distance):
        self.avg_distance_descend = signal_avg_distance + self.link_distance

class Ant_Graph(nx.Graph):
    def __new__(cls,networkx_graph_object):
        return nx.Graph.__new__(cls)

    def initialize(self,evaporation_rate=0.5):
        print('hey')
        self.nodes = [Node(i,self) for i in self.nodes()]
        for i in self.nodes:
            i.initialize()

        self.links = [Link(i,self) for i in self.edges()]
        for i in self.links:
            i.initialize()

        self.leftover = 1 - evaporation_rate

    # def node_execution(self):
        # fire nodes first (so that signal is initiated/launched) then activate link function
        # which then continues the signal to the next node to the fold function
        # look at all nodes

    def evaporate(self):
        for each in self.edges_iter():
            self.edge[each[0]][each[1]]['phero'] *= self.leftover

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

def initialize_graph(name=None,yaml_file=None,space=0,size=0,
    init_phero=0.001,num_of_nodes_to_remove=0,num_of_edges_to_remove=0):
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

        points = []
        for n,p in pos.items():
            points.append(Point(p[0],p[1]))

        for i in temp.edges_iter():
            distance = distance_calc(points[i[0]-1],points[i[1]-1]) # the indices for the 'points' list start from 0, whereas i (node numbers) start from 1
            temp.add_edge(i[0],i[1],phero=init_phero,dist=distance)

        # Remove nodes and edges to create obstacles
        temp = rand_node_remover(temp,num_of_nodes_to_remove)
        temp = rand_edge_remover(temp,num_of_edges_to_remove)

        if name is None:
            filename = str(size-num_of_nodes_to_remove)+'nodes'+ \
                str(len(temp.edges()))+'edges.yaml'

        else:
            filename = name+'.yaml'

        nx.write_yaml(temp,filename)
        graph = Ant_Graph(temp)

        # Create plot
        fig = plt.figure()
        axes = fig.gca()
        axes.set_xticks(np.arange(0,space+1,1))
        axes.set_yticks(np.arange(0,space+1,1))

        nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w')
        plt.savefig(filename[:-5],format='PNG')

    else:
        temp = nx.read_yaml(yaml_file)
        graph = Ant_Graph(temp)

    graph.initialize()

    return graph

def distance_calc(point1,point2):
    x1 = point1.x
    x2 = point2.x
    y1 = point1.y
    y2 = point2.y

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def rand_node_remover(graph,num_of_nodes_to_remove):
    for i in range(num_of_nodes_to_remove):
        lucky_node = random.randrange(2,len(graph.nodes())) # lucky_node will never be the last node (i.e. = size)
        # lucky_node will never be the starting or end nodes
        while (lucky_node not in graph.nodes()):
            lucky_node = random.randrange(2,len(graph.nodes()))

        graph.remove_node(lucky_node)

    return graph

def rand_edge_remover(graph,num_of_edges_to_remove):
    if (num_of_edges_to_remove >= len(graph.nodes())-2):
        print('Warning: The edge removal might not find a solution.')

    connected_condition = False # the while loop should only end when the removed edges doesn't disconnect the graph

    while (connected_condition == False):
        candidate_graph = graph.copy()

        for i in range(num_of_edges_to_remove):
            lucky_edge = random.choice(candidate_graph.edges())

            candidate_graph.remove_edge(lucky_edge[0],lucky_edge[1])

        connected_condition = nx.is_connected(candidate_graph)

    return candidate_graph
