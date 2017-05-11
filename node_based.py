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
        self.links = []
        self.links_ascend = [] # for current node n, links are of the form (a,n), a < n
        self.links_descend = [] # for current node n, links are of the form (b,n), b > n
        self.ant_mass = 0
        self.avg_distance = 0
        self.out_link_phero = 0

        connected_links = [(i,self.node_name) for i in self[1].neighbors(self.node_name)]
        for i in connected_links:
            index = sorted([i[0],i[1]])
            index = (index[0],index[1])

            link_object = [link_tuple for link_tuple in self.graph.links_list
            if link_tuple[:][0] == index]

            link_object = link_object[0] # to get it out of the list
            self.links.append(link_object)

            if i[0] > i[1]: # i[1] is the node_name ==> descending edge
                self.links_descend.append(link_object)

            else:
                self.links_ascend.append(link_object)

    def add_ant_mass(self,ant_mass):
        self.ant_mass += ant_mass

    def remove_all_mass(self):
        self.ant_mass = 0

    def fold_function(self):
        if self.node_name == 1: # Start node
            self.add_ant_mass(1)

        for i in self.links_descend:
            self.ant_mass += i.ant_mass_descend
            self.avg_distance += i.ant_mass_descend*i.avg_distance_descend

        for i in self.links_ascend:
            self.ant_mass += i.ant_mass_ascend
            self.avg_distance += i.ant_mass_ascend*i.avg_distance_ascend

        if self.ant_mass == 0:
            self.avg_distance = 0

        else:
            self.avg_distance = self.avg_distance/self.ant_mass

        if self.node_name == len(self.graph.nodes_list):
            self.remove_all_mass()

    def split_fold_function(self):
        out_phero = [i.phero for i in self.links]
        self.out_link_phero = sum(out_phero)

    def split_function(self): # note that for outward links the edge order switches (ascend => descend)
        if self.node_name == len(self.graph.nodes_list): # final node

            for i in self.links_ascend:
                i.ant_mass_descend = 0
                i.avg_distance_descend = 0

            self.ant_mass = 0
            self.avg_distance = 0

            return

        for i in self.links_descend:
            i.ant_mass_ascend = self.ant_mass*i.phero/self.out_link_phero
            i.avg_distance_ascend = self.avg_distance

        for i in self.links_ascend:
            i.ant_mass_descend = self.ant_mass*i.phero/self.out_link_phero
            i.avg_distance_descend = self.avg_distance
            # print(self.node_name,self.avg_distance,i.link_name,i.avg_distance_descend)

        self.ant_mass = 0
        self.avg_distance = 0

class Link(tuple):
    def __new__(cls,link_name,networkx_graph):
        return tuple.__new__(cls,(link_name,networkx_graph))

    def initialize(self):
        self.link_name = self[0] # a tuple containing the head and tail node connected
        # self.graph = self[1]
        self.link_distance = self[1].edge[self.link_name[0]][self.link_name[1]]['dist'] # d_l
        self.phero = self[1].edge[self.link_name[0]][self.link_name[1]]['phero'] # p
        self.ant_mass_ascend = 0 # n in the graph
        self.ant_mass_descend = 0 # n in the graph
        self.avg_distance_ascend = 0 # d_f from low to high (eg. 1-2)
        self.avg_distance_descend = 0 # d_f from high to low (eg. 2-1)

    def reset(self):
        self.ant_mass_ascend = 0 # n
        self.ant_mass_descend = 0 # n
        self.avg_distance_ascend = 0 # d_f from low to high (eg. 1-2)
        self.avg_distance_descend = 0 # d_f from high to low (eg. 2-1)

    def link_function(self):
        if self.ant_mass_ascend and self.ant_mass_descend != 0:
            self.avg_distance_ascend += self.link_distance
            self.avg_distance_descend += self.link_distance

        elif self.ant_mass_ascend != 0:
            self.avg_distance_ascend += self.link_distance
            self.avg_distance_descend = 0

        elif self.ant_mass_descend != 0:
            self.avg_distance_ascend = 0
            self.avg_distance_descend += self.link_distance

        else:
            self.avg_distance_descend = 0
            self.avg_distance_ascend = 0

class Ant_Graph(nx.Graph):
    def __new__(cls,networkx_graph_object):
        return nx.Graph.__new__(cls)

    def initialize(self,evaporation_rate=0.5):
        self.nodes_list = [Node(i,self) for i in self.nodes()]
        self.links_list = [Link(i,self) for i in self.edges()]

        for i in self.links_list:
            i.initialize()

        for i in self.nodes_list:
            i.initialize()

        self.leftover = 1 - evaporation_rate
        self.nodes_list[0].add_ant_mass(1)

    def node_execution(self):
        avg_dist_dict_ascend = {}
        avg_dist_dict_descend = {}
        ant_mass_dict_ascend = {}
        ant_mass_dict_descend = {}

        for i in self.nodes_list:
            i.split_fold_function()
            i.split_function()

        for i in self.links_list:
            i.link_function()

        for i in self.links_list:
            avg_dist_dict_ascend[i.link_name] = i.avg_distance_ascend
            avg_dist_dict_descend[i.link_name] = i.avg_distance_descend
            ant_mass_dict_ascend[i.link_name] = i.ant_mass_ascend
            ant_mass_dict_descend[i.link_name] = i.ant_mass_descend

        for i in self.nodes_list:
            i.fold_function()

        for i in self.links_list:
            i.reset()

        return (avg_dist_dict_ascend,avg_dist_dict_descend,ant_mass_dict_ascend,ant_mass_dict_descend)

    def ants_generation_and_activity_cycle(self):
        return self.node_execution()

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
    plt.gcf().clear()

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
