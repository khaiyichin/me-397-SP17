'''
TSP with Ant Colony Optimization Pseudocode

procedure AC0_meta-heuristic():
    while (termination_criterion_not_satisfied):
        schedule_activities:
            ants_generation_and_activity;
            pheromone_evaporation;
            daemon_actions(); {optional}
        end schedule_activities
    end while
end procedure

procedure ants_generation_and_activity():
    while (available_resources):
        schedule_the_creation_of_a_new_ant();
        for each ant/location:
            new_active_ant();
    end while
end procedure

procedure new_active_ant(): {ant lifecycle}
    initialize_ant();
    M = update_ant_memory();
    while (current_state != target_state):
        A = read_local_ant-routing_table();
        P = compute_transition_probabilities(A,M,omega);
        next_state = apply_ant_decision_policy(P,omega);
        move_to_next_state(next_state);
        if (online_step-by-step_pheromone_update): !!! delay pheromone until all ants finishes their cycle
            foreach visited_arc in psi_solution:
                deposit_pheromone_on_the_visited_arc();
                update_ant-routing_table(); the network object is the routing table with phero and dist
            end foreach
        end if
        die();
    end while
end procedure

TO-DOs:
1. develop oliver30 graph
    - coords from 'A study of permutation crossover operators on the tsp' by
    Oliver, Smith, Holland 1987.
2. create ants as objects --- partially done; might needa add more methods/attr
3. created graph as object by initializing -- done
4. create city tuple class that instantiates object and reference to networkx object -- done
5. move everything below accordingly to Ant, Ant_Graph, City etc classes:
    compute probabilities using memory from local routing table -- done
    complete movement of ant (storing memory) -- done
    update trail level (outside of individual ant loop - higher level) -- (have to properly kill ants)
    update trail intensity for edges that evaporates
6. create loop to iterate over all starting city objects for ants
9. create high level loop that has cycle_num as argument (termination)
'''
import random
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import sys
import re

Q = 10 # pheromone constant

class City(tuple):
    def __new__(cls,node_name,networkx_graph):
        return tuple.__new__(cls,(node_name,networkx_graph)) # the static method __new__ creates and return a new instance of a class from its first argument

    def node_name(self):
        return self[0]

    def neighbors(self):
        return self[1].neighbors(self[0])

    def distance_from(self,next_city_name):
        return self[1].edge[self[0]][next_city_name]['dist']

    def pheromones_on_edge(self,next_city_name):
        return self[1].edge[self[0]][next_city_name]['phero']

# city = City('NYC', graph)
# for road in city.neighbors():
#     road.length

class Ant_Graph(nx.Graph):
    def __new__(cls,graph_object):
        return nx.Graph.__new__(cls,graph_object)

    def evaporate():
        for edge in self.edges():
            edge['phero'] *= decay_rate

class Ant(object):
    def __init__(self,starting_city,phero=Q):
        self.memory = [starting_city.node_name()]
        self.current_state = starting_city.node_name()
        self.map = starting_city[1] # City() class object
        self.first_state = starting_city.node_name()
        self.travelled = 0
        self.phero = phero
        self.routing_table = []

    def generate_routing_table(self):
        alpha = 1
        beta = 5

        travelled_cities = [i for i in self.memory]
        avail_options = [i for i in self.map.neighbors(self.current_state) if i not in travelled_cities]

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
        next_city = self.choices()
        self.memory.append(next_city)
        self.travelled += self.map.edge[self.current_state][next_city]['dist']
        self.current_state = next_city

        self.routing_table = []

    def cycle(self): # each cycle
        for cities in range(len(self.map.nodes())-1):
            self.move_to_next_state()

        self.round_trip()

        print(self.memory)
        print(self.travelled)

    def round_trip(self):
        self.travelled += self.map.edge[self.current_state][self.first_state]['dist']
        self.memory.append(self.first_state)

    # def last_memory(self):
    #     return self.memory[-1]

    def lay_pheromones(self):
        self.phero_per_unit = self.phero/self.travelled
        backwards_mem = list(reversed(self.memory))

        # Lay pheromone on the networkx graph object
        for step in range(len(network.nodes())):
            head_node = backwards_mem[step]
            tail_node = backwards_mem[step+1]
            distance = network.edge[head_node][tail_node]['dist']
            network.edge[head_node][tail_node]['phero'] += self.phero_per_unit*distance

    # have to properly destruct ant
    # either __del__ or __exit__

# class Routing_Table_Element(object):
#     def __init__(self,edge):
#         self.name   = edge['name']
#         self.phero  = edge['phero']
#         self.dist   = edge['dist']
#         self.vis    = 1./self.dist
#         self.prob   = 0

# class Point(object):
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y

# def ACO_meta-heuristic():
#     while (termination_criterion_not_satisfied):

# def ants_generation_and_activity():
#     # Initialize list to store ant objects
#     travelled_ants = []
#
#     # Shoot ants through graph
#     for city_pos in range(1,len(network)+1):
#         travelled_ants.append(new_active_ant(network.node[city_pos]))
#
#     # Initialize condition for shortest length ==> a very large number
#     shortest = 10000
#
#     # Send ants to retrace their tours to lay pheromones
#     for each_ant in travelled_ants:
#         if (shortest > each_ant.travelled):
#             shortest = each_ant.travelled
#             result = shortest,each_ant.memory
#         each_ant.lay_pheromones()
#
#     return result # the shortest route for this current iteration

# def new_active_ant(start_location):
#     ant = Ant(start_location['name'])
#
#     for steps in range(len(network)-1):
#         list_of_options = read_local_ant_routing_table(ant) # should be done, but clean up
#         next_city = compute_transition_probabilities(list_of_options)
#         move_to_next_state(ant,next_city)
#
#     # Final step back to starting point
#     ant.round_trip()
#
#     return ant # returns ant object to be appended to list

# Point = namedtuple('Point', 'x', 'y')
# a = Point(x=1, y=3)
# class Town(Point):

def initialize_graph(file):
    # Read coordinates
    opened_file = open(file,"r")
    read_coords = opened_file.read()
    read_coords = re.split('\n|,',read_coords)
    opened_file.close()

    # Convert coordinate strings to integers
    int_coords = [[int(read_coords[2*i]),int(read_coords[2*i+1])] for i in
    range(len(read_coords)//2)]

    num_of_nodes = len(int_coords)
    init_phero = 0.001

    # Initialize dictionary with coordinates
    pos = dict(zip(range(1,num_of_nodes+1),int_coords))

    graph = nx.Graph()
    graph.add_nodes_from(pos.keys())

    for n,p in pos.items():
        graph.node[n]['name'] = n
        graph.node[n]['x'] = p[0]
        graph.node[n]['y'] = p[1] # nodes from 1 to

    for i in range(1,num_of_nodes+1):
        for j in range(i+1,num_of_nodes+1):
            distance = distance_calc(graph.node[i],graph.node[j])
            graph.add_edge(i,j,phero=init_phero,dist=distance)

    # Create plot with gridlines
    fig = plt.figure()
    axes = fig.gca()
    axes.set_xticks(np.arange(0,101,2))
    axes.set_yticks(np.arange(0,101,2))

    # Draw network on figure
    nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w',edgelist=[])
    plt.grid('on')
    # plt.show()

    return graph # networkx object (will be used to update pheromones later)

# def read_local_ant_routing_table(ant):
#     alpha = 1   # trail exponent
#     beta = 5    # visibility exponent
#
#     current_state = ant.last_memory()  # Current ant position in integer
#     avail_options = [i for i in range(1,len(network.nodes())+1) if i not in ant.memory] # Remaining cities in integers
#
#     list_of_edges = []
#     denom = 0
#     for option in avail_options:
#         edge = Routing_Table_Element(network.edge[current_state][option])
#         edge.prob = (edge.phero**alpha)*(edge.vis**beta)
#         denom += (edge.phero**alpha)*(edge.vis**beta)
#         list_of_edges.append(edge)
#
#     local_routing_table = []
#     for edges in list_of_edges:
#         edges.prob = edges.prob/denom
#         local_routing_table.append(edges)
#
#     return local_routing_table  # list of objects of all options (next immediate node)

# def compute_transition_probabilities(local_routing_table):
#     rand = random.random()
#
#     probabilities = [edges.prob for edges in local_routing_table]
#     choice = random.choices(local_routing_table,weights=probabilities)
#     choice = choice[0]  # random.choices return a list; we want just one element
#
#     return choice

# def move_to_next_state(ant,next_state):
#     current_state = ant.memory[-1]
#     city1,city2 = next_state.name[0]
#     if city1 != current_state:
#         next_city = city1
#     else:
#         next_city = city2
#
#     ant.visit(next_city,next_state.dist)

def distance_calc(point1,point2):
    x1 = point1['x']
    x2 = point2['x']
    y1 = point1['y']
    y2 = point2['y']

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def main():
    # Define a global networkx object 'network' of coordinates
    network = initialize_graph("oliver30Coords.txt")
    city1 = City(1,network)
    city2 = City(2,network)
    city3 = City(3,network)
    ant1 = Ant(city1)
    ant2 = Ant(city2)
    ant3 = Ant(city3)
    ant1.cycle()
    ant2.cycle()
    ant3.cycle()

    sys.exit()

if __name__ == '__main__' :
    main()
