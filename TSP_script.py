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
6. create loop to iterate over all starting city objects for ants -- done
7. create high level loop that has cycle_num as argument (termination) -- done
8. include stagnation behavior as condition (think about what kind of stagnation)
'''
import random
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import sys
import re

Q = 10 # pheromone constant
decay_rate = 0.5 # trail evaporation
rho = 1 - decay_rate # trail persistence
max_cycles = 200 # total number of cycles

class City(tuple):
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

class Ant(object):
    def __init__(self,starting_city,phero=Q):
        self.memory = [starting_city.node_name()]
        self.current_state = starting_city.node_name()
        self.map = starting_city[1] # City() class tuple, which the 2nd element is the networkx graph object
        self.first_state = starting_city.node_name()
        self.travelled = 0
        self.phero = phero
        self.routing_table = []

    def generate_routing_table(self):
        alpha = 1
        beta = 5

        travelled_cities = self.memory
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

    def round_trip(self):
        self.travelled += self.map.edge[self.current_state][self.first_state]['dist']
        self.memory.append(self.first_state)
        # self.current_state = self.first_state

    def lay_pheromones(self):
        self.phero_per_unit = self.phero/self.travelled
        backwards_mem = list(reversed(self.memory))

        # Lay pheromone on the networkx graph object
        for step in range(self.map.number_of_nodes()):
            head_node = backwards_mem[step]
            tail_node = backwards_mem[step+1]
            distance = self.map.edge[head_node][tail_node]['dist']
            self.map.edge[head_node][tail_node]['phero'] += self.phero_per_unit*distance

    # have to properly destruct ant ???
    # either __del__ or __exit__

def ACO_metaheuristic():
    list_of_shortest_each_cycle = []
    num_of_cycles = max_cycles
    current_cycle = 0

    # Initialize NetworkX Graph object
    networkx_graph = initialize_graph("oliver30Coords.txt")
    ant_graph_object = Ant_Graph(networkx_graph)

    while (current_cycle != num_of_cycles):
        shortest_for_now = ants_generation_and_activity_cycle(ant_graph_object)
        list_of_shortest_each_cycle.append(shortest_for_now)

        ant_graph_object.evaporate()
        current_cycle += 1

    dist_and_route = list(zip(*list_of_shortest_each_cycle)) # asterisk means unpacking arguments from a list
    distances = dist_and_route[0]

    optimal_dist = min(distances)
    ind = dist_and_route[0].index(optimal_dist)
    optimal_route = list_of_shortest_each_cycle[ind][1]

    print("Shortest tour consists of these cities:",optimal_route,
    "with a distance of",optimal_dist)
    print("Number of cycles ran =",current_cycle)

def ants_generation_and_activity_cycle(networkx_graph):
    # Initialize list to store ant objects
    travelled_ants = []

    # Create starting points/cities
    num_of_starting_nodes = networkx_graph.number_of_nodes() # in the TSP problem we use all cities as start points
    starting_cities = [City(i,networkx_graph) for i in range(1,num_of_starting_nodes+1)]

    # Shoot ants through graph
    for city in starting_cities:
        travelled_ants.append(Ant(city))
        travelled_ants[-1].cycle()

    # Initialize condition for shortest length ==> a very large number
    shortest = 10000

    # Send ants to retrace their tours to lay pheromones
    for each_ant in travelled_ants:
        if (shortest > each_ant.travelled):
            shortest = each_ant.travelled
            result = shortest,each_ant.memory
        each_ant.lay_pheromones()

    return result # the shortest route for this current iteration

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

def distance_calc(point1,point2):
    x1 = point1['x']
    x2 = point2['x']
    y1 = point1['y']
    y2 = point2['y']

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def main():
    ACO_metaheuristic()

if __name__ == '__main__' :
    main()
