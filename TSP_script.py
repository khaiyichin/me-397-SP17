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
4. complete local routing table (focus on one ant) -- done
5. compute probabilities using memory from local routing table -- done
6. complete movement of ant (storing memory)
7. update trail level (outside of individual ant loop - higher level)
7. create high level loop that has cycle_num as argument (termination)
'''
import random
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import sys
import re

class Ant(object):
    def __init__(self):
        self.memory = []

    def visit(self,town):
        self.memory.append(town)

    # have to properly destruct ant
    # either __del__ or __exit__

class Routing_Table_Element(object):
    def __init__(self,edge):
        self.name   = edge['name']
        self.phero  = edge['phero']
        self.dist   = edge['dist']
        self.vis    = 1./self.dist
        self.prob   = 0

class Point(object):
    def __init__(self,x,y):
        self._x = x
        self._y = y

    @property
    def x():
        return self._x

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
            graph.add_edge(i,j,phero=init_phero,dist=distance,name=[(i,j),(j,i)])

    # Create plot with gridlines
    fig = plt.figure()
    axes = fig.gca()
    axes.set_xticks(np.arange(0,101,2))
    axes.set_yticks(np.arange(0,101,2))

    # Draw network on figure
    nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w',edgelist=[])
    plt.grid('on')
    # plt.show()

    return graph # return graph object (will be used to update pheromones later)

# def ACO_meta-heuristic():
#     while (termination_criterion_not_satisfied):
#
def compute_transition_probabilities(local_routing_table):
    rand = random.random()

    probabilities = [edges.prob for edges in local_routing_table]
    choice = random.choices(local_routing_table,weights=probabilities)
    choice = choice[0]  # random.choices return a list

    return choice

# def probability_selector():


def read_local_ant_routing_table(ant,network):
    # look at options to move to
    alpha = 1   # trail exponent
    beta = 2    # visibility exponent

    current_state = ant.memory[-1]  # Current ant position in integer
    avail_options = [i for i in range(1,len(network.nodes())+1) if i not in ant.memory] # Remaining cities in integers

    list_of_edges = []
    denom = 0
    for option in avail_options:
        edge = Routing_Table_Element(network.edge[current_state][option])
        edge.prob = (edge.phero**alpha)*(edge.vis**beta)
        denom += (edge.phero**alpha)*(edge.vis**beta)
        list_of_edges.append(edge)

    local_routing_table = []
    for edges in list_of_edges:
        edges.prob = edges.prob/denom
        local_routing_table.append(edges)

    return local_routing_table  # a list of objects of all options (next immediate node)

# def ant_routing_table(trail_matrix,distance_matrix): ### no need for this anymore
#     alpha = 1
#     beta = 5
#
#     visibility_matrix = np.zeros(distance_matrix.shape)
#     for i in range(len(distance_matrix)):
#         for j in range(i+1,len(distance_matrix)):
#             visibility_matrix[i,j] = np.reciprocal(distance_matrix[i,j])
#             visibility_matrix[j,i] = visibility_matrix[i,j]
#
#     numer = np.multiply(trail_matrix**alpha,visibility_matrix**beta) # in matrix form
#     denom = np.sum(numer,axis=1) # in array form
#     general_routing_table = numer/denom # in matrix form
#     return general_routing_table

def distance_calc(point1,point2):
    x1 = point1['x']
    x2 = point2['x']
    y1 = point1['y']
    y2 = point2['y']

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# def update_ant_memory():
#
#     return M
#
# def apply_ant_decision_policy(probability_dict,constraints):
#
#
# def ants_generation_and_activity():
#
#
# def new_active_ant():
#
#     while (current_state != target_state):
#         A = read_local_routing_table(ant_object) # should be done, but clean up
#         next_link = compute_transition_probabilities(A)
#         move_to_next_state(ant,next_link)

def main():
    # Define a list of coordinates
    graph = initialize_graph("oliver30Coords.txt")

    ant1 = Ant()
    ant1.memory = [1,2,3,4,5,6,7,8,9,10,28,27,26,25,29,30,12,11,21,13,14,15,16,17,18,19,20]

    x = read_local_ant_routing_table(ant1,graph)


    y =[(x[i].name,x[i].prob) for i in range(len(x))]

    n = compute_transition_probabilities(x)
    print(n.name)
    print(graph.edge[25][29]['dist'])
    print(graph.edge[25][30]['dist'])
    sys.exit()

    # Create point objects (nodes) with coordinates and compile into a vector
    # points = [Point(coords[i][0],coords[i][1]) for i in range(len(coords))] not needed

    # while (cycle != num_of_cycles or stagnate == False): # Overall loop
    #
    #     for (num_of_ants): # Iterating through each ant
    #
    #         while (current_state != target_state): # Tour of an ant loop



    sys.exit()

    #
    # # Create map (ant routing table)
    # map = routing_table(coords)

if __name__ == '__main__' :
    main()
