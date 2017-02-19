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
        if (online_step-by-step_pheromone_update):
            foreach visited_arc in psi_solution:
                deposit_pheromone_on_the_visited_arc();
                update_ant-routing_table();
            end foreach
        end if
        die();
    end while
end procedure

TO-DOs:
1. develop oliver30 graph
    - coords from 'A study of permutation crossover operators on the tsp' by
    Oliver, Smith, Holland 1987.
2. run program(?) -tentative-
3. create ants as objects --- partially done; might needa add more methods/attr
'''

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

class routing_table(object):
    def __init__(self,cities,trail,coords):
        self.cities = cities

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

def construct_graph(file):
    # Read coordinates
    opened_file = open(file,"r")
    read_coords = opened_file.read()
    read_coords = re.split('\n|,',read_coords)
    opened_file.close()

    # Convert coordinate strings to integers
    int_coords = [[int(read_coords[2*i]),int(read_coords[2*i+1])] for i in
    range(len(read_coords)//2)]

    # Initialize dictionary with coordinates
    pos = dict(zip(range(1,len(int_coords)+1),int_coords))

    graph = nx.Graph()
    graph.add_nodes_from(pos.keys())

    for n,p in pos.items():
        graph.node[n]['pos'] = p

    # Create plot with gridlines
    fig = plt.figure()
    axes = fig.gca()
    axes.set_xticks(np.arange(0,101,2))
    axes.set_yticks(np.arange(0,101,2))

    # Draw network on figure
    nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w')
    plt.grid('on')
    # plt.show()

    return int_coords

# def trail_intensity(alpha):
#
# def visibility(beta):
#
# def ACO_meta-heuristic():
#     while (termination_criterion_not_satisfied):
#
# def compute_transition_probabilities(A,M,constraints):
#     # if M contains the town already ==> 0
#     rand = random.random()
#
#     options = [i for i in A.cities if i not in M] # eliminating towns that have been visited
#     sum([j for j in options]) # sum of all trail and visibility in local node
#     probabilities = {}
#     return probability
#
# def read_local_ant-routing_table():
#     # look at options to move to
#     return A # list of all options (next immediate node)
#
#
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
#     cities_list = []

def distance(point1,point2):
    x1 = point1.x
    x2 = point2.x
    y1 = point1.y
    y2 = point2.y

def main():
    coords = construct_graph("oliver30Coords.txt")

    # Create point objects (nodes) with coordinates and compile into a vector
    points = [Point(coords[i][0],coords[i][1]) for i in range(len(coords))]

    #
    # # Create matrix of distances
    # numpy.ones(len())
    # location_matrix = np.matrix([i for i in points])
    # distance_matrix = numpy.square(location_matrix)
    #
    # # Create map (ant routing table)
    # map = routing_table(coords)

if __name__ == '__main__' :
    main()
