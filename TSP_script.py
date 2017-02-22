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
2. create ants as objects --- partially done; might needa add more methods/attr
3. establish distance and trail matrices for routing table -- done
4. complete local routing table (focus on one ant)
5. compute probabilities using memory from local routing table
6. generate update procedure; the trail_mat (and other update activities) will be called in here
7. create high level loop that has cycle_num as argument (termination)
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
# def read_local_ant_routing_table(ant,routing_table): ### DUPLICATE - FOCUS ON ONE ANT INSTEAD (as compared to ant_routing_table())
#     # look at options to move to
#     current_state = ant.memory[-1] # Current ant position
#     avail_options = [i for i in range(routing_table.shape) if i not in ant.memory] # Remaining cities
#
#     for i in avail_options:
#         tra =  trail_matrix[current_state,i] # should be upper triangular
#         vis = distance_matrix[current_state,i] # should be upper triangular
#
#         trail_intensity[current_state,i] = tra
#         visibility[current_state,i] = vis
#
#         denom += (tra**alpha)*(vis**beta)
#     A = ((trail_intensity**alpha)*(visibility**beta)) / denom
#     return A # list of all options (next immediate node)

def ant_routing_table(trail_matrix,distance_matrix):
    alpha = 1
    beta = 5

    visibility_matrix = np.zeros(distance_matrix.shape)
    for i in range(len(distance_matrix)):
        for j in range(i+1,len(distance_matrix)):
            visibility_matrix[i,j] = np.reciprocal(distance_matrix[i,j])
            visibility_matrix[j,i] = visibility_matrix[i,j]

    numer = np.multiply(trail_matrix**alpha,visibility_matrix**beta) # in matrix form
    denom = np.sum(numer,axis=1) # in array form
    general_routing_table = numer/denom # in matrix form
    return general_routing_table

def trail_mat(past_trail_mat,update_mat):
    # Create trail_intensity matrix that will be continuously updated
    rho = 0.99
    return rho*past_trail_mat + update_mat

def distance_mat(all_points):
    # Create matrix of distances
    distance_matrix = np.zeros((len(all_points),len(all_points)))

    for i in range(0,len(all_points)-1):
        for j in range(i+1,len(all_points)):
            distance_matrix[i,j] = distance(all_points[i],all_points[j])
            distance_matrix[j,i] = distance_matrix[i,j]

    np.set_printoptions(precision=5)
    np.savetxt('distance_matrix.txt',distance_matrix,fmt="%-5.1f")

    return distance_matrix

def distance(point1,point2):
    x1 = point1.x
    x2 = point2.x
    y1 = point1.y
    y2 = point2.y

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
#     cities_list = []

def main():
    # Define a list of coordinates
    coords = construct_graph("oliver30Coords.txt")

    # Create point objects (nodes) with coordinates and compile into a vector
    points = [Point(coords[i][0],coords[i][1]) for i in range(len(coords))]

    # Create a matrix of distances
    distances = distance_mat(points)

    # Read general routing table
    ant_routing_table(np.ones(distances.shape)/10,distances) #### FOR TESTING PURPOSE -- ERASE ASAP

    #
    # # Create map (ant routing table)
    # map = routing_table(coords)

if __name__ == '__main__' :
    main()
