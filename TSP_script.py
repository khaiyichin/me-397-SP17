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
3. create ants as objects
'''

import networkx as nx
from matplotlib import pyplot as plt
import numpy
import sys
import re

class ant(object):
    def __init__(self):
        self.memory = []

    def visit(self,town):
        self.memory.append(town)

def construct_graph(file):
    # Read coordinates
    opened_file = open(file,"r")
    read_coords = opened_file.read()
    read_coords = re.split('\n|,',read_coords)
    opened_file.close()

    # Convert coordinate strings to integers
    int_coords = [[int(read_coords[i]),int(read_coords[i+1])] for i in
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
    axes.set_xticks(numpy.arange(0,101,2))
    axes.set_yticks(numpy.arange(0,101,2))

    # Draw network on figure
    nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w')
    plt.grid('on')
    plt.show()

def ACO_meta-heuristic():
    while (termination_criterion_not_satisfied):

def compute_transition_probabilities(A,M,constraints):
    # if M contains the town already ==> 0
    return probability

def update_ant_memory():
    return M

def apply_ant_decision_policy(probability_dict,constraints):

def ants_generation_and_activity():

def new_active_ant():
    cities_list = []

def main():
    construct_graph("oliver30Coords.txt")

if __name__ == '__main__' :
    main()
