'''
Path Planning with Signal Based ACO
'''
import networkx as nx
import random
import sys
from signal_based import initialize_graph, Ant, Node
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import errno

Q = 10 # pheromone constant
num_of_ants = 10000 # total number of ants
nodes = 3 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 0 # number of edges to remove
max_cycles = 1 # total number of cycles

def ACO_metaheuristic():
    # Initialize NetworkX Graph object
    ant_graph = initialize_graph(yaml_file='4nodes4edges.yaml',space=euc_space,size=nodes,
    num_of_nodes_to_remove=rand_nodes,num_of_edges_to_remove=rand_edges)

    # Initialize lists to store data for each cycle
    shortest_each_cycle = []

    immutable_dict = {i:j for i,j in ant_graph.ant_counter_per_edge.items()} # exactly the same things, but immutable
    num_of_ants_per_edge_each_cycle = [immutable_dict] # this list updates ALL the elements every cycle if it appends the object attribute

    pheromone_levels_each_cycle =[{(i,j):k for i,j,k in ant_graph.edges(data='phero')}] # creating a dict for the edges; .edges() returns a list

    current_cycle = 0

    # Run until desired number of cycles reached
    while (current_cycle != max_cycles):
        shortest_for_now = ants_generation_and_activity_cycle(ant_graph)
        shortest_each_cycle.append(shortest_for_now)

        immutable_dict = {i:j for i,j in ant_graph.ant_counter_per_edge.items()}
        num_of_ants_per_edge_each_cycle.append(immutable_dict)

        pheromone_levels_each_cycle.append({(i,j):k for i,j,k in ant_graph.edges(data='phero')})

        ant_graph.evaporate()
        ant_graph.initialize()
        current_cycle += 1

    # Process data for shortes route
    dist_and_route = list(zip(*shortest_each_cycle)) # asterisk means unpacking arguments from a list
    distances = dist_and_route[0]

    optimal_dist = min(distances)
    ind = dist_and_route[0].index(optimal_dist)
    optimal_route = shortest_each_cycle[ind][1]

def ants_generation_and_activity_cycle(ant_graph):
    # Initialize list to store ant objects
    average_dist_per_edge_data = []
    traveled_ants = []
    average_distance = []
    total_ant_distance = 0
    ant_mass = {edge:np.zeros(num_of_ants) for edge in ant_graph.edges()}
    # print(ant_graph.edges(data=True))

    # Shoot ants through graph one at a time
    for ant in range(1,num_of_ants+1):
        traveled_ants.append(Ant(Node(1,ant_graph)))
        traveled_ants[-1].cycle()
        current_ant_distance = traveled_ants[-1].traveled
        total_ant_distance += current_ant_distance

        for i in ant_graph.ant_counter_per_edge:
            ant_graph.ant_counter_per_edge[i] += traveled_ants[-1].traveled_edges.count(i)

        for edge in list(set(traveled_ants[-1].traveled_edges)):
            num_of_passing_ants = ant_graph.ant_counter_per_edge[edge]
            ant_graph.total_ant_distance_counter[edge] += current_ant_distance
            cumulative_distance_on_edge = ant_graph.total_ant_distance_counter[edge]
            ant_graph.average_dist_per_edge[edge] = num_of_passing_ants/cumulative_distance_on_edge

        for edge in ant_mass.keys():
            num_of_passing_ants = ant_graph.ant_counter_per_edge[edge]
            ant_mass[edge][ant-1] = num_of_passing_ants/ant

        # Process data for average distance traveled on each edge
        avg_dist_dict = {i:j for i,j in ant_graph.average_dist_per_edge.items()}
        average_dist_per_edge_data.append(avg_dist_dict)
        # print(traveled_ants[-1].traveled_edges,ant_graph.total_ant_distance_counter)
        # print("Total ants =",ant,"; Distance per Edge =",ant_graph.average_dist_per_edge)

        # Process data for average distance traveled on entire graph (all edges)
        average_distance.append(ant/total_ant_distance)
        # print("Total ants =",ant,"; Average Distance =",average_distance[-1])

        # print(traveled_ants[-1].traveled_edges,current_ant_distance)
        # print(ant_graph.ant_counter_per_edge,ant_graph.total_ant_distance_counter)

    shortest = 100000000

    # process_avg_dist_per_edge(ant_graph,average_dist_per_edge_data) # only interested in mass for now
    process_ant_mass(ant_graph,ant_mass)

    # Make ants trace back routes and lay pheromones
    for each_ant in traveled_ants:
        if (shortest > each_ant.traveled):
            shortest = each_ant.traveled
            result = shortest,each_ant.memory
        each_ant.lay_pheromones()

    return result

def process_avg_dist_per_edge(ant_graph,data):
    folder_name = str(datetime.date.today())
    make_sure_path_exists(folder_name)

    time_string = str(datetime.datetime.now().strftime('%H%M%S'))

    edge_names = list(data[0].keys())
    data_dict = {edge:[] for edge in edge_names}
    diff_dict = {edge:[] for edge in edge_names}
    num_of_ants_shot = list(range(1,len(data)+1))

    figure = plt.gcf()
    figure.set_size_inches(14,10)

    sum_of_data = np.zeros(len(data))
    sum_of_diff = np.zeros(len(data)-1)

    for edge in edge_names:
        for i in range(len(data)):
            data_dict[edge].append(data[i][edge])

        sum_of_data += np.array(data_dict[edge])
        plt.plot(num_of_ants_shot,data_dict[edge],label='Edge'+str(edge))
        last_val = (num_of_ants_shot[-1],data_dict[edge][-1])
        plt.text(last_val[0],last_val[1],str(edge)+str(last_val[1]))

    plt.legend(ncol=5)
    plt.grid()
    plt.ylabel('Ant Mass per Unit Distance')
    plt.xlabel('Number of Ants Shot')
    plt.title('Ant Mass per Unit Distance across Each Link')
    filename = 'signal_MperD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(edge_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    # Plot sum of mass per unit distance across all links
    plt.plot(num_of_ants_shot,sum_of_data)
    last_val = (num_of_ants_shot[-1],sum_of_data[-1])
    plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.grid()
    plt.ylabel('Ant Mass per Unit Distance')
    plt.xlabel('Number of Ants Shot')
    plt.title('Ant Mass per Unit Distance across Links')
    filename = 'signal_sumMperD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(edge_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    sum_of_diff = np.zeros(len(data))
    for edge in edge_names:
        diff_dict[edge].append(data[0][edge])

        for i in range(len(data)-1):
            diff_dict[edge].append(data[i+1][edge]-data[i][edge])

        sum_of_diff += np.array(diff_dict[edge])

    plt.plot(num_of_ants_shot,sum_of_diff)

    plt.grid()
    plt.ylabel('Change of Ant Mass per Unit Distance')
    plt.xlabel('Number of Ants Shot')
    plt.title('Change of Ant Mass per Unit Distance across All Links')
    filename = 'signal_diffMperD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(edge_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

def process_ant_mass(ant_graph,data):
    folder_name = str(datetime.date.today())
    make_sure_path_exists(folder_name)

    temp = [len(i) for i in data.values()] # length of lists in dict
    time_string = str(datetime.datetime.now().strftime('%H%M%S'))
    edge_names = list(data.keys())
    data_dict = {edge:[] for edge in edge_names}
    num_of_ants_shot = list(range(1,temp[0]+1))

    figure = plt.gcf()
    figure.set_size_inches(14,10)

    sum_of_data = np.zeros(temp[0])
    sum_of_diff = np.zeros(temp[0]-1)

    for edge in edge_names:
        for i in range(len(data[edge])-1):
            sum_of_diff[i] = sum_of_data[i+1] - sum_of_data[i]

        data_dict[edge] = data[edge]
        sum_of_data += np.array(data_dict[edge])

    for edge in edge_names:
        plt.plot(num_of_ants_shot,data_dict[edge],label='Edge'+str(edge))
        last_val = (num_of_ants_shot[-1],data_dict[edge][-1])
        plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.legend(ncol=5,loc=2)
    plt.grid()
    plt.ylabel('Probabilistic Ant Mass')
    plt.xlabel('Number of Ants Shot')
    plt.title('Probabilistic Ant Mass across Each Link')
    filename = 'signal_M_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(edge_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    # Plot sum of mass per unit distance across all links
    plt.plot(num_of_ants_shot,sum_of_data)
    last_val = (num_of_ants_shot[-1],sum_of_data[-1])
    plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.grid()
    plt.ylabel('Probabilistic Ant Mass')
    plt.xlabel('Number of Ants Shot')
    plt.title('Probabilistic Ant Mass across All Links')
    filename = 'signal_sumM_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(edge_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    # Plot change of mass in all links
    plt.plot(num_of_ants_shot[1:],sum_of_diff)
    last_val = (num_of_ants_shot[-1],sum_of_diff[-1])
    plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.grid()
    plt.ylabel('Change of Probabilistic Ant Mass')
    plt.xlabel('Number of Ants Shot')
    plt.title('Change of Probabilistic Ant Mass across All Links')
    filename = 'signal_diffM_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(edge_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main():
    ACO_metaheuristic()

if __name__ == '__main__':
    main()
