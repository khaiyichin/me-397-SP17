'''
Path Planning with Node Based ACO (Only forward subprogram based on Jack's proposal)
'''
import networkx as nx
import random
import sys
from node_based import initialize_graph, Node, Link
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import errno

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
nodes = 3 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 0 # number of edges to remove
max_cycles = 1 # total number of cycles
node_executions = 100

def process_data(ant_graph,dist_ascend_data,dist_descend_data,mass_ascend_data,mass_descend_data):
    all_data = [dist_ascend_data,dist_descend_data,mass_ascend_data,mass_descend_data]
    folder_name = str(datetime.date.today())
    make_sure_path_exists(folder_name)

    time_string = str(datetime.datetime.now().strftime('%H%M%S'))

    # Only interested in mass for now

    # Process ascending distances
    # data = all_data[0]
    # link_names = list(data[0].keys())
    # data_dict = {link:[] for link in link_names}
    # diff_dict_dist = {link:[] for link in link_names}
    # num_of_node_executions = list(range(1,len(data)+1))
    #
    # figure = plt.gcf()
    # figure.set_size_inches(14,10)
    #
    # sum_of_dist = np.zeros(len(data))
    # sum_of_diff_dist = np.zeros(len(data)-1)
    #
    # for link in link_names:
    #     for i in range(len(data)):
    #         data_dict[link].append(data[i][link])
    #
    #     sum_of_dist += np.array(data_dict[link])
    #
    # asc_dist_dict = data_dict # temporary storage for ascending distances

    # Process descending distances
    # data = all_data[1]
    # link_names = list(data[0].keys())
    # data_dict = {link:[] for link in link_names}
    # num_of_node_executions = list(range(1,len(data)+1))
    #
    # total_dist_dict = {link:[] for link in link_names} # total avg distance in each link (ascending and descending)
    #
    # figure = plt.gcf()
    # figure.set_size_inches(14,10)
    #
    # for link in link_names:
    #     if max(max(link_names)) in link:
    #         data_dict[link] = (np.ones(len(data)))
    #     else:
    #         for i in range(len(data)):
    #             data_dict[link].append(data[i][link])
    #
    #     total_dist_dict[link] = np.array(asc_dist_dict[link])+np.array(data_dict[link])
    #     sum_of_dist += np.array(data_dict[link])
    #
    #     for i in range(len(sum_of_dist)-1):
    #         sum_of_diff_dist[i] = sum_of_dist[i+1] - sum_of_dist[i]
    #
    # dsc_dist_dict = data_dict # temporary storage for descending distances

    # Plot average traveled distance in each link
    # for link in link_names:
    #     plt.plot(num_of_node_executions,total_dist_dict[link],label='Link'+str(link))
    #
    # plt.legend(ncol=5)
    # plt.grid()
    # plt.ylabel('Average Distance Traveled')
    # plt.xlabel('Number of Node Executions')
    # plt.title('Average Distance across Each Link')
    # filename = 'node_D_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    # plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)
    #
    # plt.gcf().clear()
    #
    # # Plot sum of avg distance across all links
    # plt.plot(num_of_node_executions,sum_of_dist)
    #
    # plt.grid()
    # plt.ylabel('Average Distance Traveled')
    # plt.xlabel('Number of Node Executions')
    # plt.title('Average Distance across All Links')
    # filename = 'node_sumD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    # plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)
    #
    # plt.gcf().clear()
    #
    # # Plot difference in avg distance after next node execution
    # plt.plot(num_of_node_executions[1:],sum_of_diff_dist)
    #
    # plt.grid()
    # plt.ylabel('Average Distance Traveled')
    # plt.xlabel('Number of Node Executions')
    # plt.title('Average Distance across All Links')
    # filename = 'node_diffD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    # plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)
    #
    # plt.gcf().clear()

    # Process Mass Data
    # Process ascending masses
    data = all_data[2]
    link_names = list(data[0].keys())
    data_dict = {link:[] for link in link_names}
    data_dict_mass = {link:[] for link in link_names}
    data_dict_mass_per_dist = {link:[] for link in link_names}
    num_of_node_executions = list(range(1,len(data)+1))

    sum_of_mass = np.zeros(len(data))
    sum_of_diff_mass = np.zeros(len(data)-1)

    total_mass_dict = {link:[] for link in link_names} # total mass in each link (ascending and descending)

    for link in link_names:
        for i in range(len(data)):
            data_dict[link].append(data[i][link])

        # data_dict_mass_per_dist[link] = np.divide(data_dict[link],asc_dist_dict[link]) # only interested in mass for now
        sum_of_mass += np.array(data_dict[link])

    temp_mass_dict = data_dict # temporary storage for ascending mass

    # Process descending masses
    data = all_data[3]
    link_names = list(data[0].keys())
    data_dict = {link:[] for link in link_names}
    data_dict_mass = {link:[] for link in link_names}
    num_of_node_executions = list(range(1,len(data)+1))

    for link in link_names:
        for i in range(len(data)):
            data_dict[link].append(data[i][link])

        total_mass_dict[link] = np.array(temp_mass_dict[link])+np.array(data_dict[link])
        # data_dict_mass_per_dist[link] += np.divide(data_dict[link],dsc_dist_dict[link]) # only interested in mass for now
        sum_of_mass += np.array(data_dict[link])

        for i in range(len(sum_of_mass)-1):
            sum_of_diff_mass[i] = sum_of_mass[i+1] - sum_of_mass[i]

    # Plotting Data
    # Plot ant mass in each link
    figure = plt.gcf()
    figure.set_size_inches(14,10)

    for link in link_names:
        plt.plot(num_of_node_executions,total_mass_dict[link],label='Link'+str(link))
        last_val = (num_of_node_executions[-1],total_mass_dict[link][-1])
        plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.legend(ncol=5,loc=2)
    plt.grid()
    plt.ylabel('Ant Mass')
    plt.xlabel('Number of Node Executions')
    plt.title('Ant Mass across Each Link')
    filename = 'node_M_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    # Plot sum of ant mass across all links
    plt.plot(num_of_node_executions,sum_of_mass)
    last_val = (num_of_node_executions[-1],sum_of_mass[-1])
    plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.grid()
    plt.ylabel('Ant Mass')
    plt.xlabel('Number of Node Executions')
    plt.title('Ant Mass across All Links')
    filename = 'node_sumM_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    # Plot difference in ant mass after next node execution
    plt.plot(num_of_node_executions[1:],sum_of_diff_mass)
    last_val = (num_of_node_executions[-1],sum_of_diff_mass[-1])
    plt.text(last_val[0],last_val[1],str(last_val[1]))

    plt.grid()
    plt.ylabel('Change of Ant Mass ')
    plt.xlabel('Number of Node Executions')
    plt.title('Change of Ant Mass across All Links')
    filename = 'node_diffM_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    plt.gcf().clear()

    sum_of_mass_per_dist = np.zeros(len(data))

    # # Plot ant mass per unit distance in each link
    # for link in link_names:
    #     x = np.divide(total_mass_dict[link],total_dist_dict[link])
    #     plt.plot(num_of_node_executions,data_dict_mass_per_dist[link],label='Link'+str(link))
    #     last_val = (num_of_node_executions[-1],data_dict_mass_per_dist[link][-1])
    #     plt.text(last_val[0],last_val[1],str(link)+str(last_val[1]))
    #     sum_of_mass_per_dist += x
    #
    # plt.legend(ncol=5)
    # plt.grid()
    # plt.ylabel('Ant Mass per Unit Distance')
    # plt.xlabel('Number of Node Executions')
    # plt.title('Ant Mass per Unit Distance across All Links')
    # filename = 'node_MperD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    # plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)
    #
    # plt.gcf().clear()

    # Plot ant mass per unit distance
    # # x = np.divide(sum_of_mass,sum_of_dist)
    # plt.plot(num_of_node_executions,sum_of_mass_per_dist)
    # last_val = (num_of_node_executions[-1],sum_of_mass_per_dist[-1])
    # plt.text(last_val[0],last_val[1],str(last_val[1]))
    #
    # plt.grid()
    # plt.ylabel('Ant Mass per Unit Distance')
    # plt.xlabel('Number of Node Executions')
    # plt.title('Ant Mass per Unit Distance across All Links')
    # filename = 'node_sumMperD_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    # plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)
    #
    # plt.gcf().clear()

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main():
    ant_graph = initialize_graph(yaml_file='4nodes4edges.yaml',space=euc_space,size=nodes,
    num_of_nodes_to_remove=rand_nodes,num_of_edges_to_remove=rand_edges)

    dist_ascend = []
    dist_descend = []
    mass_ascend = []
    mass_descend = []

    for i in range(node_executions):
        data = ant_graph.node_execution()
        dist_ascend.append(data[0])
        dist_descend.append(data[1])
        mass_ascend.append(data[2])
        mass_descend.append(data[3])

    process_data(ant_graph,dist_ascend,dist_descend,mass_ascend,mass_descend)

if __name__ == '__main__':
    main()
