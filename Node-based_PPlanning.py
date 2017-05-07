'''
Path Planning with Node Based ACO (Only forward subprogram based on Jack's proposal)
'''
import networkx as nx
import random
import sys
from node_based import initialize_graph, Node, Link
import matplotlib.pyplot as plt
import datetime
import os
import errno

# init_phero = 0.001 # initializing graphs with nonzero pheromones
Q = 10 # pheromone constant
nodes = 5 # size of graph (number of nodes)
euc_space = 10 # euclidean space in each dimension (max x,y coordinates)
rand_nodes = 0 # number of nodes to remove
rand_edges = 0 # number of edges to remove
max_cycles = 1 # total number of cycles
node_executions = 4

def process_data(ant_graph,dist_ascend_data,dist_descend_data,mass_ascend_data,mass_descend_data):
    all_data = [dist_ascend_data,dist_descend_data,mass_ascend_data,mass_descend_data]
    folder_name = str(datetime.date.today())
    make_sure_path_exists(folder_name)

    time_string = str(datetime.datetime.now().strftime('%H%M%S'))

    # Process ascending distances
    data = all_data[0]
    link_names = list(data[0].keys())
    data_dict = {link:[] for link in link_names}
    num_of_node_executions = list(range(1,len(data)+1))

    figure = plt.gcf()
    figure.set_size_inches(14,10)

    for link in link_names:
        for i in range(len(data)):
            data_dict[link].append(data[i][link])

        plt.plot(num_of_node_executions,data_dict[link],label='Link'+str(link))

    plt.legend(ncol=5)
    plt.grid()
    plt.ylabel('Average Distance Traveled')
    plt.xlabel('Number of Node Executions')
    plt.title('Average Distance in the Ascending Link Direction')
    filename = 'node_D+_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    # plt.show()
    plt.gcf().clear()

    # Process descending distances
    data = all_data[1]
    link_names = list(data[0].keys())
    data_dict = {link:[] for link in link_names}
    num_of_node_executions = list(range(1,len(data)+1))

    figure = plt.gcf()
    figure.set_size_inches(14,10)

    for link in link_names:
        for i in range(len(data)):
            data_dict[link].append(data[i][link])

        plt.plot(num_of_node_executions,data_dict[link],label='Link'+str(link))

    plt.legend(ncol=5)
    plt.grid()
    plt.ylabel('Average Distance Traveled')
    plt.xlabel('Number of Node Executions')
    plt.title('Average Distance in the Descending Link Direction')
    filename = 'node_D-_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    # plt.show()
    plt.gcf().clear()

    # Process ascending masses
    data = all_data[2]
    link_names = list(data[0].keys())
    data_dict = {link:[] for link in link_names}
    num_of_node_executions = list(range(1,len(data)+1))

    figure = plt.gcf()
    figure.set_size_inches(14,10)

    for link in link_names:
        for i in range(len(data)):
            data_dict[link].append(data[i][link])

        plt.plot(num_of_node_executions,data_dict[link],label='Link'+str(link))

    plt.legend(ncol=5)
    plt.grid()
    plt.ylabel('Ant Mass')
    plt.xlabel('Number of Node Executions')
    plt.title('Ant Mass in the Ascending Link Direction')
    filename = 'node_M+_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    # plt.show()
    plt.gcf().clear()

    # Process descending masses
    data = all_data[3]
    link_names = list(data[0].keys())
    data_dict = {link:[] for link in link_names}
    num_of_node_executions = list(range(1,len(data)+1))

    figure = plt.gcf()
    figure.set_size_inches(14,10)

    for link in link_names:
        for i in range(len(data)):
            data_dict[link].append(data[i][link])

        plt.plot(num_of_node_executions,data_dict[link],label='Link'+str(link))

    plt.legend(ncol=5)
    plt.grid()
    plt.ylabel('Ant Mass')
    plt.xlabel('Number of Node Executions')
    plt.title('Ant Mass in the Descending Link Direction')
    filename = 'node_M-_'+str(len(ant_graph.nodes()))+'nodes_'+str(len(link_names))+'edges'
    plt.savefig(folder_name+'/'+filename+time_string,format='PNG',dpi=100)

    # plt.show()
    plt.gcf().clear()

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
        print('After execution ' + str(i))
        data = ant_graph.node_execution()

        dist_ascend.append(data[0])
        dist_descend.append(data[1])
        mass_ascend.append(data[2])
        mass_descend.append(data[3])

    # process_data(ant_graph,dist_ascend,dist_descend,mass_ascend,mass_descend)

if __name__ == '__main__':
    main()
