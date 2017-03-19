'''
Path Planning with ACO
5x5 node graph

Rough graph representation:
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O
|X|X|X|X|
O-O-O-O-O

where the lines (-,|,X) are the connecting links and the circles (O) are the nodes.

TO-DOs:
1. develop 5x5 graph -- halfway; still need to add the edges properly like the graph above
2. create ant objects
3. create tuple class instantiates and referenc to networkx object
    don't want to access networkx object globally

4. move ants?

'''
import networkx as nx
import sys
from matplotlib import pyplot as plt
import numpy as np

def initialize_graph(square_grid_number):

    int_coords = []
    for i in range(square_grid_number):
        for j in range(square_grid_number):
            int_coords.append([j,i])


    num_of_nodes = len(int_coords)
    init_phero = 0.001

    # Initialize dictionary with coordinates
    pos = dict(zip(range(1,num_of_nodes+1),int_coords))

    graph = nx.Graph()
    graph.add_nodes_from(pos.keys())

    for n,p in pos.items():
        graph.node[n]['name'] = n
        graph.node[n]['x'] = p[0]
        graph.node[n]['y'] = p[1]

    corner = [1,square_grid_number,(square_grid_number**2)-square_grid_number+1,square_grid_number**2]
    for i in range(1,num_of_nodes+1):
        list_of_neighbors = []

        if (i%square_grid_number != 0 and i%square_grid_number != 1 and
        i/square_grid_number < square_grid_number-1 and int(i/square_grid_number) > 0):
            list_of_neighbors.append(i-square_grid_number-1)    # SW
            list_of_neighbors.append(i-square_grid_number)      # S
            list_of_neighbors.append(i-square_grid_number+1)    # SE
            list_of_neighbors.append(i-1)                       # W
            list_of_neighbors.append(i+1)                       # E
            list_of_neighbors.append(i+square_grid_number-1)    # NW
            list_of_neighbors.append(i+square_grid_number)      # N
            list_of_neighbors.append(i+square_grid_number+1)    # NE
        elif i not in corner:
            if i%square_grid_number == 0: # right boundary
                list_of_neighbors.append(i-square_grid_number-1)# SW
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number-1)# NW
            elif i%square_grid_number == 1: # left boundary
                list_of_neighbors.append(i-square_grid_number+1)# SE
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number+1)# NE
            elif i/square_grid_number > square_grid_number-1: # upper boundary
                list_of_neighbors.append(i-square_grid_number-1)# SW
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-square_grid_number+1)# SE
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+1)                   # E
            else: # lower boundary
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i+square_grid_number-1)# NW
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number+1)# NE
        else: # corners
            if i==corner[0]: # bottom left
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number+1)# NE
            elif i==corner[1]: # bottom right
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i+square_grid_number)  # N
                list_of_neighbors.append(i+square_grid_number-1)# NW
            elif i==corner[2]: # upper left
                list_of_neighbors.append(i+1)                   # E
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-square_grid_number+1)# SE
            else: # upper right
                list_of_neighbors.append(i-1)                   # W
                list_of_neighbors.append(i-square_grid_number)  # S
                list_of_neighbors.append(i-square_grid_number-1)# SW

        for j in list_of_neighbors:
            distance = distance_calc(graph.node[i],graph.node[j])
            graph.add_edge(i,j,phero=init_phero,dist=distance)

    # Create plot with gridlines
    fig = plt.figure()
    axes = fig.gca()
    axes.set_xticks(np.arange(0,6,1))
    axes.set_yticks(np.arange(0,6,1))

    # Draw network on figure
    nx.draw_networkx(graph,pos,node_size=175,font_size=9,node_color='w')
    plt.grid('on')
    plt.show()

    return graph

def distance_calc(point1,point2):
    x1 = point1['x']
    x2 = point2['x']
    y1 = point1['y']
    y2 = point2['y']

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def main():
    initialize_graph(5)

if __name__ == '__main__':
    main()
