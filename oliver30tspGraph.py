import networkx as nx
from matplotlib import pyplot as plt
import numpy
import sys
import re

# Read coordinates of oliver30Coords.txt
opened_file = open("oliver30Coords.txt","r")
read_coords = opened_file.read()
read_coords = re.split('\n|,',read_coords)

# Convert coordinate strings to integers
int_coords = [[int(read_coords[i]),int(read_coords[i+1])] for i in
range(len(read_coords)//2)]

# Initialize dictionary with coordinates
pos = dict(zip(range(1,len(int_coords)+1),int_coords))

oliver30 = nx.Graph()
oliver30.add_nodes_from(pos.keys())

for n,p in pos.items():
    oliver30.node[n]['pos'] = p

# Create plot with gridlines
fig = plt.figure()
axes = fig.gca()
axes.set_xticks(numpy.arange(0,101,2))
axes.set_yticks(numpy.arange(0,101,2))

# Draw network on figure
nx.draw_networkx(oliver30,pos,node_size=175,font_size=9,node_color='w')
plt.grid('on')
plt.title('Oliver 30 Network')
plt.show()
