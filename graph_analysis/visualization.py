import networkx as nx
import numpy as np 
from get_graphs import lastfm_graph
import matplotlib.pyplot as plt

degree_centrality = nx.degree_centrality(lastfm_graph)

threshold = 30
selected_nodes = [node for node, degree in lastfm_graph.degree() if degree > threshold]
subgraph = lastfm_graph.subgraph(selected_nodes)
centrality = np.array([degree_centrality[node] for node in selected_nodes])
print("subgraph created.", subgraph)


# pos = nx.kamada_kawai_layout(subgraph)
# print("positions calculated.")

# nx.draw_networkx(subgraph, pos=pos, node_color=centrality, node_size=centrality*2e3,with_labels=False)
# plt.show()


### PYDOT ###
prog = 'neato'
root = None
pos = nx.nx_pydot.graphviz_layout(subgraph,prog,root)
# pos = nx.nx_pydot.pydot_layout(subgraph)
plt.figure()    
nx.draw(subgraph,pos,edge_color='black',width=1,linewidths=1, node_size=10,node_color='blue',alpha=0.9)
plt.axis('on')
plt.show()


### GRAPHVIZ ###
# A = nx.nx_agraph.to_agraph(subgraph)
# A.draw("k5.png", prog="neato")