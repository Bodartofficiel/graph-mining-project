import networkx as nx
from get_graphs import deezer_graph, lastfm_graph

# path = "deezer_europe/deezer_europe_edges.csv"
# deezer_edges = pd.read_csv(path)
# deezer_graph = nx.from_pandas_edgelist(deezer_edges, "node_1", "node_2")

def eigenvector_ranking(graph):
    ec_dic = nx.eigenvector_centrality(deezer_graph, max_iter=100, tol=1e-05)
    ec_ranking = {key: rank for rank, key in enumerate(sorted(ec_dic, key=ec_dic.get, reverse=True), 1)}
    return(ec_ranking)

deezer_ec_ranking = eigenvector_ranking(deezer_graph)
lastfm_ec_ranking = eigenvector_ranking(lastfm_graph)

