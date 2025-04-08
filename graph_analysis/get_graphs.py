import json
import os
from pathlib import Path

import networkx as nx

file_dir = Path(__file__).parent
lastfm_data_path = file_dir.parent / "lastfm_asia"
deezer_data_path = file_dir.parent / "deezer_europe"

assert os.path.isdir(
    deezer_data_path
), f"Deezer data path is missing: {deezer_data_path}"
assert os.path.isdir(
    lastfm_data_path
), f"LastFM data path is missing: {lastfm_data_path}"

lastfm_edges_path = lastfm_data_path / "lastfm_asia_edges.csv"
deezer_edges_path = deezer_data_path / "deezer_europe_edges.csv"
lastfm_nattr_path = lastfm_data_path / "lastfm_asia_features.json"
deezer_nattr_path = deezer_data_path / "deezer_europe_features.json"

assert os.path.exists(
    deezer_edges_path
), f"Deezer edges file is missing: {deezer_edges_path}"
assert os.path.exists(
    lastfm_edges_path
), f"LastFM edges file is missing: {lastfm_edges_path}"


def read_edges(file_path):
    edges = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            source, target = line.strip().split(",")
            edges.append((int(source), int(target)))
    return edges


lastfm_edges = read_edges(lastfm_edges_path)
deezer_edges = read_edges(deezer_edges_path)

deezer_graph: nx.graph.Graph = nx.from_edgelist(deezer_edges)
lastfm_graph: nx.graph.Graph = nx.from_edgelist(lastfm_edges)

with open(deezer_nattr_path, "r") as f:
    deezer_node_attributes = json.load(f)

with open(lastfm_nattr_path, "r") as f:
    lastfm_node_attributes = json.load(f)

for node_idx, node_features in deezer_node_attributes.items():
    deezer_graph.nodes[int(node_idx)]["feat"] = node_features

for node_idx, node_features in lastfm_node_attributes.items():
    lastfm_graph.nodes[int(node_idx)]["feat"] = node_features
