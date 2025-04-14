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
lastfm_targt_path = lastfm_data_path / "lastfm_asia_target.csv"
deezer_targt_path = deezer_data_path / "deezer_europe_target.csv"

assert os.path.exists(
    deezer_edges_path
), f"Deezer edges file is missing: {deezer_edges_path}"
assert os.path.exists(
    lastfm_edges_path
), f"LastFM edges file is missing: {lastfm_edges_path}"
assert os.path.exists(
    deezer_nattr_path
), f"Deezer node attributes file is missing: {deezer_nattr_path}"
assert os.path.exists(
    lastfm_nattr_path
), f"LastFM node attributes file is missing: {lastfm_nattr_path}"
assert os.path.exists(
    deezer_targt_path
), f"Deezer target file is missing: {deezer_targt_path}"
assert os.path.exists(
    lastfm_targt_path
), f"LastFM target file is missing: {lastfm_targt_path}"


### Create edges
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

### Node features

with open(deezer_nattr_path, "r") as f:
    deezer_node_attributes = json.load(f)

with open(lastfm_nattr_path, "r") as f:
    lastfm_node_attributes = json.load(f)

# Get max of values of edge features
max_deezer_features = max(
    [max(features) for features in deezer_node_attributes.values() if len(features) > 0]
)
max_lastfm_features = max(
    [max(features) for features in lastfm_node_attributes.values() if len(features) > 0]
)
# print(max_deezer_features)
# print(max_lastfm_features)

for node_idx, node_features in deezer_node_attributes.items():
    one_hot_encoded = [0] * (max_deezer_features + 1)
    for feature in node_features:
        one_hot_encoded[feature] = 1
    deezer_graph.nodes[int(node_idx)]["feat"] = one_hot_encoded

for node_idx, node_features in lastfm_node_attributes.items():
    one_hot_encoded = [0] * (max_lastfm_features + 1)
    for feature in node_features:
        one_hot_encoded[feature] = 1
    lastfm_graph.nodes[int(node_idx)]["feat"] = one_hot_encoded


### Target
def read_targets(file_path):
    targets = {}
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            node, target = line.strip().split(",")
            targets[int(node)] = int(target)
    return targets


deezer_targets = read_targets(deezer_targt_path)
lastfm_targets = read_targets(lastfm_targt_path)

for node_idx, target in deezer_targets.items():
    deezer_graph.nodes[node_idx]["target"] = target

for node_idx, target in lastfm_targets.items():
    lastfm_graph.nodes[node_idx]["target"] = target
