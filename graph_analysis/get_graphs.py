import json
import os
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

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
max_deezer_features = (
    max(
        [
            max(features)
            for features in deezer_node_attributes.values()
            if len(features) > 0
        ]
    )
    + 1
)
max_lastfm_features = (
    max(
        [
            max(features)
            for features in lastfm_node_attributes.values()
            if len(features) > 0
        ]
    )
    + 1
)


# print(max_deezer_features)
# print(max_lastfm_features)

# Get max number of features for padding
deezer_max_len = max([len(features) for features in deezer_node_attributes.values()])
lastfm_max_len = max([len(features) for features in lastfm_node_attributes.values()])

for node, feature in deezer_node_attributes.items():
    deezer_graph.nodes[int(node)]["x"] = torch.tensor(
        feature + [max_deezer_features] * (deezer_max_len - len(feature))
    )

for node, feature in lastfm_node_attributes.items():
    lastfm_graph.nodes[int(node)]["x"] = torch.tensor(
        feature + [max_lastfm_features] * (lastfm_max_len - len(feature))
    )

# print(deezer_max_len)
# print(lastfm_max_len)


# class EmbeddingModel(nn.Module):
#     def __init__(self, num_songs, embedding_dim):
#         super(EmbeddingModel, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings=num_songs, embedding_dim=embedding_dim
#         )
#
#     def forward(self, x):
#         return self.embedding(x)
#
#
# embedding_dim = 50
# deezer_embedding_model = EmbeddingModel(max_deezer_features, embedding_dim)
# lastfm_embedding_model = EmbeddingModel(max_lastfm_features, embedding_dim)
#
# for node_idx, node_features in deezer_node_attributes.items():
#     node_padded_features = node_features + [0] * (deezer_max_len - len(node_features))
#     padded_vector = torch.LongTensor(node_padded_features)
#     with torch.no_grad():
#         try:
#             embeddings = deezer_embedding_model(padded_vector)
#         except:
#             print(node_idx, padded_vector, torch.max(padded_vector), len(padded_vector))
#             os._exit(0)
#     deezer_graph.nodes[int(node_idx)]["feat"] = embeddings
#
# for node_idx, node_features in lastfm_node_attributes.items():
#     node_padded_features = node_features + [0] * (deezer_max_len - len(node_features))
#     padded_vector = torch.LongTensor(node_padded_features)
#     with torch.no_grad():
#         embeddings = lastfm_embedding_model(padded_vector)
#     deezer_graph.nodes[int(node_idx)]["feat"] = embeddings


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
    deezer_graph.nodes[node_idx]["y"] = target

for node_idx, target in lastfm_targets.items():
    lastfm_graph.nodes[node_idx]["y"] = target

max_lastfm_target = max(lastfm_targets.values())
print(f"Max target in LastFM: {max_lastfm_target}")

deezer_pyg = from_networkx(deezer_graph)
lastfm_pyg = from_networkx(lastfm_graph)
