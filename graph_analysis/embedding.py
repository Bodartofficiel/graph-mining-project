from get_graphs import deezer_graph, lastfm_graph
import numpy as np
import networkx as nx 
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

### Spectral Embedding ###
percent_nodes = 0.2
n_nodes = int(lastfm_graph.number_of_nodes() * percent_nodes)
print(f"Selected nodes: {n_nodes} ({percent_nodes*100:.0f}%)")

# Random walk sampling
def random_walk_sampling(graph, start_node, walk_len, damping_factor=0.15):
    visited = {start_node}
    current_node = start_node
    for _ in range(walk_len):
        neighbors = list(graph.neighbors(current_node))
        if np.random.random()>damping_factor and neighbors:
            current_node = np.random.choice(neighbors)
            visited.add(current_node)
        else :
            current_node = np.random.choice(graph.nodes)
    return(visited)

random_nodes = random_walk_sampling(lastfm_graph,start_node=np.random.choice(lastfm_graph.nodes),walk_len=n_nodes)
subgraph = lastfm_graph.subgraph(random_nodes)
print("Subgraph created!")


A = nx.adjacency_matrix(subgraph).toarray()
D = np.diag(A.sum(axis=1)) 

# Pour éviter une division par 0
epsilon = 1e-6
D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal() + epsilon))
I = np.identity(len(D))

# Laplacian normalisé
L = I - D_inv_sqrt @ A @ D_inv_sqrt

# Laplacian
# L = D - A

# Valeurs et vecteurs propres
vals, vecs = np.linalg.eig(L)

tolerance = 1e-14
vals = np.real(vals)
vals[np.abs(vals) < tolerance] = 0
vecs = np.real(vecs)
vecs[np.abs(vecs) < tolerance] = 0

def k_smallest_eigvals(k, vals):
    nonzero_vals = [(i, val) for i, val in enumerate(vals) if val > 0]
    sorted_vals = sorted(nonzero_vals, key=lambda x: x[1])
    return sorted_vals[:k]

def get_U(k,vals,vecs):
    k_eigvals = k_smallest_eigvals(k,vals)
    k_eigvecs = [vecs[:,i] for (i,_) in k_eigvals]
    return(np.array(k_eigvecs).T)

# K-Means clustering with silhouette analysis
print("Running K-Means clustering with silhouette analysis...")
scores = []
explored_range = np.arange(n_nodes//100, n_nodes//10, 5)
print(f"Explored range [{n_nodes//100},{n_nodes//10}]")

# Pre-compute embeddings for each dimension to avoid redundant calculations
embeddings = {}
for k in tqdm(explored_range, desc="Computing embeddings and silhouette scores"):
    # if k not in embeddings:
    embeddings[k] = get_U(k, vals, vecs)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings[k])
    score = silhouette_score(embeddings[k], kmeans.labels_)
    scores.append(score)

# Find optimal k (maximum silhouette score)
optimal_k = explored_range[np.argmax(scores)]
max_score = max(scores)

# Smoothing for visualization
scores_smooth = gaussian_filter1d(scores, sigma=2)

# Create an informative visualization with seaborn

plt.figure(figsize=(7, 4))
sns.set_style("whitegrid")
sns.lineplot(x=explored_range, y=scores, marker='o', label='Silhouette Score', alpha=0.7)
sns.lineplot(x=explored_range, y=scores_smooth, linewidth=2.5, label='Smoothed Score', color='royalblue')

# Highlight the optimal k
# plt.axvline(x=optimal_k, color='crimson', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
# plt.axhline(y=max_score, color='crimson', linestyle=':', linewidth=1.5, alpha=0.6)

# Annotate the optimal point
# plt.scatter([optimal_k], [max_score], s=150, color='crimson', zorder=5, edgecolor='white')
# plt.annotate(f'({optimal_k}, {max_score:.4f})', 
#              xy=(optimal_k, max_score),
#              xytext=(10, 10),
#              textcoords='offset points',
#              fontsize=12,
#              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Analysis for K-Means Clustering', fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
# plt.savefig('kmeans_silhouette_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Use the optimal k for final clustering
print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max_score:.4f})")
# optimal_embedding = embeddings[optimal_k]
# final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(optimal_embedding)


# DBSCAN
# clustering = DBSCAN(eps=3, min_samples=2).fit(U)
# print('DBSCAN')
# print(np.unique(clustering.labels_))
# print(clustering.labels_[:10])
# print(silhouette_score(U,clustering.labels_))
