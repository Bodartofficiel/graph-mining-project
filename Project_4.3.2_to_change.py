def greedy_routing(embedding_wrapper, source, target):
    """
    Takes steps from euclidiean distance in embedding space until target is found or next_step is not a neighbor of current_node.
    """
    embedding = [embedding_wrapper.get_embedding(node) for node in used_graph.nodes()]
    visited = {source}
    current = source
    steps = 0
    no_new = 0
    while current != target:
        distances = np.linalg.norm(embedding - embedding[embedding_wrapper.node_to_i[current]], axis=1)
        distances = np.delete(distances,embedding_wrapper.node_to_i[current])
        next_node = np.argmin(distances)
        next_node = embedding_wrapper.i_to_node[next_node]
        if (not next_node in used_graph.neighbors(current)) or no_new==10:
            return False, steps  # Failure if no direct edge
        current = next_node
        steps += 1

        if current in visited:
            no_new += 1
        else :
            no_new = 0
        visited.add(current)
    return True, steps

def random_k_away_neighbor(k,source,graph):
    for _ in range(k):
        neigbors = list(graph.neighbors(source))
        if neigbors:
            source = random.choice(neigbors)
        else :
            source = random.choice(list(used_graph.nodes()))
    return(source)

steps_node2vec, steps_laplacian = 0, 0
for _ in range(1000):
    source = random.choice(list(used_graph.nodes()))
    target = random_k_away_neighbor(5,source,used_graph)
    success_node2vec, current_steps_node2vec = greedy_routing(node2vec_embedding_wrapper, source, target)
    success_laplacian, current_steps_laplacian = greedy_routing(laplacian_embedding_wrapper, source, target)
    steps_node2vec += current_steps_node2vec
    steps_laplacian += current_steps_laplacian

print(f"Node2vec Greedy Routing completed {steps_node2vec} correct steps.")
print(f"Laplacian Eigenmaps Greedy Routing completed {steps_laplacian} correct steps.")