import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN

# Sample data: List of (x, y) coordinates
points = np.array([
    [1, 2], [1.1, 2.1], [2, 2], [8, 8], [8.1, 8], [7.9, 8.2], [50, 50]
])

# Step 1: Build a Graph (fully connected or k-NN)
G = nx.Graph()
for i, (x, y) in enumerate(points):
    G.add_node(i, pos=(x, y))

# Connect nodes within a threshold distance (proximity graph)
threshold = 1.5  # Adjust based on your data
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        if np.linalg.norm(points[i] - points[j]) < threshold:
            G.add_edge(i, j)

# Step 2: Cluster using DBSCAN on spatial data
dbscan = DBSCAN(eps=1.5, min_samples=2)  # Adjust `eps` as needed
labels = dbscan.fit_predict(points)

# Step 3: Merge clusters in the graph
cluster_dict = {}
for i, label in enumerate(labels):
    if label != -1:  # Ignore noise points (-1)
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(i)

# Contract nodes in each cluster into a single representative
merged_G = nx.Graph()
for label, nodes in cluster_dict.items():
    # Merge cluster nodes into a single "super node"
    cluster_center = np.mean(points[nodes], axis=0)
    merged_G.add_node(label, pos=tuple(cluster_center))

# Step 4: Visualize the Original vs. Merged Graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Original Graph
plt.subplot(1, 2, 1)
pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
plt.title("Original Graph")

# Merged Graph
plt.subplot(1, 2, 2)
merged_pos = nx.get_node_attributes(merged_G, "pos")
nx.draw(merged_G, merged_pos, with_labels=True, node_color="red")
plt.title("Merged Clusters")

plt.show()
