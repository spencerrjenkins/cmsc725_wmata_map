import numpy as np
from shapely.geometry import box

import community

import networkx as nx
import random
import math

def get_points(df, extremities, layers=8):
    ids = []
    if layers <= 0:
        return ids
    if df.shape[0] < 2:
        return ids
    df_sorted = df.sort_values("point_likelihood",ascending=False)
    top_point = df_sorted.iloc[1]
    ids.append(top_point.SID)
    ex_bl = [extremities[0], extremities[1], top_point["INTPTLON20"], top_point["INTPTLAT20"]]
    df_bl = df.iloc[df.sindex.query(box(*ex_bl))] # bottom left
    ex_br = [top_point["INTPTLON20"], extremities[1], extremities[2], top_point["INTPTLAT20"]]
    df_br = df.iloc[df.sindex.query(box(*ex_br))] # bottom right
    ex_tl = [extremities[0], top_point["INTPTLAT20"], top_point["INTPTLON20"], extremities[3]]
    df_tl = df.iloc[df.sindex.query(box(*ex_tl))] # top left
    ex_tr = [top_point["INTPTLON20"], top_point["INTPTLAT20"], extremities[2], extremities[3]]
    df_tr = df.iloc[df.sindex.query(box(*ex_tr))] # top right
    ids += get_points(df_bl, ex_bl, layers-1) + get_points(df_br, ex_br, layers-1) + get_points(df_tl, ex_tl, layers-1) + get_points(df_tr, ex_tr, layers-1)
    return ids
def contract_louvain_communities_with_positions(G, pos, resolution=1.0):
    # Detect communities using Louvain method with adjustable resolution
    partition = community.best_partition(G, resolution=resolution)

    # Group nodes by community
    community_nodes = {}
    for node, comm in partition.items():
        if comm not in community_nodes:
            community_nodes[comm] = []
        community_nodes[comm].append(node)

    # Compute new positions as the centroid of each community
    new_positions = {}
    for comm, nodes in community_nodes.items():
        x_vals = [pos[n][0] for n in nodes]
        y_vals = [pos[n][1] for n in nodes]
        new_positions[comm] = (np.mean(x_vals), np.mean(y_vals))

    # Contract the communities
    contracted_G = nx.Graph()
    for comm in community_nodes:
        contracted_G.add_node(comm)

    for u, v in G.edges():
        u_comm = partition[u]
        v_comm = partition[v]
        if u_comm != v_comm:
            contracted_G.add_edge(u_comm, v_comm)

    return contracted_G, new_positions

def angle_between(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return math.degrees(math.acos(max(min(cos_angle, 1), -1)))

def reduce_degree(graph, pos, max_degree=4, angle_threshold=10):
    for node in list(graph.nodes()):
        while graph.degree(node) > max_degree:
            def compute_angles():
                neighbors = list(graph.neighbors(node))
                angles = []
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v1 = (pos[neighbors[i]][0] - pos[node][0], pos[neighbors[i]][1] - pos[node][1])
                        v2 = (pos[neighbors[j]][0] - pos[node][0], pos[neighbors[j]][1] - pos[node][1])
                        angle = angle_between(v1, v2)
                        angles.append((angle, neighbors[i], neighbors[j]))
                return [a for a in angles if a[0] < angle_threshold]

            # Continuously remove edges with small angles, updating after each removal
            angles = compute_angles()
            angles.sort()

            while angles:
                _, n1, n2 = angles.pop(0)

                if graph.has_edge(node, n1) and graph.has_edge(node, n2):
                    if graph[node][n1]['weight'] > graph[node][n2]['weight']:
                        graph.remove_edge(node, n1)
                    else:
                        graph.remove_edge(node, n2)

                # Recompute angles after each edge removal
                angles = compute_angles()
                angles.sort()

            # Fallback to removing the edge with the largest weight if the degree is still too high
            if graph.degree(node) > max_degree:
                edges_with_weights = [(neighbor, graph[node][neighbor]['weight']) for neighbor in graph.neighbors(node)]
                edges_with_weights.sort(key=lambda x: x[1], reverse=True)
                while graph.degree(node) > max_degree and edges_with_weights:
                    neighbor_to_remove = edges_with_weights.pop(0)[0]
                    graph.remove_edge(node, neighbor_to_remove)

    return graph

def remove_isolated_nodes(graph):
    # Find all isolated nodes (degree 0)
    isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]
    
    # Remove isolated nodes from the graph
    graph.remove_nodes_from(isolated_nodes)
    
    return graph
def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude.
    Coordinates should be in (latitude, longitude) format.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Radius of Earth in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine calculation
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def assign_edge_weights(graph, positions):
    """
    Assign weights to edges in the graph based on the geographic distance
    between connected vertices.
    
    Parameters:
    - graph: A networkx graph object.
    - positions: A dictionary where keys are node names and values are (lat, lon) tuples.
    """
    for u, v in graph.edges():
        if u in positions and v in positions:
            coord1 = positions[u]
            coord2 = positions[v]
            distance = haversine(coord1, coord2)
            graph[u][v]['weight'] = distance

    return graph
def perform_walks(graph, pos, num_walks=5, max_distance=200000):
    traversed_edges = set()
    intersections = {}

    def get_straightest_edge(node, prev_node, visited):
        neighbors = [n for n in graph.neighbors(node) if (node, n) not in traversed_edges and (n, node) not in traversed_edges and n not in visited]
        if not neighbors:
            return None

        if prev_node is None:
            return random.choice(neighbors)

        v1 = (pos[node][0] - pos[prev_node][0], pos[node][1] - pos[prev_node][1])

        def angle_to_prev(n):
            v2 = (pos[n][0] - pos[node][0], pos[n][1] - pos[node][1])
            return abs(angle_between(v1, v2))

        return min(neighbors, key=angle_to_prev)

    walks = []

    for _ in range(num_walks):
        start_node = random.choice(list(graph.nodes()))
        walk = [start_node]
        prev_node = None
        current_distance = 0

        while current_distance < max_distance:
            next_node = get_straightest_edge(walk[-1], prev_node, set(walk))

            if next_node is None:
                break

            edge = (walk[-1], next_node)

            # Check for intersection
            if edge in traversed_edges or (edge[1], edge[0]) in traversed_edges:
                if intersections.get(edge, 0) >= 1:
                    break
                intersections[edge] = intersections.get(edge, 0) + 1

            traversed_edges.add(edge)
            walk.append(next_node)
            prev_node = walk[-2]

            current_distance += graph[walk[-2]][walk[-1]]['weight']

        walks.append(walk)

    return walks
def plot_walks(graph, pos, walks, ax):

    # Draw the walks
    def random_hex_color():
        return "#" + ''.join(random.choices('0123456789ABCDEF', k=6))

    for i, walk in enumerate(walks):
        walk_edges = [(walk[j], walk[j + 1]) for j in range(len(walk) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=walk_edges, edge_color=random_hex_color(), width=2, ax=ax)
    return ax