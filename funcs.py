import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import contextily as cx
from pyproj import Transformer, Geod

from matplotlib.patches import Circle
import community

import networkx as nx
import random
import math


def load_shapefile(filepath, crs="EPSG:4326"):
    """Load a shapefile and convert to the specified CRS."""
    try:
        gdf = gpd.read_file(filepath)
        return gdf.to_crs(crs)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_county_codes(fips_path, states, county_names):
    """Return FIPS codes for given states and county names."""
    fips_df = pd.read_csv(fips_path)
    fips_df = fips_df[fips_df["state"].isin(states)]
    fips_df["code"] = fips_df["fips"].apply(lambda a: str(a)[-3:])
    codes = fips_df[fips_df["name"].isin(county_names)]["code"]
    return codes


def compute_transit_potential(df):
    """Compute transit potential score for each block."""
    df["transit_potential"] = np.log(
        df["POP20"] / (df["ALAND20"] + df["AWATER20"]) * 1000 + 1
    )
    return df


def save_geojson(df, path):
    """Save GeoDataFrame to GeoJSON."""
    df.to_file(path, driver="GeoJSON")


def load_geojson(path):
    """Load GeoJSON as GeoDataFrame."""
    return gpd.read_file(path)


def reset_and_concat(*dfs):
    """Reset index and concatenate multiple GeoDataFrames."""
    dfs = [df.reset_index(drop=True) for df in dfs]
    return pd.concat(dfs, ignore_index=True)


def plot_kde_heatmap(df_points, bandwidth=2000, grid_size=70, cmap="Reds"):
    df_points_web = df_points.to_crs(epsg=3857)
    coords_web = np.vstack([df_points_web.geometry.x, df_points_web.geometry.y]).T
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords_web)
    minx, miny, maxx, maxy = df_points_web.total_bounds
    x_grid = np.linspace(minx, maxx, grid_size)
    y_grid = np.linspace(miny, maxy, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    log_dens = kde.score_samples(grid_coords)
    density = np.exp(log_dens).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(np.flipud(density), extent=[minx, maxx, miny, maxy], cmap=cmap, alpha=0.6)
    df_points_web.plot(ax=ax, color="blue", markersize=10, label="Data Points")
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    plt.colorbar(ax.images[0], label="Density")
    plt.title("Kernel Density Estimation (KDE) Heatmap")
    plt.legend()
    plt.show()
    return kde


def get_points(df, extremities, layers=8):
    ids = []
    if layers <= 0:
        return ids
    if df.shape[0] < 2:
        return ids
    df_sorted = df.sort_values("point_likelihood", ascending=False)
    top_point = df_sorted.iloc[1]
    ids.append(top_point.SID)
    ex_bl = [
        extremities[0],
        extremities[1],
        top_point["INTPTLON20"],
        top_point["INTPTLAT20"],
    ]
    df_bl = df.iloc[df.sindex.query(box(*ex_bl))]  # bottom left
    ex_br = [
        top_point["INTPTLON20"],
        extremities[1],
        extremities[2],
        top_point["INTPTLAT20"],
    ]
    df_br = df.iloc[df.sindex.query(box(*ex_br))]  # bottom right
    ex_tl = [
        extremities[0],
        top_point["INTPTLAT20"],
        top_point["INTPTLON20"],
        extremities[3],
    ]
    df_tl = df.iloc[df.sindex.query(box(*ex_tl))]  # top left
    ex_tr = [
        top_point["INTPTLON20"],
        top_point["INTPTLAT20"],
        extremities[2],
        extremities[3],
    ]
    df_tr = df.iloc[df.sindex.query(box(*ex_tr))]  # top right
    ids += (
        get_points(df_bl, ex_bl, layers - 1)
        + get_points(df_br, ex_br, layers - 1)
        + get_points(df_tl, ex_tl, layers - 1)
        + get_points(df_tr, ex_tr, layers - 1)
    )
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


def reduce_degree(graph, pos, max_degree=4, angle_threshold=10):
    for node in list(graph.nodes()):
        while graph.degree(node) > max_degree:

            def compute_angles():
                neighbors = list(graph.neighbors(node))
                angles = []
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v1 = (
                            pos[neighbors[i]][0] - pos[node][0],
                            pos[neighbors[i]][1] - pos[node][1],
                        )
                        v2 = (
                            pos[neighbors[j]][0] - pos[node][0],
                            pos[neighbors[j]][1] - pos[node][1],
                        )
                        angle = angle_between(v1, v2)
                        angles.append((angle, neighbors[i], neighbors[j]))
                return [a for a in angles if a[0] < angle_threshold]

            # Continuously remove edges with small angles, updating after each removal
            angles = compute_angles()
            angles.sort()

            while False and angles:
                _, n1, n2 = angles.pop(0)

                if graph.has_edge(node, n1) and graph.has_edge(node, n2):
                    if graph[node][n1]["weight"] > graph[node][n2]["weight"]:
                        graph.remove_edge(node, n1)
                    else:
                        graph.remove_edge(node, n2)

                # Recompute angles after each edge removal
                angles = compute_angles()
                angles.sort()

            # Fallback to removing the edge with the largest weight if the degree is still too high
            if graph.degree(node) > max_degree:
                edges_with_weights = [
                    (neighbor, graph[node][neighbor]["weight"])
                    for neighbor in graph.neighbors(node)
                ]
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


def haversine(pt1, pt2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude.
    Coordinates should be in (latitude, longitude) format.
    """
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon1, lat1 = transformer.transform(*pt1)
    lon2, lat2 = transformer.transform(*pt2)

    # Geodesic distance using WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)

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
            graph[u][v]["weight"] = distance

    return graph


def angle_between(v1, v2):
    """Calculate the angle in degrees between two vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(min(cos_theta, 1), -1)  # Clamp for numerical stability
    return math.degrees(math.acos(cos_theta))

def deviation_between(v1, v2):
    """
    Compute the signed angle (in degrees) from -v1 to v2.
    Negative if v2 is to the left of -v1, positive if to the right.
    """
    # Negate v1
    """Calculate the signed angle in degrees between two vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    # Replace acos-based angle with atan2 for signed result (1 line change)
    angle = math.atan2(v1[0]*v2[1] - v1[1]*v2[0], dot)
    degrees = math.degrees(angle)
    if degrees > 0:
        return 180 - degrees
    else:
        return -180 - math.degrees(angle)

def perform_walks(
    graph,
    pos,
    num_walks=5,
    min_distance=0,
    max_distance=200000,
    traversed_edges=set(),
    complete_traversed_edges=[],
    min_angle=110,
):
    def get_straightest_edge(node, prev_node, visited, sign):
        neighbors = [
            n
            for n in graph.neighbors(node)
            if (node, n) not in traversed_edges and n not in visited
        ]
        if not neighbors:
            return None, None, None

        if prev_node is None:
            return random.choice(neighbors), 0, 0

        v1 = (pos[prev_node][0] - pos[node][0], pos[prev_node][1] - pos[node][1])

        # Compute angles and filter by angle
        candidates = []
        for n in neighbors:
            edge = (node, n)
            v2 = (pos[n][0] - pos[node][0], pos[n][1] - pos[node][1])
            # plt.scatter(pos[prev_node][0], pos[prev_node][1])
            # plt.scatter(pos[node][0], pos[node][1], c='black')
            # plt.scatter(pos[n][0], pos[n][1])
            # plt.show()
            # plt.clf()
            angle = angle_between(v1, v2)
            deviation = deviation_between(v1,v2)
            if angle > min_angle and (not sign or deviation * sign > 0):
                candidates.append((n, deviation, angle))

        if not candidates:
            return None, None, None

        # Choose the neighbor with the largest angle
        argmax = candidates.index(max(candidates, key=lambda x: x[2]))
        return candidates[argmax]

    walks = []

    i = 0
    timeout = 100
    while i < num_walks and timeout > 0:
        if len(complete_traversed_edges) < i + 1:
            complete_traversed_edges.append(set())
        start_node = random.choice(list(graph.nodes()))
        walk = [start_node]
        prev_node = None
        current_distance = 0
        curr_traversed_edges = set()
        total_turn = 0
        requested_sign = 0

        while current_distance < max_distance:
            next_node, deviation, angle = get_straightest_edge(walk[-1], prev_node, set(walk), requested_sign)

            if next_node is None:
                break

            edge = (walk[-1], next_node)
            curr_traversed_edges.add(edge)
            curr_traversed_edges.add((edge[1], edge[0]))
            # complete_traversed_edges[i].add(edge)
            # complete_traversed_edges[i].add((edge[1],edge[0]))
            walk.append(next_node)
            total_turn += deviation
            #print(deviation, total_turn)
            if total_turn > 100:
                # request negative
                requested_sign = -1
            elif total_turn < -100:
                requested_sign = 1
            else:
                requested_sign = 0
            prev_node = walk[-2]

            current_distance += graph[walk[-2]][walk[-1]]["weight"]

        if current_distance > min_distance:
            walks.append(walk)
            traversed_edges = set.union(traversed_edges, curr_traversed_edges)
            complete_traversed_edges[i] = curr_traversed_edges
            i += 1
        else:
            timeout -= 1

    return walks, traversed_edges, complete_traversed_edges


def score_node(node, positions, kde, radius=2000):
    node_pos = np.array(positions[node]).reshape(1, -1)
    # Sample points in a circle around the node
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_points = node_pos + radius * np.c_[np.cos(angles), np.sin(angles)]
    points = np.vstack([node_pos, circle_points])
    log_dens = kde.score_samples(points)
    return np.mean(np.exp(log_dens)) * 1e10


def score_walk_by_kde(walk, positions, kde, radius=2000):
    """
    Scores a walk by summing the KDE density within a given radius of each node in the walk.

    Parameters:
        walk (list): List of node IDs in the walk.
        positions (dict): Mapping from node ID to (x, y) coordinates (in same CRS as KDE).
        kde (KernelDensity): Fitted sklearn KernelDensity object.
        radius (float): Radius (in same units as positions) to consider around each node.

    Returns:
        float: Total KDE score for the walk.
    """

    score = 0.0
    for node in walk:
        score += score_node(node, positions, kde, radius)
    return score


def plot_walks(graph, pos, walks, ax, kde, bounds, radius=None):
    # Draw the walks
    def random_hex_color():
        return "#" + "".join(random.choices("0123456789ABCDEF", k=6))

    for walk in walks:
        walk_edges = [(walk[j], walk[j + 1]) for j in range(len(walk) - 1)]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=walk_edges,
            edge_color=random_hex_color(),
            width=2,
            ax=ax,
        )
        if radius is not None and type(radius) in [int, float]:
            for node in walk:
                x, y = pos[node]
                # Draw a circle in map units (EPSG:3857 is meters)
                circle = Circle(
                    (x, y),
                    radius=radius,
                    edgecolor="orange",
                    facecolor="none",
                    linewidth=1.5,
                    alpha=0.5,
                )
                ax.add_patch(circle)
                if x > bounds[0] and x < bounds[2] and y > bounds[1] and y < bounds[3]:
                    ax.text(
                        x,
                        y,
                        str(round(score_node(node, pos, kde, radius), 2)),
                        color="black",
                        fontsize=8,
                        ha="center",
                        va="center",
                        zorder=10,
                    )
    return ax


def replace_lowest_scoring_walk(
    walks,
    positions,
    kde,
    graph,
    traversed_edges,
    complete_traversed_edges,
    min_distance=0,
    max_distance=200000,
    radius=2000,
):
    """
    Removes the walk with the lowest KDE score from the list and adds a new walk using perform_walks.

    Parameters:
        walks (list of list): List of walks (each walk is a list of node IDs).
        positions (dict): Mapping from node ID to (x, y) coordinates.
        kde (KernelDensity): Fitted sklearn KernelDensity object.
        graph: networkx graph object.
        pos: dict of node positions (same as positions).
        min_distance (float): Minimum distance for new walk.
        max_distance (float): Maximum distance for new walk.
        radius (float): Radius for KDE scoring.

    Returns:
        list: Updated list of walks.
    """
    if not walks:
        return walks, traversed_edges, complete_traversed_edges

    # Score all walks
    scores = [score_walk_by_kde(walk, positions, kde, radius) for walk in walks]
    min_idx = int(np.argmin(scores))

    complete_traversed_edges = (
        complete_traversed_edges[:min_idx] + complete_traversed_edges[min_idx + 1 :]
    )
    traversed_edges = set.union(*complete_traversed_edges)
    # Remove the lowest scoring walk
    walks = walks[:min_idx] + walks[min_idx + 1 :]

    # Generate a new walk using your perform_walks implementation
    new_walks, traversed_edges, complete_traversed_edges = perform_walks(
        graph,
        positions,
        num_walks=1,
        min_distance=min_distance,
        max_distance=max_distance,
        traversed_edges=traversed_edges,
        complete_traversed_edges=complete_traversed_edges,
    )
    if new_walks:
        walks.append(new_walks[0])

    return walks, traversed_edges, complete_traversed_edges


def save_graph_to_geojson(graph, positions, out_path):
    """
    Save the graph (nodes and edges) to a GeoJSON file.
    Nodes are saved as Point features, edges as LineString features.
    All coordinates are converted to EPSG:4326.
    """
    from shapely.geometry import Point, LineString
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer

    # Assume input positions are in EPSG:3857 (Web Mercator) or another CRS
    # If you know the input CRS, set it here. For now, assume EPSG:3857
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def to_latlon(coord):
        # coord: (x, y) in input CRS
        lon, lat = transformer.transform(coord[0], coord[1])
        return (lon, lat)

    node_features = []
    for node, pos in positions.items():
        latlon = to_latlon(pos)
        node_features.append({"geometry": Point(latlon), "type": "node", "id": node})
    edge_features = []
    for u, v in graph.edges():
        if u in positions and v in positions:
            latlon_u = to_latlon(positions[u])
            latlon_v = to_latlon(positions[v])
            edge_features.append(
                {
                    "geometry": LineString([latlon_u, latlon_v]),
                    "type": "edge",
                    "source": u,
                    "target": v,
                }
            )
    features = node_features + edge_features
    gdf = gpd.GeoDataFrame(features)
    gdf.to_file(out_path, driver="GeoJSON")


def save_lines_to_geojson(
    lines, graph, positions, kde, out_path, node_station_status=None
):
    """
    Save the transit lines to a GeoJSON file, including KDE value at each vertex.
    Each line is a LineString feature, with a list of KDE values as a property.
    Also saves the length of each segment in a 'segment_lengths' property.
    Also saves a list of booleans 'is_station' for each node if node_station_status is provided.
    All coordinates are converted to EPSG:4326.
    """
    from shapely.geometry import LineString
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer
    import numpy as np

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def to_latlon(coord):
        lon, lat = transformer.transform(coord[0], coord[1])
        return (lon, lat)

    def get_kde_value(node):
        if node in positions:
            return float(score_node(node, positions, kde))
        return None

    def segment_length(c1, c2):
        # Calculate Euclidean distance in projected CRS (meters)
        return float(np.linalg.norm(np.array(c2) - np.array(c1)))

    features = []
    for idx, line in enumerate(lines):
        coords = [to_latlon(positions[n]) for n in line if n in positions]
        kde_values = [get_kde_value(n) for n in line if n in positions]
        segment_lengths = [
            segment_length(positions[line[i]], positions[line[i + 1]])
            for i in range(len(line) - 1)
            if line[i] in positions and line[i + 1] in positions
        ]
        is_station = None
        if node_station_status is not None:
            is_station = [
                bool(node_station_status.get(n, True)) for n in line if n in positions
            ]
        feature = {
            "geometry": LineString(coords),
            "type": "line",
            "line_id": idx,
            "kde_values": kde_values,
            "segment_lengths": segment_lengths,
        }
        if is_station is not None:
            feature["is_station"] = is_station
        features.append(feature)
    gdf = gpd.GeoDataFrame(features)
    gdf.to_file(out_path, driver="GeoJSON")


def load_lines_from_geojson(path):
    """
    Loads lines and positions from a GeoJSON file created by save_lines_to_geojson.
    Returns:
        lines: list of lists of node coordinates (as tuples)
        positions: dict mapping node coordinate (tuple) to (x, y) coordinates
    """
    import geojson

    with open(path, "r") as f:
        gj = geojson.load(f)
    lines = []
    positions = {}
    for feature in gj["features"]:
        coords = feature["geometry"]["coordinates"]
        line = []
        for c in coords:
            node = tuple(c)
            line.append(node)
            positions[node] = node  # Store as (lon, lat)
        lines.append(line)
    return lines, positions


def mark_station_nodes(walks, graph, positions, min_station_dist=1000):
    """
    Mark nodes as stations or non-stations based on:
    - Terminal stations (endpoints of a line) are always stations.
    - Transfer stations (nodes shared by two or more lines) are always stations.
    - Any other node is marked as a non-station if it is less than min_station_dist (meters) away from the previous STATION node in the walk.
    Returns: dict {node: True/False}
    """
    from collections import defaultdict

    station_nodes = set()
    node_line_count = defaultdict(int)
    # Count how many lines each node appears in
    for walk in walks:
        for node in walk:
            node_line_count[node] += 1
    # Mark terminals and transfers as stations
    for walk in walks:
        n = len(walk)
        for i, node in enumerate(walk):
            if i == 0 or i == n - 1:
                station_nodes.add(node)
            elif node_line_count[node] > 1:
                station_nodes.add(node)
    node_station_status = {}
    for a, walk in enumerate(walks):
        n = len(walk)
        prev_station_node = None
        prev_node = None
        for i, node in enumerate(walk):
            #if a == 10 and i > 0:
            #        print(a, i, haversine(positions[walk[i]], positions[walk[i-1]]))
            if node in station_nodes:
                node_station_status[node] = True
                prev_station_node = node
            else:
                c=2
                is_station = True
                curr_prev_node = prev_node
                curr_node = node
                total_distance = 0
                while curr_node != prev_station_node:
                    if curr_prev_node in graph[curr_node]:
                        total_distance += graph[curr_node][curr_prev_node].get("weight", None)
                    elif curr_node in graph[curr_prev_node]:
                        total_distance += graph[curr_prev_node][curr_node].get("weight", None)
                    else:
                        #print(" - ", c, haversine(positions[curr_node], positions[curr_prev_node]))
                        total_distance += haversine(positions[curr_node], positions[curr_prev_node])
                    curr_node = curr_prev_node
                    curr_prev_node = walk[i-c]
                    c+=1
                if total_distance < min_station_dist:
                    is_station = False
                node_station_status[node] = is_station
                if is_station:
                    prev_station_node = node
            prev_node = node
    return node_station_status
