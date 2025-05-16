import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from shapely.geometry import LineString, Point

# Color palette (WMATA style, fallback to matplotlib colors if more needed)
COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
    "#bcf60c",  # lime
    "#fabebe",  # pink
    "#008080",  # teal
    "#e6beff",  # lavender
    "#9a6324",  # brown
    "#800000",  # maroon
    "#aaffc3",  # mint
    "#808000",  # olive
    "#ffd8b1",  # peach
    "#000075",  # navy
    "#808080",  # gray
    "#000000",  # black
    "#a9a9a9",  # dark gray
    "#ff4500",  # orange red
    "#2e8b57",  # sea green
    "#1e90ff",  # dodger blue
    "#ff69b4",  # hot pink
    "#7cfc00",  # lawn green
    "#8a2be2",  # blue violet
    "#00ced1",  # dark turquoise
]


def load_lines_geojson(path):
    with open(path, "r") as f:
        gj = json.load(f)
    features = gj["features"]
    lines = []
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        group = feat["properties"].get("group", 0)
        name_list = feat["properties"].get("name_list", [])
        line_id = feat["properties"].get("line_id", 0)
        lines.append(
            {
                "coords": coords,
                "group": group,
                "name_list": name_list,
                "line_id": line_id,
            }
        )
    return lines


def schematic_layout(lines):
    """
    Grid-based schematic layout:
    - Each line is assigned a unique y-row (integer).
    - For each line, stations are placed at consecutive x positions (integers).
    - Transfer stations (shared by multiple lines) are aligned at the same x across all lines, and their y is the average of the lines they appear on.
    - Returns: dict {station_key: (x, y)}, schematic_lines (list of station_keys), key_to_name
    """
    # 1. Assign each line to a unique y-row
    group_ids = []
    for line in lines:
        g = line["group"]
        if g not in group_ids:
            group_ids.append(g)
    group_to_y = {g: i for i, g in enumerate(group_ids)}
    # 2. Build a mapping from station to all (line, index) appearances
    station_appearances = {}
    key_to_name = {}
    for line in lines:
        group = line["group"]
        y = group_to_y[group]
        name_list = line.get("name_list", [])
        for i, coord in enumerate(line["coords"]):
            key = tuple(np.round(coord, 6))
            if key not in station_appearances:
                station_appearances[key] = []
            station_appearances[key].append((group, y, i))
            if key not in key_to_name:
                name = ""
                if name_list and i < len(name_list):
                    name = name_list[i]
                key_to_name[key] = name
    # 3. Assign x positions: for each station, use the minimum index it appears at on any line
    station_x = {}
    for key, appearances in station_appearances.items():
        # Use the minimum index (leftmost appearance) for x
        x = min(idx for _, _, idx in appearances)
        station_x[key] = x
    # 4. For each line, ensure its stations are at consecutive x (with possible gaps for transfers)
    #    For transfer stations, align x across all lines
    #    For y, use the line's y-row, but for transfer stations, average y across all lines
    pos = {}
    for key, appearances in station_appearances.items():
        x = station_x[key]
        y_vals = [y for _, y, _ in appearances]
        y = np.mean(y_vals)
        pos[key] = (x, y)
    # 5. Build schematic_lines as lists of station_keys in order for each line
    schematic_lines = []
    for line in lines:
        line_keys = [tuple(np.round(coord, 6)) for coord in line["coords"]]
        schematic_lines.append(line_keys)
    return pos, schematic_lines, key_to_name


def plot_schematic(lines, out_path="schematic.png", figsize=(14, 8)):
    pos, schematic_lines, key_to_name = schematic_layout(lines)
    fig, ax = plt.subplots(figsize=figsize)
    # Draw grid for reference (optional, can comment out)
    all_x = [xy[0] for xy in pos.values()]
    all_y = [xy[1] for xy in pos.values()]
    for x in range(int(min(all_x)) - 1, int(max(all_x)) + 2):
        ax.axvline(x, color="#eee", lw=0.5, zorder=0)
    for y in range(int(min(all_y)) - 1, int(max(all_y)) + 2):
        ax.axhline(y, color="#eee", lw=0.5, zorder=0)
    # Draw lines
    for idx, line in enumerate(lines):
        group = line["group"]
        color = COLORS[group % len(COLORS)]
        keys = [tuple(np.round(coord, 6)) for coord in line["coords"]]
        xy = np.array([pos[k] for k in keys])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linewidth=5,
            solid_capstyle="round",
            zorder=1,
        )
    # Draw stations and offset text labels to avoid overlap
    label_offsets = {}
    for i, (key, (x, y)) in enumerate(pos.items()):
        ax.scatter(x, y, s=120, color="white", edgecolor="black", zorder=2)
        name = key_to_name.get(key, "")
        if name:
            # Offset label above or below depending on y (to avoid overlap)
            offset = 0.7 if (round(y * 2) % 2 == 0) else -0.9
            # If this (x, y) is already used, offset further
            if (x, round(y, 2)) in label_offsets:
                offset += label_offsets[(x, round(y, 2))] * 0.5
                label_offsets[(x, round(y, 2))] += 1
            else:
                label_offsets[(x, round(y, 2))] = 1
            ax.text(
                x,
                y + offset,
                name,
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=11,
                fontweight="bold",
                zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.7),
            )
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def process_all_geojsons():
    data_dir = os.path.join("data", "output")
    img_dir = "img"
    os.makedirs(img_dir, exist_ok=True)
    for fname in os.listdir(data_dir):
        if fname.endswith(".geojson"):
            geojson_path = os.path.join(data_dir, fname)
            out_path = os.path.join(img_dir, os.path.splitext(fname)[0] + ".png")
            print(f"Processing {geojson_path} -> {out_path}")
            lines = load_lines_geojson(geojson_path)
            plot_schematic(lines, out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments: process all geojsons in data/output
        process_all_geojsons()
    elif len(sys.argv) == 2:
        lines = load_lines_geojson(sys.argv[1])
        out_path = os.path.join(
            "img", os.path.splitext(os.path.basename(sys.argv[1]))[0] + ".png"
        )
        plot_schematic(lines, out_path)
    elif len(sys.argv) > 2:
        lines = load_lines_geojson(sys.argv[1])
        out_path = sys.argv[2]
        plot_schematic(lines, out_path)
