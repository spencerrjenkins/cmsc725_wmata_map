# cmsc725_wmata_map

CMSC 725 Spring 2025 Class Project: Designing an Optimal WMATA Metro Map

## Project Writeup Guidelines for `eda.ipynb`

## 1. Introduction

- Briefly describe the goal of the project (e.g., analyzing and optimizing transit networks in the DC/MD/VA region).
- State the motivation and potential impact.

## 2. Data Sources

- List and describe all data sources used:
  - Census block shapefiles for DC, MD, VA.
  - Real transit shapefiles (WMATA, MARC, VRE, DC Streetcar, Purple Line).
  - Points of interest and facilities from DC and MD open data portals.
- Mention any preprocessing or filtering (e.g., selecting specific counties).

## 3. Data Processing

- Explain how geospatial data is loaded and filtered using GeoPandas.
- Describe how population and land area are used to compute "transit potential" for each block.
- Discuss the creation of a unified GeoDataFrame for the region.

## 4. Feature Engineering

- Describe the calculation of "transit potential" and its log transformation.
- Explain the extraction and combination of population and non-population points.

## 5. Spatial Analysis

- Summarize the use of Kernel Density Estimation (KDE) to estimate transit demand hotspots.
- Explain the construction of spatial graphs (Gabriel graph, MST, community contraction).

## 6. Graph and Network Modeling

- Describe how the spatial graph is built and contracted using Louvain communities.
- Explain the assignment of edge weights and reduction to a minimum spanning tree (MST).

## 7. Machine Learning & RL

- Summarize the use of KDE for reward calculation.
- Describe the setup of a custom Gym environment for reinforcement learning (RL) to optimize network design.
- Mention the use of a Graph Convolutional Network (GCN) for node feature learning.

## 8. Visualization

- List the types of maps and plots generated (e.g., KDE heatmaps, network overlays, basemaps).
- Explain the use of contextily for basemaps and matplotlib for plotting.

## 9. Results & Discussion

- Summarize key findings from the spatial and network analysis.
- Discuss the implications of the RL and GCN modeling for transit planning.

## 10. Conclusion & Future Work

- Reflect on the strengths and limitations of the approach.
- Suggest possible extensions (e.g., more data, improved RL reward functions, real-world validation).

## 11. References

- List all data sources, libraries, and relevant literature.

---
**Tip:** Use figures and code snippets from the notebook to illustrate each section.

Problems

1. Data Used
Limited Socioeconomic Features:
The main "transit potential" metric is based only on population and land/water area. It does not incorporate income, car ownership, employment density, or other key transit demand drivers.
Block-Level Aggregation:
Using census blocks is good for granularity, but may introduce noise or instability in sparsely populated or irregularly shaped blocks.
Non-Population Points:
The integration of non-population points (schools, hospitals, etc.) is a good idea, but the method for combining and filtering these points is somewhat ad hoc and may miss important locations or introduce duplicates.
Data Quality/Completeness:
The code assumes the presence and correctness of many shapefiles and GeoJSONs. There is little error checking or handling of missing/corrupt data.
Temporal Staticity:
All data appears to be a snapshot in time; no consideration is given to temporal changes (e.g., population growth, planned developments).

2. Suitability of Algorithms
KDE for Transit Demand:
Kernel Density Estimation is a reasonable first step, but it assumes spatial smoothness and does not account for barriers (rivers, highways) or actual travel demand patterns.
Gabriel Graph & Louvain Clustering:
These are generic spatial and community detection algorithms. While useful for reducing complexity, they are not tailored to transit network design, which often requires consideration of connectivity, redundancy, and coverage.
RL Environment:
The RL environment is quite simplistic:
The action space is just the set of edges; there is no mechanism for adding/removing nodes, or for more complex network modifications.
The reward is based only on KDE scores at edge endpoints, not on actual network performance (e.g., shortest paths, coverage, accessibility).
The environment does not simulate passenger flows, transfers, or operational constraints.
GNN Model:
The GCN is defined but not actually trained or evaluated in a meaningful way. Its purpose in the pipeline is unclear.
No Baseline Comparison:
There is no comparison to existing networks or to simple heuristics (e.g., shortest path, minimum spanning tree) to evaluate the quality of the generated network.
3. Visualization Quality
Basic Plots:
The visualizations are functional but basic. They overlay lines and points on a basemap, but:
There is little interactivity or ability to explore results.
Legends, labels, and color schemes could be improved for clarity.
The KDE heatmap may not be well-aligned with the basemap due to projection issues.
No Quantitative Evaluation:
Visualizations are qualitative only; there are no summary statistics, coverage metrics, or accessibility maps to quantify network performance.
No User-Focused Outputs:
The outputs are not tailored for stakeholders (e.g., planners, the public) and may be hard to interpret without technical background.
