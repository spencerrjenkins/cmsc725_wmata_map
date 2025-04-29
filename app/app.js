// Helper to fetch and parse GeoJSON
async function fetchGeoJSON(url) {
    const resp = await fetch(url);
    return await resp.json();
}
const currentLinesLayerNames = [];
const catchmentCircles = [];
// Color palette for lines
const COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#1b9e77',
    '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02',
    '#a6761d', '#666666', '#b15928', '#b2df8a', '#fb9a99'
];
const map = L.map('map').setView([38.9, -77.05], 10);
var Stadia_AlidadeSmooth = L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.{ext}', {
    minZoom: 0,
    maxZoom: 20,
    attribution: '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    ext: 'png'
}).addTo(map);

const lineMarkers = {};
const layers = {};
const layerToggles = document.getElementById('layer-toggles');
// Maintain order of layers for toggles
const layerOrder = [];
const lineLayerNames = [];

// Utility: round coordinate for matching
function roundCoord(coord) {
    return [Number(coord[0].toFixed(6)), Number(coord[1].toFixed(6))];
}

// Store which lines stop at each vertex
const vertexLineMap = {};
const vertexKDEMap = {};

// Load network (graph) - draw first, under lines
fetchGeoJSON('../data/output/network.geojson').then(data => {
    const networkLayer = L.geoJSON(data, {
        style: feature => {
            if (feature.geometry.type === 'LineString') {
                return { color: '#222', weight: 1, opacity: 0.5, dashArray: '4 4' };
            }
            return {};
        },
        pointToLayer: (feature, latlng) => L.circleMarker(latlng, { radius: 2, color: '#222', fillOpacity: 0.5 })
    });
    layers['Network'] = networkLayer;
    // Do not add to map by default
    addLayerToggle('Network', false);
    layerOrder.push('Network');
});

// Add support for toggling between lines_naive and lines_genetic
let currentLinesSource = 'lines_naive'; // 'lines_naive', or 'lines_genetic'

function loadLinesGeoJSON(source) {
    // Remove existing line layers and markers
    currentLinesLayerNames.forEach(name => {
        if (layers[name]) map.removeLayer(layers[name]);
        if (lineMarkers[name]) lineMarkers[name].forEach(m => map.removeLayer(m));
        delete layers[name];
        delete lineMarkers[name];
    });
    currentLinesLayerNames.length = 0;
    // Load the selected lines file
    fetchGeoJSON(`../data/output/${source}.geojson`).then(data => {
        (data.features || []).forEach((feature, i) => {
            const color = COLORS[i % COLORS.length];
            const totalDistance = (feature.properties.segment_lengths || []).reduce((a, b) => a + b, 0);
            const numStations = feature.geometry.coordinates.length;
            const name = `Line ${feature.properties.line_id ?? i}`;
            lineMarkers[name] = [];
            const lineLayer = L.geoJSON(feature, {
                style: { color, weight: 4, opacity: 1 },
                onEachFeature: (feat, layer) => {
                    const totalDistance = (feat.properties.segment_lengths || []).reduce((a, b) => a + b, 0);
                    const numStations = feat.geometry.coordinates.length;
                    layer.bindTooltip(
                        `Line ${feat.properties.line_id}<br>Total distance: ${(totalDistance / 1000).toFixed(2)} km<br>Stations: ${numStations}`,
                        {
                            sticky: true,
                            direction: 'top',
                            offset: [0, -10]
                        }
                    );
                }
            });
            layers[name] = lineLayer;
            lineLayer.addTo(map);
            lineMarkers[name] = [];
            currentLinesLayerNames.push(name);
            // Add vertex markers, popups, and circles
            const coords = feature.geometry.coordinates;
            const kdeValues = feature.properties.kde_values || [];
            coords.forEach((coord, idx) => {
                const key = roundCoord(coord).join(',');
                const kde = vertexKDEMap[key];
                const linesHere = vertexLineMap[key] || [];
                const icon = L.icon({
                    iconUrl: 'assets/wmata.svg',
                    iconSize: [10, 10],
                    iconAnchor: [7, 7],
                    popupAnchor: [0, -7]
                });
                const marker = L.marker([coord[1], coord[0]], { icon }).addTo(map);
                attachRouteFinderToMarker(marker, coord[1], coord[0]);
                const popup = `<b>Vertex</b><br>KDE Score: ${kde?.toFixed(2) ?? 'N/A'}<br>Lines: ${linesHere.map(l => `Line ${l}`).join(', ')}`;
                marker.bindPopup(popup);
                lineMarkers[name].push(marker);
            });
        });
        // After all lines are loaded, sort currentLinesLayerNames numerically
        currentLinesLayerNames.sort((a, b) => {
            const numA = parseInt(a.replace('Line ', ''));
            const numB = parseInt(b.replace('Line ', ''));
            return numA - numB;
        });
        // Remove all previous line toggles before rendering new ones
        renderLayerToggles();
    });
}

// Add UI for toggling between lines_naive and lines_genetic
const linesSourceSelect = document.createElement('select');
linesSourceSelect.id = 'lines-source-select';
['lines_naive', 'lines_genetic'].forEach(src => {
    const opt = document.createElement('option');
    opt.value = src;
    opt.textContent = src.replace('lines', 'Lines').replace('_', ' ').replace('naive', 'Naive').replace('genetic', 'Genetic');
    linesSourceSelect.appendChild(opt);
});
linesSourceSelect.onchange = function () {
    currentLinesSource = this.value;
    loadLinesGeoJSON(currentLinesSource);
};
document.getElementById('controls').appendChild(linesSourceSelect);

// On page load, load default lines
loadLinesGeoJSON(currentLinesSource);

// Load lines and add tooltips, vertex markers, and circles
fetchGeoJSON('../data/output/lines_naive.geojson').then(data => {
    // First, build a map of vertex => [line_ids], and vertex => kde
    (data.features || []).forEach((feature, i) => {
        const coords = feature.geometry.coordinates;
        const kdeValues = feature.properties.kde_values || [];
        coords.forEach((coord, idx) => {
            const key = roundCoord(coord).join(',');
            if (!vertexLineMap[key]) vertexLineMap[key] = [];
            vertexLineMap[key].push(feature.properties.line_id);
            if (!vertexKDEMap[key]) vertexKDEMap[key] = kdeValues[idx];
        });
    });
    // Draw lines with tooltips
    (data.features || []).forEach((feature, i) => {
        const color = COLORS[i % COLORS.length];
        const totalDistance = (feature.properties.segment_lengths || []).reduce((a, b) => a + b, 0);
        const numStations = feature.geometry.coordinates.length;
        const name = `Line ${feature.properties.line_id ?? i}`;
        lineMarkers[name] = [];
        const lineLayer = L.geoJSON(feature, {
            style: { color, weight: 4, opacity: 1 },
            onEachFeature: (feat, layer) => {
                const totalDistance = (feat.properties.segment_lengths || []).reduce((a, b) => a + b, 0);
                const numStations = feat.geometry.coordinates.length;
                layer.bindTooltip(
                    `Line ${feat.properties.line_id}<br>Total distance: ${(totalDistance / 1000).toFixed(2)} km<br>Stations: ${numStations}`,
                    {
                        sticky: true,
                        direction: 'top',
                        offset: [0, -10]
                    }
                );
            }
        });
        layers[name] = lineLayer;
        lineLayer.addTo(map);
        addLayerToggle(name, true);
        lineLayerNames.push(name);
        // Add vertex markers, popups, and circles
        const coords = feature.geometry.coordinates;
        const kdeValues = feature.properties.kde_values || [];
        coords.forEach((coord, idx) => {
            const key = roundCoord(coord).join(',');
            const kde = vertexKDEMap[key];
            const linesHere = vertexLineMap[key] || [];
            const icon = L.icon({
                iconUrl: 'assets/wmata.svg',
                iconSize: [10, 10],
                iconAnchor: [7, 7],
                popupAnchor: [0, -7]
            });
            const marker = L.marker([coord[1], coord[0]], { icon }).addTo(map);
            attachRouteFinderToMarker(marker, coord[1], coord[0]);
            const popup = `<b>Vertex</b><br>KDE Score: ${kde?.toFixed(2) ?? 'N/A'}<br>Lines: ${linesHere.map(l => `Line ${l}`).join(', ')}`;
            marker.bindPopup(popup);
            lineMarkers[name].push(marker);
            // Circle of 700m
            const circle = L.circle([coord[1], coord[0]], {
                radius: 700,
                color: color,
                fill: false,
                weight: 1,
                opacity: 0.3
            });
            catchmentCircles.push(circle);
            // Do not add to map by default
        });
    });
    // After all lines are loaded, sort lineLayerNames numerically
    lineLayerNames.sort((a, b) => {
        const numA = parseInt(a.replace('Line ', ''));
        const numB = parseInt(b.replace('Line ', ''));
        return numA - numB;
    });
    // Update layerOrder: all lines in order, then Network last
    layerOrder.length = 0;
    lineLayerNames.forEach(n => layerOrder.push(n));
    if (layers['Network']) layerOrder.push('Network');
    // Re-render toggles
    renderLayerToggles();
});

function addLayerToggle(name, checked) {
    // No-op: toggles are rendered in renderLayerToggles
}

function renderLayerToggles() {
    while (layerToggles.firstChild) layerToggles.removeChild(layerToggles.firstChild);
    // Render all line toggles in order (use currentLinesLayerNames for currently loaded network)
    currentLinesLayerNames.forEach(n => createToggle(n, map.hasLayer(layers[n])));
    // Render network toggle if present
    if (layers['Network']) createToggle('Network', map.hasLayer(layers['Network']));
    // Add catchment area toggle at the end
    layerToggles.appendChild(catchmentToggleLabel);
}

function createToggle(name, checked) {
    const id = `layer-toggle-${name.replace(/\s/g, '-')}`;
    const label = document.createElement('label');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = id;
    checkbox.checked = checked;
    checkbox.onchange = () => {
        if (checkbox.checked) {
            layers[name].addTo(map);
            if (lineMarkers[name]) lineMarkers[name].forEach(m => m.addTo(map));
        } else {
            map.removeLayer(layers[name]);
            if (lineMarkers[name]) lineMarkers[name].forEach(m => map.removeLayer(m));
        }
    };
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(' ' + name));
    layerToggles.appendChild(label);
}

// Create Station catchment area toggle (define before any function uses it)
const catchmentToggleLabel = document.createElement('label');
const catchmentToggle = document.createElement('input');
catchmentToggle.type = 'checkbox';
catchmentToggle.id = 'catchment-toggle';
catchmentToggle.checked = false;
catchmentToggle.onchange = () => {
    catchmentCircles.forEach(circle => {
        if (catchmentToggle.checked) {
            if (!map.hasLayer(circle)) map.addLayer(circle);
        } else {
            if (map.hasLayer(circle)) map.removeLayer(circle);
        }
    });
};
catchmentToggleLabel.appendChild(catchmentToggle);
catchmentToggleLabel.appendChild(document.createTextNode(' Station catchment areas'));
// On page load, ensure circles are not shown
catchmentCircles.forEach(circle => { if (map.hasLayer(circle)) map.removeLayer(circle); });

// --- Route Finder Logic ---
let routeFinderState = 'idle'; // 'idle', 'selectingStart', 'selectingEnd'
let routeStart = null;
let routeEnd = null;
let routeHighlightLayer = null;
let routeNodeMarkers = [];
let linesGraph = null;
let nodeIdByLatLng = {};
let latLngByNodeId = {};
let lineIdByNodePair = {};
let nodeToLineIds = {};

const routeFinderBtn = document.getElementById('route-finder-btn');
const routeFinderStatus = document.getElementById('route-finder-status');
const routeFinderResult = document.getElementById('route-finder-result');

// Load lines graph for routing
fetchGeoJSON('../data/output/lines_naive.geojson').then(data => {
    linesGraph = { nodes: {}, edges: {} };
    (data.features || []).forEach(f => {
        const coords = f.geometry.coordinates;
        const lineId = f.properties.line_id;
        for (let i = 0; i < coords.length; ++i) {
            const latlng = [coords[i][1], coords[i][0]];
            const key = latlng.join(',');
            linesGraph.nodes[key] = latlng;
            if (!nodeToLineIds[key]) nodeToLineIds[key] = new Set();
            nodeToLineIds[key].add(lineId);
        }
        for (let i = 0; i < coords.length - 1; ++i) {
            const a = [coords[i][1], coords[i][0]].join(',');
            const b = [coords[i + 1][1], coords[i + 1][0]].join(',');
            const dist = L.latLng(coords[i][1], coords[i][0]).distanceTo(L.latLng(coords[i + 1][1], coords[i + 1][0]));
            if (!linesGraph.edges[a]) linesGraph.edges[a] = [];
            if (!linesGraph.edges[b]) linesGraph.edges[b] = [];
            linesGraph.edges[a].push({ to: b, dist, lineId });
            linesGraph.edges[b].push({ to: a, dist, lineId });
            // For description
            const pairKey = [a, b].sort().join('|');
            if (!lineIdByNodePair[pairKey]) lineIdByNodePair[pairKey] = new Set();
            lineIdByNodePair[pairKey].add(lineId);
        }
    });
});

// Helper: get currently visible line IDs
function getVisibleLineIds() {
    return lineLayerNames.filter(name => {
        const checkbox = document.getElementById(`layer-toggle-${name.replace(/\s/g, '-')}`);
        return checkbox && checkbox.checked;
    }).map(name => parseInt(name.replace('Line ', '')));
}

routeFinderBtn.onclick = () => {
    if (routeFinderState === 'selectingStart' || routeFinderState === 'selectingEnd') {
        // Clear/reset
        routeFinderState = 'selectingStart';
        routeStart = null;
        routeEnd = null;
        if (routeHighlightLayer) { map.removeLayer(routeHighlightLayer); routeHighlightLayer = null; }
        routeNodeMarkers.forEach(m => map.removeLayer(m));
        routeNodeMarkers = [];
        routeFinderStatus.textContent = 'Click the starting station.';
        routeFinderResult.textContent = '';
        routeFinderBtn.textContent = 'clear';
        // Remove Exit button if present
        const exitBtn = document.getElementById('route-finder-exit-btn');
        if (exitBtn) exitBtn.remove();
        return;
    }
    // Start route finder
    routeFinderState = 'selectingStart';
    routeStart = null;
    routeEnd = null;
    if (routeHighlightLayer) { map.removeLayer(routeHighlightLayer); routeHighlightLayer = null; }
    routeNodeMarkers.forEach(m => map.removeLayer(m));
    routeNodeMarkers = [];
    routeFinderStatus.textContent = 'Click the starting station.';
    routeFinderResult.textContent = '';
    routeFinderBtn.textContent = 'clear';
    // Remove Exit button if present
    const exitBtn = document.getElementById('route-finder-exit-btn');
    if (exitBtn) exitBtn.remove();
};

// Helper: find nearest node in lines graph, but only for visible lines
function findNearestLineNodeVisible(lat, lng) {
    const visibleLineIds = new Set(getVisibleLineIds());
    let minDist = Infinity, minKey = null;
    for (const [key, ll] of Object.entries(linesGraph.nodes || {})) {
        // Only consider nodes on visible lines
        const linesHere = nodeToLineIds[key] ? Array.from(nodeToLineIds[key]) : [];
        if (!linesHere.some(lid => visibleLineIds.has(lid))) continue;
        const d = Math.abs(ll[0] - lat) + Math.abs(ll[1] - lng);
        if (d < minDist) { minDist = d; minKey = key; }
    }
    return minKey;
}

// Attach click handlers to station markers after they are created
function attachRouteFinderToMarker(marker, lat, lng) {
    marker.on('click', (e) => {
        if (routeFinderState === 'selectingStart') {
            routeStart = findNearestLineNodeVisible(lat, lng);
            if (!routeStart) return;
            routeFinderState = 'selectingEnd';
            routeFinderStatus.textContent = 'Click the destination station.';
            // Highlight start marker
            const m = L.circleMarker([lat, lng], { radius: 12, color: 'green', fillOpacity: 0.7 }).addTo(map);
            routeNodeMarkers.push(m);
            routeFinderBtn.textContent = 'clear';
        } else if (routeFinderState === 'selectingEnd') {
            routeEnd = findNearestLineNodeVisible(lat, lng);
            if (!routeEnd || routeEnd === routeStart) return;
            routeFinderState = 'idle';
            // Highlight end marker
            const m = L.circleMarker([lat, lng], { radius: 12, color: 'red', fillOpacity: 0.7 }).addTo(map);
            routeNodeMarkers.push(m);
            routeFinderStatus.textContent = 'Finding route...';
            // Only consider visible lines
            const visibleLineIds = new Set(getVisibleLineIds());
            const path = dijkstraLinesVisible(linesGraph, routeStart, routeEnd, visibleLineIds);
            if (path.length < 2) {
                routeFinderStatus.textContent = 'No route found.';
                return;
            }
            // Draw route
            const routeCoords = [];
            let totalRouteDistance = 0;
            for (let i = 0; i < path.length - 1; ++i) {
                const from = path[i], to = path[i + 1];
                routeCoords.push(linesGraph.nodes[from]);
                // Calculate distance for this segment
                const edge = (linesGraph.edges[from] || []).find(e => e.to === to);
                if (edge) totalRouteDistance += edge.dist;
            }
            routeCoords.push(linesGraph.nodes[path[path.length - 1]]);
            if (routeHighlightLayer) map.removeLayer(routeHighlightLayer);
            routeHighlightLayer = L.polyline(routeCoords, { color: 'blue', weight: 8, opacity: 0.7 }).addTo(map);
            // Show description with lines used and total distance
            let lineSequence = [];
            for (let i = 0; i < path.length - 1; ++i) {
                const a = path[i], b = path[i + 1];
                const pairKey = [a, b].sort().join('|');
                const lines = lineIdByNodePair[pairKey] ? Array.from(lineIdByNodePair[pairKey]) : [];
                if (lines.length > 0) {
                    if (lineSequence.length === 0 || lineSequence[lineSequence.length - 1] !== lines[0]) {
                        lineSequence.push(lines[0]);
                    }
                }
            }
            routeFinderStatus.textContent = 'Route found!';
            // Calculate travel time: 60 km/h + 6 min per transfer
            const speed_kmh = 60;
            const totalDistanceKm = totalRouteDistance / 1000;
            let transfers = 0;
            for (let i = 1; i < lineSequence.length; ++i) {
                if (lineSequence[i] !== lineSequence[i - 1]) transfers++;
            }
            const travelTimeHours = totalDistanceKm / speed_kmh;
            const travelTimeMinutes = travelTimeHours * 60 + transfers * 6;
            routeFinderResult.innerHTML = `<b>Route:</b> ${lineSequence.map(l => `Line ${l}`).join(' → ')}<br><b>Stops:</b> ${path.length - 1}<br><b>Total distance:</b> ${totalDistanceKm.toFixed(2)} km<br><b>Transfers:</b> ${transfers}<br><b>Estimated travel time:</b> ${travelTimeMinutes.toFixed(1)} min`;
            // Add Exit Route Finder button
            let exitBtn = document.getElementById('route-finder-exit-btn');
            if (!exitBtn) {
                exitBtn = document.createElement('button');
                exitBtn.id = 'route-finder-exit-btn';
                exitBtn.textContent = 'Exit Route Finder';
                exitBtn.style.marginLeft = '10px';
                routeFinderBtn.parentNode.insertBefore(exitBtn, routeFinderBtn.nextSibling);
                exitBtn.onclick = () => {
                    routeFinderState = 'idle';
                    routeStart = null;
                    routeEnd = null;
                    if (routeHighlightLayer) { map.removeLayer(routeHighlightLayer); routeHighlightLayer = null; }
                    routeNodeMarkers.forEach(m => map.removeLayer(m));
                    routeNodeMarkers = [];
                    routeFinderStatus.textContent = '';
                    routeFinderResult.textContent = '';
                    routeFinderBtn.textContent = 'Route Finder';
                    exitBtn.remove();
                };
            }
            routeFinderBtn.textContent = 'clear';
        }
    });
}

// Dijkstra's algorithm for shortest path on lines graph, only using visible lines
function dijkstraLinesVisible(graph, start, end, visibleLineIds) {
    const dist = {}, prev = {}, Q = new Set(Object.keys(graph.nodes));
    for (const v of Q) dist[v] = Infinity;
    dist[start] = 0;
    while (Q.size) {
        let u = null, min = Infinity;
        for (const v of Q) { if (dist[v] < min) { min = dist[v]; u = v; } }
        if (u === null || u === end) break;
        Q.delete(u);
        for (const e of (graph.edges[u] || [])) {
            if (!visibleLineIds.has(e.lineId)) continue;
            const alt = dist[u] + e.dist;
            if (alt < dist[e.to]) {
                dist[e.to] = alt;
                prev[e.to] = u;
            }
        }
    }
    // Reconstruct path
    const path = [];
    let u = end;
    while (u !== undefined) { path.unshift(u); u = prev[u]; }
    if (path[0] != start) return [];
    return path;
}