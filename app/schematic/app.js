// D3 force-directed schematic for transit lines
// Expects lines GeoJSON in lines.json (same format as your schematic.py loader)

const COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#808080", "#000000", "#a9a9a9", "#ff4500", "#2e8b57", "#1e90ff",
    "#ff69b4", "#7cfc00", "#8a2be2", "#00ced1"
];

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("#graph").attr("width", width).attr("height", height);

// Load lines.json (exported from your pipeline, or convert a .geojson to .json)
d3.json('../../data/output/lines_genetic.geojson').then(data => {
    // Handle GeoJSON: features array
    const nodeMap = new Map();
    const links = [];
    data.features.forEach((feat, i) => {
        const props = feat.properties || {};
        const coords = feat.geometry.coordinates;
        const color = COLORS[props.group % COLORS.length];
        const name_list = props.name_list || [];
        coords.forEach((coord, idx) => {
            const key = coord.join(",");
            if (!nodeMap.has(key)) {
                nodeMap.set(key, {
                    id: key,
                    name: (name_list && idx < name_list.length) ? name_list[idx] : "",
                    group: props.group,
                    lines: new Set([props.line_id]),
                });
            } else {
                nodeMap.get(key).lines.add(props.line_id);
            }
            // Add link to previous node
            if (idx > 0) {
                const prevKey = coords[idx - 1].join(",");
                links.push({
                    source: prevKey,
                    target: key,
                    group: props.group,
                    line_id: props.line_id,
                    color: color
                });
            }
        });
    });
    // Convert nodeMap to array and lines to array
    const nodes = Array.from(nodeMap.values()).map(n => ({ ...n, lines: Array.from(n.lines) }));

    // Only include nodes that are actual stations (i.e., have a name)
    const stationNodes = nodes.filter(n => n.name && n.name.trim() !== "");
    // Remove links that connect to non-station nodes
    const stationNodeIds = new Set(stationNodes.map(n => n.id));
    const stationLinks = links.filter(l => stationNodeIds.has(l.source) && stationNodeIds.has(l.target));

    // D3 force simulation
    const simulation = d3.forceSimulation(stationNodes)
        .force("link", d3.forceLink(stationLinks).id(d => d.id).distance(80).strength(1))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(28));

    // --- Use a group for all network elements for proper scaling ---
    const networkGroup = svg.append("g").attr("id", "network");

    // Draw links
    const link = networkGroup.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.7)
        .selectAll("line")
        .data(stationLinks)
        .join("line")
        .attr("stroke-width", 4)
        .attr("stroke", d => COLORS[d.group % COLORS.length]);

    // Draw nodes
    const node = networkGroup.append("g")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .selectAll("g")
        .data(stationNodes)
        .join("g")
        .call(drag(simulation));

    node.append("circle")
        .attr("r", 15)
        .attr("fill", d => COLORS[d.group % COLORS.length]);

    node.append("text")
        .attr("x", 0)
        .attr("y", 5)
        .attr("text-anchor", "middle")
        .attr("font-size", 13)
        .attr("fill", "#000") // black text
        .text(d => d.name);

    node.append("title")
        .text(d => d.name || d.id);

    let scale = 1, tx = 0, ty = 0;
    // --- Remove group transform, instead apply transform to node/link positions ---
    simulation.on("end", () => {
        const margin = 80;
        const xs = stationNodes.map(d => d.x), ys = stationNodes.map(d => d.y);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        const netWidth = maxX - minX || 1;
        const netHeight = maxY - minY || 1;
        scale = Math.min(
            (width - 2 * margin) / netWidth,
            (height - 2 * margin) / netHeight,
            3
        );
        tx = (width - scale * (minX + maxX)) / 2;
        ty = (height - scale * (minY + maxY)) / 2;
        // Trigger a tick to update positions with new scale/translate
        simulation.alpha(0.01).restart();
        setTimeout(() => simulation.stop(), 100);
    });

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x * scale + tx)
            .attr("y1", d => d.source.y * scale + ty)
            .attr("x2", d => d.target.x * scale + tx)
            .attr("y2", d => d.target.y * scale + ty);
        node
            .attr("transform", d => `translate(${d.x * scale + tx},${d.y * scale + ty})`);
    });

    // Drag behavior
    function drag(simulation) {
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }
});
