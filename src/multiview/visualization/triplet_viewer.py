"""Generate standalone HTML viewers for triplet JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_triplet_viewer(
    triplets_json_path: str | Path,
    output_html_path: str | Path | None = None,
) -> Path:
    """Generate standalone HTML viewer with embedded triplet data.

    Args:
        triplets_json_path: Path to triplets.json file
        output_html_path: Path for output HTML (default: viewer.html in same dir)

    Returns:
        Path to generated HTML file
    """
    triplets_path = Path(triplets_json_path)

    if not triplets_path.exists():
        raise FileNotFoundError(f"Triplets file not found: {triplets_json_path}")

    # Read triplets data
    with open(triplets_path) as f:
        triplets_data = json.load(f)

    # Determine output path
    if output_html_path is None:
        output_html_path = triplets_path.parent / "viewer.html"
    else:
        output_html_path = Path(output_html_path)

    # Generate HTML with embedded data
    html_content = _generate_html_template(
        triplets_data=triplets_data,
        triplets_path=triplets_path,
    )

    # Write output file
    with open(output_html_path, "w") as f:
        f.write(html_content)

    logger.info(
        f"Generated triplet viewer with {len(triplets_data)} triplets: {output_html_path}"
    )
    return output_html_path


def _generate_html_template(
    triplets_data: list[dict],
    triplets_path: Path,
) -> str:
    """Generate the HTML template with embedded triplet data.

    Args:
        triplets_data: List of triplet dictionaries
        triplets_path: Path to the source triplets.json file

    Returns:
        HTML string
    """
    # Check if embedding coordinates are available
    has_embeddings = (
        triplets_data
        and isinstance(triplets_data[0].get("anchor"), dict)
        and "embedding_viz" in triplets_data[0]["anchor"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Triplet Viewer - {triplets_path.name}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f9f9f9;
            overflow-x: hidden;
        }}
        .container {{
            max-width: {'100%' if has_embeddings else '1400px'};
            margin: {'0' if has_embeddings else '20px auto'};
            padding: {'0' if has_embeddings else '20px'};
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #0066cc;
            padding: {'15px 20px 10px 20px' if has_embeddings else '0 0 10px 0'};
            margin: 0 0 20px 0;
            background: white;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: {'0 20px 20px 20px' if has_embeddings else '0 0 20px 0'};
        }}
        .view-controls {{
            background: white;
            padding: 15px 20px;
            border-bottom: 2px solid #ddd;
            display: flex;
            gap: 15px;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .view-controls button {{
            padding: 8px 16px;
            border: 2px solid #0066cc;
            background: white;
            color: #0066cc;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .view-controls button:hover {{
            background: #e3f2fd;
        }}
        .view-controls button.active {{
            background: #0066cc;
            color: white;
        }}
        .embedding-view {{
            display: {'flex' if has_embeddings else 'none'};
            height: calc(100vh - 200px);
        }}
        .embedding-canvas {{
            flex: 1;
            background: white;
            position: relative;
            overflow: hidden;
        }}
        .embedding-sidebar {{
            width: 400px;
            background: white;
            border-left: 2px solid #ddd;
            overflow-y: auto;
            padding: 20px;
        }}
        .list-view {{
            display: {'none' if has_embeddings else 'block'};
            padding: 20px;
        }}
        .triplet {{
            background: white;
            border: 2px solid #ddd;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .triplet-header {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #0066cc;
        }}
        .images {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .image-container {{
            flex: 1;
            min-width: 280px;
            background: #fafafa;
            padding: 15px;
            border-radius: 6px;
        }}
        .image-container h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
        }}
        .image-container.anchor {{
            border: 2px solid #0066cc;
        }}
        .image-container.anchor h3 {{ color: #0066cc; }}
        .image-container.positive {{
            border: 2px solid #00aa00;
        }}
        .image-container.positive h3 {{ color: #00aa00; }}
        .image-container.negative {{
            border: 2px solid #cc0000;
        }}
        .image-container.negative h3 {{ color: #cc0000; }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 4px;
            display: block;
        }}
        .metadata {{
            margin-top: 10px;
            font-size: 13px;
            color: #666;
            line-height: 1.6;
        }}
        .metadata strong {{
            color: #333;
        }}
        .quality {{
            margin-top: 15px;
            padding: 15px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
            font-size: 13px;
        }}
        .quality strong {{
            color: #333;
        }}
        .quality-rating {{
            display: inline-block;
            padding: 3px 8px;
            background: #ffc107;
            color: #000;
            border-radius: 3px;
            font-weight: bold;
            margin-left: 5px;
        }}
        .reasoning {{
            margin-top: 8px;
            font-size: 12px;
            line-height: 1.5;
            color: #555;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
        .triplet-selector {{
            margin-bottom: 20px;
        }}
        .triplet-selector select {{
            width: 100%;
            padding: 8px;
            border: 2px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .embedding-point {{
            cursor: pointer;
            transition: r 0.2s;
        }}
        .embedding-point:hover {{
            r: 8;
        }}
        .embedding-point.highlighted {{
            stroke-width: 3;
        }}
        #zoom-controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        #zoom-controls button {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            background: white;
            cursor: pointer;
            border-radius: 3px;
        }}
        #zoom-controls button:hover {{
            background: #f0f0f0;
        }}
        .legend {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Triplet Viewer</h1>
        {"<div class='view-controls'><button class='active' onclick='switchView(\"embedding\")'>Embedding View</button><button onclick='switchView(\"list\")'>List View</button></div>" if has_embeddings else ""}
        <div class="info">
            <strong>Source:</strong> {triplets_path}<br>
            <strong>Total Triplets:</strong> {len(triplets_data)}
            {"<br><strong>Reducer:</strong> " + triplets_data[0]['anchor']['embedding_viz']['reducer'] + "<br><strong>Embedding:</strong> " + triplets_data[0]['anchor']['embedding_viz'].get('embedding_method', triplets_data[0]['anchor']['embedding_viz'].get('embedding_preset', 'unknown')) if has_embeddings else ""}
        </div>

        {"<div class='embedding-view' id='embeddingView'><div class='embedding-canvas' id='embeddingCanvas'><div id='zoom-controls'><button onclick='zoomIn()'>+</button><button onclick='zoomOut()'>-</button><button onclick='resetZoom()'>Reset</button></div><div class='legend'><div class='legend-item'><div class='legend-color' style='background: #999;'></div><span>Document</span></div><div class='legend-item'><div class='legend-color' style='background: #0066cc;'></div><span>Anchor</span></div><div class='legend-item'><div class='legend-color' style='background: #00aa00;'></div><span>Positive</span></div><div class='legend-item'><div class='legend-color' style='background: #cc0000;'></div><span>Negative</span></div></div></div><div class='embedding-sidebar'><div class='triplet-selector'><label><strong>Select Triplet:</strong></label><select id='tripletSelect' onchange='selectTriplet(this.value)'><option value=''>-- All Points --</option></select></div><div id='selectedTriplet'></div></div></div>" if has_embeddings else ""}

        <div class="list-view" id="listView">
            <div id="triplets"></div>
        </div>
    </div>

    <script>
        const triplets = {json.dumps(triplets_data, indent=2)};
        const hasEmbeddings = {'true' if has_embeddings else 'false'};
        let currentView = hasEmbeddings ? 'embedding' : 'list';
        let svgElement, gElement, zoom;

        // Initialize on load
        window.addEventListener('DOMContentLoaded', function() {{
            displayTriplets(triplets);
            if (hasEmbeddings) {{
                initEmbeddingView();
            }}
        }});

        function switchView(view) {{
            currentView = view;
            const embeddingView = document.getElementById('embeddingView');
            const listView = document.getElementById('listView');
            const buttons = document.querySelectorAll('.view-controls button');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            if (view === 'embedding') {{
                embeddingView.style.display = 'flex';
                listView.style.display = 'none';
            }} else {{
                embeddingView.style.display = 'none';
                listView.style.display = 'block';
            }}
        }}

        function initEmbeddingView() {{
            // Extract all unique documents with their positions
            const documentMap = new Map();
            triplets.forEach(triplet => {{
                ['anchor', 'positive', 'negative'].forEach(role => {{
                    const docId = triplet[role + '_id'];
                    const doc = triplet[role];
                    if (doc && doc.embedding_viz && !documentMap.has(docId)) {{
                        documentMap.set(docId, {{
                            id: docId,
                            x: doc.embedding_viz.x,
                            y: doc.embedding_viz.y,
                            text: doc.text || doc,
                            image_path: doc.image_path,
                            role: null,
                            tripletId: null
                        }});
                    }}
                }});
            }});

            const documents = Array.from(documentMap.values());

            // Populate triplet selector
            const select = document.getElementById('tripletSelect');
            triplets.forEach(triplet => {{
                const option = document.createElement('option');
                option.value = triplet.triplet_id;
                option.textContent = `Triplet ${{triplet.triplet_id}}`;
                select.appendChild(option);
            }});

            // Setup D3 visualization
            const container = document.getElementById('embeddingCanvas');
            const width = container.clientWidth;
            const height = container.clientHeight;

            // Create SVG
            const svg = d3.select('#embeddingCanvas')
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .style('background', '#fafafa');

            svgElement = svg;

            // Add zoom behavior
            zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {{
                    gElement.attr('transform', event.transform);
                }});

            svg.call(zoom);

            // Create main group for all points
            const g = svg.append('g');
            gElement = g;

            // Calculate scale to fit all points
            const xExtent = d3.extent(documents, d => d.x);
            const yExtent = d3.extent(documents, d => d.y);
            const xRange = xExtent[1] - xExtent[0];
            const yRange = yExtent[1] - yExtent[0];
            const padding = 0.1;

            const xScale = d3.scaleLinear()
                .domain([xExtent[0] - xRange * padding, xExtent[1] + xRange * padding])
                .range([50, width - 50]);

            const yScale = d3.scaleLinear()
                .domain([yExtent[0] - yRange * padding, yExtent[1] + yRange * padding])
                .range([height - 50, 50]);

            // Draw points
            g.selectAll('circle')
                .data(documents)
                .enter()
                .append('circle')
                .attr('class', 'embedding-point')
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.y))
                .attr('r', 5)
                .attr('fill', '#999')
                .attr('stroke', '#666')
                .attr('stroke-width', 1)
                .attr('opacity', 0.7)
                .attr('data-doc-id', d => d.id)
                .on('mouseover', function(event, d) {{
                    d3.select(this).attr('r', 8);
                    showTooltip(event, d);
                }})
                .on('mouseout', function(event, d) {{
                    d3.select(this).attr('r', 5);
                    hideTooltip();
                }})
                .on('click', function(event, d) {{
                    findTripletForDocument(d.id);
                }});

            // Create tooltip
            const tooltip = d3.select('body')
                .append('div')
                .style('position', 'absolute')
                .style('background', 'white')
                .style('padding', '10px')
                .style('border', '1px solid #ddd')
                .style('border-radius', '4px')
                .style('pointer-events', 'none')
                .style('opacity', 0)
                .style('max-width', '300px')
                .style('font-size', '12px')
                .style('z-index', '1000');

            function showTooltip(event, d) {{
                const text = typeof d.text === 'string' ? d.text : JSON.stringify(d.text);
                const preview = text.slice(0, 200) + (text.length > 200 ? '...' : '');
                tooltip
                    .style('opacity', 1)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY + 10) + 'px')
                    .html(`<strong>Doc ${{d.id}}</strong><br>${{preview}}`);
            }}

            function hideTooltip() {{
                tooltip.style('opacity', 0);
            }}

            // Store scale functions globally
            window.xScale = xScale;
            window.yScale = yScale;
            window.documents = documents;
        }}

        function selectTriplet(tripletId) {{
            if (!tripletId) {{
                // Reset all points
                d3.selectAll('.embedding-point')
                    .attr('fill', '#999')
                    .attr('stroke', '#666')
                    .attr('stroke-width', 1)
                    .attr('r', 5)
                    .attr('opacity', 0.7)
                    .classed('highlighted', false);
                document.getElementById('selectedTriplet').innerHTML = '';
                return;
            }}

            const triplet = triplets.find(t => t.triplet_id == tripletId);
            if (!triplet) return;

            // Reset all points first
            d3.selectAll('.embedding-point')
                .attr('fill', '#999')
                .attr('stroke', '#666')
                .attr('stroke-width', 1)
                .attr('r', 5)
                .attr('opacity', 0.3)
                .classed('highlighted', false);

            // Highlight triplet points
            const roles = {{
                anchor: {{ color: '#0066cc', label: 'Anchor' }},
                positive: {{ color: '#00aa00', label: 'Positive' }},
                negative: {{ color: '#cc0000', label: 'Negative' }}
            }};

            Object.entries(roles).forEach(([role, config]) => {{
                const docId = triplet[role + '_id'];
                d3.selectAll('.embedding-point')
                    .filter(d => d.id === docId)
                    .attr('fill', config.color)
                    .attr('stroke', config.color)
                    .attr('stroke-width', 3)
                    .attr('r', 8)
                    .attr('opacity', 1)
                    .classed('highlighted', true);
            }});

            // Draw lines between points
            const anchorDoc = window.documents.find(d => d.id === triplet.anchor_id);
            const positiveDoc = window.documents.find(d => d.id === triplet.positive_id);
            const negativeDoc = window.documents.find(d => d.id === triplet.negative_id);

            gElement.selectAll('.triplet-line').remove();

            if (anchorDoc && positiveDoc) {{
                gElement.append('line')
                    .attr('class', 'triplet-line')
                    .attr('x1', window.xScale(anchorDoc.x))
                    .attr('y1', window.yScale(anchorDoc.y))
                    .attr('x2', window.xScale(positiveDoc.x))
                    .attr('y2', window.yScale(positiveDoc.y))
                    .attr('stroke', '#00aa00')
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', '5,5')
                    .attr('opacity', 0.5)
                    .lower();
            }}

            if (anchorDoc && negativeDoc) {{
                gElement.append('line')
                    .attr('class', 'triplet-line')
                    .attr('x1', window.xScale(anchorDoc.x))
                    .attr('y1', window.yScale(anchorDoc.y))
                    .attr('x2', window.xScale(negativeDoc.x))
                    .attr('y2', window.yScale(negativeDoc.y))
                    .attr('stroke', '#cc0000')
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', '5,5')
                    .attr('opacity', 0.5)
                    .lower();
            }}

            // Display triplet details in sidebar
            displayTripletDetails(triplet);
        }}

        function findTripletForDocument(docId) {{
            // Find first triplet containing this document
            const triplet = triplets.find(t =>
                t.anchor_id === docId || t.positive_id === docId || t.negative_id === docId
            );
            if (triplet) {{
                document.getElementById('tripletSelect').value = triplet.triplet_id;
                selectTriplet(triplet.triplet_id);
            }}
        }}

        function displayTripletDetails(triplet) {{
            const container = document.getElementById('selectedTriplet');
            container.innerHTML = `
                <div class="triplet">
                    <div class="triplet-header">Triplet ${{triplet.triplet_id}}</div>
                    ${{renderTripletRole(triplet, 'anchor')}}
                    ${{renderTripletRole(triplet, 'positive')}}
                    ${{renderTripletRole(triplet, 'negative')}}
                    ${{renderQualityAssessment(triplet)}}
                </div>
            `;
        }}

        function renderTripletRole(triplet, role) {{
            const doc = triplet[role];
            const isString = typeof doc === 'string';
            const textContent = isString ? doc : (doc.text || '');
            const imagePath = isString ? null : doc.image_path;

            const colorMap = {{
                anchor: '#0066cc',
                positive: '#00aa00',
                negative: '#cc0000'
            }};

            let html = `
                <div class="image-container ${{role}}" style="border-color: ${{colorMap[role]}};">
                    <h3 style="color: ${{colorMap[role]}};">${{role.charAt(0).toUpperCase() + role.slice(1)}}</h3>
                    <div style="padding: 10px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;
                                margin-bottom: 10px; font-family: monospace; font-size: 12px; line-height: 1.4;
                                max-height: 150px; overflow-y: auto; white-space: pre-wrap;">
                        ${{textContent}}
                    </div>
            `;

            // Add image if available
            if (imagePath && imagePath.startsWith('data:image')) {{
                html += `<img src="${{imagePath}}" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px;" />`;
            }}

            html += '</div>';
            return html;
        }}

        function renderQualityAssessment(triplet) {{
            if (triplet.quality_assessment_with_annotations) {{
                const qa = triplet.quality_assessment_with_annotations;
                return `
                    <div class="quality">
                        <strong>Quality (with annotations):</strong> ${{qa.label}}
                        <span class="quality-rating">${{qa.rating}}/5</span><br>
                        <div class="reasoning" style="margin-top: 8px; font-size: 11px;">
                            <strong>Reasoning:</strong> ${{qa.reasoning?.text || 'N/A'}}
                        </div>
                    </div>
                `;
            }}
            if (triplet.quality_assessment_without_annotations) {{
                const qa = triplet.quality_assessment_without_annotations;
                return `
                    <div class="quality">
                        <strong>Quality (without annotations):</strong> ${{qa.label}}
                        <span class="quality-rating">${{qa.rating}}/5</span><br>
                        <div class="reasoning" style="margin-top: 8px; font-size: 11px;">
                            <strong>Reasoning:</strong> ${{qa.reasoning?.text || 'N/A'}}
                        </div>
                    </div>
                `;
            }}
            if (triplet.quality_assessment) {{
                const qa = triplet.quality_assessment;
                return `
                    <div class="quality">
                        <strong>Quality:</strong> ${{qa.label}}
                        <span class="quality-rating">${{qa.rating}}/5</span><br>
                        <div class="reasoning" style="margin-top: 8px; font-size: 11px;">
                            <strong>Reasoning:</strong> ${{qa.reasoning?.text || 'N/A'}}
                        </div>
                    </div>
                `;
            }}
            return '';
        }}

        function zoomIn() {{
            svgElement.transition().call(zoom.scaleBy, 1.3);
        }}

        function zoomOut() {{
            svgElement.transition().call(zoom.scaleBy, 0.7);
        }}

        function resetZoom() {{
            svgElement.transition().call(zoom.transform, d3.zoomIdentity);
        }}

        function displayTriplets(triplets) {{
            const container = document.getElementById('triplets');
            container.innerHTML = '';

            triplets.forEach(triplet => {{
                const tripletDiv = document.createElement('div');
                tripletDiv.className = 'triplet';

                const header = document.createElement('div');
                header.className = 'triplet-header';
                header.textContent = `Triplet ${{triplet.triplet_id}}`;
                tripletDiv.appendChild(header);

                const imagesDiv = document.createElement('div');
                imagesDiv.className = 'images';

                // Add anchor, positive, negative
                ['anchor', 'positive', 'negative'].forEach(role => {{
                    if (!triplet[role]) return;

                    const imgContainer = document.createElement('div');
                    imgContainer.className = `image-container ${{role}}`;

                    const title = document.createElement('h3');
                    title.textContent = role.charAt(0).toUpperCase() + role.slice(1);
                    imgContainer.appendChild(title);

                    // Get document content
                    const doc = triplet[role];
                    const isString = typeof doc === 'string';
                    const textContent = isString ? doc : (doc.text || '');
                    const imagePath = isString ? null : doc.image_path;

                    // Always show text content
                    const textDiv = document.createElement('div');
                    textDiv.style.padding = '15px';
                    textDiv.style.background = '#f0f0f0';
                    textDiv.style.border = '1px solid #ccc';
                    textDiv.style.borderRadius = '4px';
                    textDiv.style.whiteSpace = 'pre-wrap';
                    textDiv.style.marginBottom = '10px';
                    textDiv.style.fontFamily = 'monospace';
                    textDiv.style.fontSize = '13px';
                    textDiv.style.lineHeight = '1.5';
                    textDiv.style.maxHeight = '300px';
                    textDiv.style.overflowY = 'auto';
                    textDiv.textContent = textContent;
                    imgContainer.appendChild(textDiv);

                    // Check if we have an image (data URI or URL)
                    const hasImage = imagePath && (
                        imagePath.startsWith('data:image') ||
                        imagePath.startsWith('http://') ||
                        imagePath.startsWith('https://')
                    );

                    // Show image if available (and not just placeholder text)
                    if (hasImage && textContent !== '<image>') {{
                        const img = document.createElement('img');
                        img.src = imagePath;
                        img.alt = role;
                        img.style.marginTop = '10px';
                        img.onerror = function() {{
                            this.style.display = 'none';
                        }};
                        imgContainer.appendChild(img);
                    }} else if (hasImage && textContent === '<image>') {{
                        // For image-only documents, show the image
                        const img = document.createElement('img');
                        img.src = imagePath;
                        img.alt = role;
                        imgContainer.insertBefore(img, textDiv);
                        textDiv.style.display = 'none'; // Hide the "<image>" placeholder
                    }}

                    // Show metadata if document is a dict with additional fields
                    if (!isString && typeof doc === 'object') {{
                        const metadata = document.createElement('div');
                        metadata.className = 'metadata';

                        let metaHtml = '';
                        if (doc.label) metaHtml += `<strong>Label:</strong> ${{doc.label}}<br>`;
                        if (doc.functional_type) metaHtml += `<strong>Type:</strong> ${{doc.functional_type}}<br>`;
                        if (doc.material) metaHtml += `<strong>Material:</strong> ${{doc.material}}<br>`;
                        if (doc.closure) metaHtml += `<strong>Closure:</strong> ${{doc.closure}}<br>`;
                        if (doc.heel_height) metaHtml += `<strong>Heel:</strong> ${{doc.heel_height}}<br>`;
                        if (doc.gender) metaHtml += `<strong>Gender:</strong> ${{doc.gender}}<br>`;
                        if (doc.toe_style) metaHtml += `<strong>Toe:</strong> ${{doc.toe_style}}<br>`;

                        if (metaHtml) {{
                            metadata.innerHTML = metaHtml;
                            imgContainer.appendChild(metadata);
                        }}
                    }}

                    imagesDiv.appendChild(imgContainer);
                }});

                tripletDiv.appendChild(imagesDiv);

                // Add quality assessment if available
                if (triplet.quality_assessment_with_annotations) {{
                    const quality = document.createElement('div');
                    quality.className = 'quality';
                    const qa = triplet.quality_assessment_with_annotations;
                    quality.innerHTML = `
                        <strong>Quality (with annotations):</strong> ${{qa.label}} <span class="quality-rating">${{qa.rating}}/5</span><br>
                        <div class="reasoning"><strong>Reasoning:</strong> ${{qa.reasoning?.text || 'N/A'}}</div>
                    `;
                    tripletDiv.appendChild(quality);
                }}

                if (triplet.quality_assessment_without_annotations) {{
                    const quality = document.createElement('div');
                    quality.className = 'quality';
                    const qa = triplet.quality_assessment_without_annotations;
                    quality.innerHTML = `
                        <strong>Quality (without annotations):</strong> ${{qa.label}} <span class="quality-rating">${{qa.rating}}/5</span><br>
                        <div class="reasoning"><strong>Reasoning:</strong> ${{qa.reasoning?.text || 'N/A'}}</div>
                    `;
                    tripletDiv.appendChild(quality);
                }}

                if (triplet.quality_assessment && !triplet.quality_assessment_with_annotations && !triplet.quality_assessment_without_annotations) {{
                    const quality = document.createElement('div');
                    quality.className = 'quality';
                    const qa = triplet.quality_assessment;
                    quality.innerHTML = `
                        <strong>Quality:</strong> ${{qa.label}} <span class="quality-rating">${{qa.rating}}/5</span><br>
                        <div class="reasoning"><strong>Reasoning:</strong> ${{qa.reasoning?.text || 'N/A'}}</div>
                    `;
                    tripletDiv.appendChild(quality);
                }}

                container.appendChild(tripletDiv);
            }});
        }}
    </script>
</body>
</html>"""
