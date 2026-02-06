/**
 * DendrogramView renderer for hierarchical clustering visualization.
 *
 * Renders a proper dendrogram with:
 * - U-shaped branches at actual merge heights
 * - Cluster-colored branches
 * - Grid layout for leaf images (multiple rows)
 * - Colored image borders by cluster
 * - Matplotlib-quality styling
 */

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

interface DendrogramViewProps {
  coords: Float32Array;
  documents: string[];
  thumbnailUrls?: (string | null)[];
  displayMode?: 'points' | 'thumbnails';
  linkageMatrix?: Float32Array;
  dendrogramImageUrl?: string;  // URL to matplotlib-generated dendrogram image
  width?: number;
  height?: number;
  imagesPerRow?: number;
  numClusters?: number;
}

interface DendrogramNode {
  id: number;
  left?: number;
  right?: number;
  distance: number;
  count: number;
  x?: number;  // Computed x position
  y?: number;  // Computed y position (height)
}

interface DendrogramLink {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cluster?: number;
}

/**
 * Build dendrogram structure from linkage matrix.
 * Returns nodes array where nodes[i] is the i-th node (0..n-1 are leaves, n+ are internal)
 */
function buildDendrogramNodes(linkage: Float32Array, n: number): DendrogramNode[] {
  const numMerges = linkage.length / 4;
  const nodes: DendrogramNode[] = [];

  // Create leaf nodes
  for (let i = 0; i < n; i++) {
    nodes.push({
      id: i,
      distance: 0,
      count: 1,
    });
  }

  // Build internal nodes from linkage matrix
  for (let i = 0; i < numMerges; i++) {
    const idx1 = Math.floor(linkage[i * 4]);
    const idx2 = Math.floor(linkage[i * 4 + 1]);
    const distance = linkage[i * 4 + 2];
    const count = Math.floor(linkage[i * 4 + 3]);

    nodes.push({
      id: n + i,
      left: idx1,
      right: idx2,
      distance,
      count,
    });
  }

  return nodes;
}

/**
 * Compute dendrogram layout (leaf ordering and positions).
 * This implements scipy's dendrogram algorithm.
 */
function computeDendrogramLayout(
  nodes: DendrogramNode[],
  n: number,
  leafSpacing: number = 10
): { leafOrder: number[]; links: DendrogramLink[] } {
  const rootIdx = nodes.length - 1;

  // Compute leaf ordering (post-order traversal)
  const leafOrder: number[] = [];
  const leafPositions = new Map<number, number>();
  let currentX = 0;

  function traverse(nodeIdx: number): number {
    const node = nodes[nodeIdx];

    if (node.left === undefined) {
      // Leaf node
      leafOrder.push(nodeIdx);
      const x = currentX * leafSpacing;
      leafPositions.set(nodeIdx, x);
      nodes[nodeIdx].x = x;
      nodes[nodeIdx].y = 0;
      currentX++;
      return x;
    }

    // Internal node - recursively traverse children
    const leftX = traverse(node.left);
    const rightX = traverse(node.right!);

    // Position at midpoint of children
    const x = (leftX + rightX) / 2;
    nodes[nodeIdx].x = x;
    nodes[nodeIdx].y = node.distance;

    return x;
  }

  traverse(rootIdx);

  // Generate dendrogram links (U-shaped connections)
  const links: DendrogramLink[] = [];

  function generateLinks(nodeIdx: number) {
    const node = nodes[nodeIdx];
    if (node.left === undefined) return;

    const leftChild = nodes[node.left];
    const rightChild = nodes[node.right!];

    // U-shaped connection:
    // 1. Horizontal line at merge height
    links.push({
      x1: leftChild.x!,
      y1: node.y!,
      x2: rightChild.x!,
      y2: node.y!,
    });

    // 2. Vertical line from left child to merge height
    links.push({
      x1: leftChild.x!,
      y1: leftChild.y!,
      x2: leftChild.x!,
      y2: node.y!,
    });

    // 3. Vertical line from right child to merge height
    links.push({
      x1: rightChild.x!,
      y1: rightChild.y!,
      x2: rightChild.x!,
      y2: node.y!,
    });

    // Recurse to children
    generateLinks(node.left);
    generateLinks(node.right!);
  }

  generateLinks(rootIdx);

  return { leafOrder, links };
}

/**
 * Assign cluster labels by cutting dendrogram at a specific height.
 */
function assignClusters(
  nodes: DendrogramNode[],
  numClusters: number,
  linkage: Float32Array,
  n: number
): Map<number, number> {
  // Find cut height that creates numClusters
  const numMerges = linkage.length / 4;
  const mergeIdx = numMerges - numClusters + 1;
  let cutHeight = 0;

  if (mergeIdx >= 0 && mergeIdx < numMerges) {
    cutHeight = linkage[mergeIdx * 4 + 2] + 1e-10;
  }

  const clusterMap = new Map<number, number>();
  let currentCluster = 0;

  function assignClusterToSubtree(nodeIdx: number) {
    const node = nodes[nodeIdx];

    if (node.left === undefined) {
      // Leaf node
      clusterMap.set(nodeIdx, currentCluster);
      return;
    }

    // If this node is above cut height, assign all descendants to current cluster
    if (node.distance >= cutHeight) {
      const clusterId = currentCluster;
      currentCluster++;

      function markSubtree(idx: number) {
        const n = nodes[idx];
        if (n.left === undefined) {
          clusterMap.set(idx, clusterId);
        } else {
          markSubtree(n.left);
          markSubtree(n.right!);
        }
      }

      markSubtree(nodeIdx);
    } else {
      // Below cut height, continue traversing
      assignClusterToSubtree(node.left);
      assignClusterToSubtree(node.right!);
    }
  }

  assignClusterToSubtree(nodes.length - 1);

  return clusterMap;
}

/**
 * Assign cluster colors to dendrogram links.
 */
function assignLinkClusters(
  links: DendrogramLink[],
  nodes: DendrogramNode[],
  clusterMap: Map<number, number>,
  cutHeight: number
): void {
  // For each link, find which cluster it belongs to
  // by finding a leaf node reachable through this link
  links.forEach(link => {
    // Find a node at this position
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      if (node.x === link.x1 && node.y === link.y1) {
        // Found start node - trace to a leaf
        let currentIdx = i;
        while (nodes[currentIdx].left !== undefined) {
          currentIdx = nodes[currentIdx].left!;
        }
        link.cluster = clusterMap.get(currentIdx);
        break;
      }
    }
  });
}

/**
 * Get cluster colors using d3 color schemes.
 */
function getClusterColors(numClusters: number): string[] {
  if (numClusters <= 10) {
    return d3.schemeCategory10.slice(0, numClusters);
  } else {
    const colors = [];
    for (let i = 0; i < numClusters; i++) {
      const hue = (i * 360) / numClusters;
      colors.push(d3.hsl(hue, 0.7, 0.5).toString());
    }
    return colors;
  }
}

export const DendrogramView: React.FC<DendrogramViewProps> = ({
  coords,
  documents,
  thumbnailUrls,
  displayMode = 'thumbnails',
  linkageMatrix,
  dendrogramImageUrl,
  width = 1200,
  height = 600,
  imagesPerRow,
  numClusters,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const hasThumbnailsAvailable = thumbnailUrls && thumbnailUrls.some(url => url !== null);
  const useThumbnails = displayMode === 'thumbnails' && hasThumbnailsAvailable;

  // If we have a matplotlib-generated dendrogram image, just display that
  if (dendrogramImageUrl) {
    return (
      <div style={{ position: 'relative', width: '100%', overflow: 'auto' }}>
        <img
          src={dendrogramImageUrl}
          alt="Hierarchical Clustering Dendrogram"
          style={{
            width: '100%',
            height: 'auto',
            border: '1px solid #ddd',
            borderRadius: '4px',
            background: 'white',
          }}
        />
      </div>
    );
  }

  // Fallback: render D3 dendrogram if no matplotlib image available
  useEffect(() => {
    if (!svgRef.current || !linkageMatrix) {
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const n = documents.length;

    try {
      // Build dendrogram structure
      const nodes = buildDendrogramNodes(linkageMatrix, n);

      // Compute layout
      const leafSpacing = 10;
      const { leafOrder, links } = computeDendrogramLayout(nodes, n, leafSpacing);

      // Auto-calculate number of clusters
      const effectiveNumClusters = numClusters || Math.max(3, Math.min(20, Math.floor(n / 10)));

      // Assign cluster labels
      const clusterMap = assignClusters(nodes, effectiveNumClusters, linkageMatrix, n);
      const clusterColors = getClusterColors(effectiveNumClusters);

      // Find cut height for coloring
      const numMerges = linkageMatrix.length / 4;
      const mergeIdx = numMerges - effectiveNumClusters + 1;
      const cutHeight = mergeIdx >= 0 && mergeIdx < numMerges
        ? linkageMatrix[mergeIdx * 4 + 2] + 1e-10
        : 0;

      // Assign clusters to links
      assignLinkClusters(links, nodes, clusterMap, cutHeight);

      // Calculate dimensions
      const margin = { top: 40, right: 40, bottom: 200, left: 40 };
      const dendrogramWidth = width - margin.left - margin.right;
      const dendrogramHeight = 300;  // Fixed dendrogram height

      // Scale for dendrogram
      const maxX = (n - 1) * leafSpacing;
      const maxY = Math.max(...nodes.map(n => n.y || 0));

      const xScale = d3.scaleLinear()
        .domain([0, maxX])
        .range([0, dendrogramWidth]);

      const yScale = d3.scaleLinear()
        .domain([0, maxY])
        .range([dendrogramHeight, 0]);  // Inverted: 0 at bottom

      // Create main group
      const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

      // Draw dendrogram links
      links.forEach(link => {
        const clusterId = link.cluster;
        const color = clusterId !== undefined
          ? clusterColors[clusterId % clusterColors.length]
          : '#888888';

        // Only color links above cut height
        const linkMaxY = Math.max(link.y1, link.y2);
        const finalColor = linkMaxY >= cutHeight ? color : '#888888';

        g.append('line')
          .attr('x1', xScale(link.x1))
          .attr('y1', yScale(link.y1))
          .attr('x2', xScale(link.x2))
          .attr('y2', yScale(link.y2))
          .attr('stroke', finalColor)
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.8);
      });

      // Calculate grid layout for images
      const effectiveImagesPerRow = imagesPerRow || Math.min(n, Math.max(10, Math.floor(dendrogramWidth / 70)));
      const numRows = Math.ceil(n / effectiveImagesPerRow);
      const imageSize = useThumbnails ? 50 : 8;
      const gridSpacingX = dendrogramWidth / effectiveImagesPerRow;
      const gridSpacingY = 70;
      const gridStartY = dendrogramHeight + 50;

      console.log(`Dendrogram: ${n} leaves in ${numRows} rows (${effectiveImagesPerRow} per row), ${effectiveNumClusters} clusters`);

      // Draw colored bands for clusters
      if (useThumbnails) {
        const bandHeight = 8;
        const bandY = dendrogramHeight + 5;

        let i = 0;
        while (i < leafOrder.length) {
          const currentLeafIdx = leafOrder[i];
          const currentCluster = clusterMap.get(currentLeafIdx);

          let j = i;
          while (j < leafOrder.length && clusterMap.get(leafOrder[j]) === currentCluster) {
            j++;
          }

          const xStart = xScale(nodes[leafOrder[i]].x!);
          const xEnd = xScale(nodes[leafOrder[j - 1]].x!);
          const clusterColor = currentCluster !== undefined
            ? clusterColors[currentCluster % clusterColors.length]
            : '#ccc';

          g.append('rect')
            .attr('x', xStart - 3)
            .attr('y', bandY)
            .attr('width', xEnd - xStart + 6)
            .attr('height', bandHeight)
            .attr('fill', clusterColor)
            .attr('opacity', 0.4)
            .attr('rx', 2);

          i = j;
        }
      }

      // Render leaf images/points in grid layout
      leafOrder.forEach((leafIdx, idx) => {
        const gridRow = Math.floor(idx / effectiveImagesPerRow);
        const gridCol = idx % effectiveImagesPerRow;
        const xPos = gridCol * gridSpacingX + gridSpacingX / 2;
        const yPos = gridStartY + gridRow * gridSpacingY;

        const clusterId = clusterMap.get(leafIdx);
        const borderColor = clusterId !== undefined
          ? clusterColors[clusterId % clusterColors.length]
          : '#ccc';

        if (useThumbnails && thumbnailUrls && thumbnailUrls[leafIdx]) {
          const thumbnailUrl = thumbnailUrls[leafIdx];

          const imageGroup = g.append('g')
            .attr('class', 'leaf-image')
            .attr('transform', `translate(${xPos}, ${yPos})`)
            .style('cursor', 'pointer');

          // Border
          imageGroup.append('rect')
            .attr('x', -imageSize / 2 - 2)
            .attr('y', -imageSize / 2 - 2)
            .attr('width', imageSize + 4)
            .attr('height', imageSize + 4)
            .attr('fill', 'white')
            .attr('stroke', borderColor)
            .attr('stroke-width', 2.5)
            .attr('rx', 3);

          // Image
          imageGroup.append('image')
            .attr('x', -imageSize / 2)
            .attr('y', -imageSize / 2)
            .attr('width', imageSize)
            .attr('height', imageSize)
            .attr('href', thumbnailUrl)
            .attr('preserveAspectRatio', 'xMidYMid slice')
            .style('clip-path', 'inset(0 round 2px)');

          // Hover effects
          imageGroup
            .on('mouseenter', function (event) {
              d3.select(this)
                .transition()
                .duration(150)
                .attr('transform', `translate(${xPos}, ${yPos}) scale(1.4)`);

              d3.select(this).raise();

              setTooltip({
                x: event.pageX,
                y: event.pageY,
                text: documents[leafIdx] || `Doc ${leafIdx}`,
              });
            })
            .on('mouseleave', function () {
              d3.select(this)
                .transition()
                .duration(150)
                .attr('transform', `translate(${xPos}, ${yPos}) scale(1)`);

              setTooltip(null);
            });
        } else {
          // Colored circles
          g.append('circle')
            .attr('cx', xPos)
            .attr('cy', yPos)
            .attr('r', imageSize)
            .attr('fill', borderColor)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('mouseenter', function (event) {
              d3.select(this)
                .transition()
                .duration(150)
                .attr('r', imageSize * 1.5);

              setTooltip({
                x: event.pageX,
                y: event.pageY,
                text: documents[leafIdx] || `Doc ${leafIdx}`,
              });
            })
            .on('mouseleave', function () {
              d3.select(this)
                .transition()
                .duration(150)
                .attr('r', imageSize);

              setTooltip(null);
            });
        }
      });

    } catch (error) {
      console.error('Error rendering dendrogram:', error);
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .text('Error rendering dendrogram')
        .style('fill', '#f00');
    }

  }, [coords, documents, thumbnailUrls, useThumbnails, linkageMatrix, width, height, imagesPerRow, numClusters]);

  // Calculate dynamic height
  const n = documents.length;
  const effectiveImagesPerRow = imagesPerRow || Math.min(n, Math.max(10, Math.floor((width - 80) / 70)));
  const numRows = Math.ceil(n / effectiveImagesPerRow);
  const gridHeight = numRows * 70 + 100;
  const totalHeight = 300 + 50 + gridHeight + 80;  // dendrogram + gap + grid + margins

  return (
    <div style={{ position: 'relative', width: '100%', overflow: 'auto' }}>
      <svg
        ref={svgRef}
        width={width}
        height={totalHeight}
        style={{
          border: '1px solid #ddd',
          background: 'white',
          borderRadius: '4px',
        }}
      />
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x + 10,
            top: tooltip.y + 10,
            background: 'rgba(0, 0, 0, 0.85)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            maxWidth: '400px',
            pointerEvents: 'none',
            zIndex: 1000,
            wordWrap: 'break-word',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          }}
        >
          {tooltip.text.length > 300 ? tooltip.text.slice(0, 300) + '...' : tooltip.text}
        </div>
      )}
    </div>
  );
};
