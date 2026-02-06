/**
 * HeatmapView renderer for pairwise similarity matrix visualization.
 * Shows a square heatmap of document similarities - friendly to BM25 and embedding-based methods.
 *
 * Supports two input formats:
 * 1. Embeddings (NxD): Computes cosine similarity matrix on the fly
 * 2. Pre-computed similarity matrix (NxN): Uses directly (e.g., from BM25)
 */

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { TripletData, MultiviewManifest } from '../types/manifest';

interface HeatmapViewProps {
  embeddings: Float32Array;
  documents: string[];
  triplets?: TripletData[];
  manifest?: MultiviewManifest;
  width?: number;
  height?: number;
  maxDocs?: number;
}

function cosineSimilarity(a: Float32Array, b: Float32Array, dim: number, offsetA: number, offsetB: number): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < dim; i++) {
    const valA = a[offsetA + i];
    const valB = b[offsetB + i];
    dotProduct += valA * valB;
    normA += valA * valA;
    normB += valB * valB;
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function l1Distance(a: Float32Array, b: Float32Array, dim: number, offsetA: number, offsetB: number): number {
  let sum = 0;

  for (let i = 0; i < dim; i++) {
    const valA = a[offsetA + i];
    const valB = b[offsetB + i];
    sum += Math.abs(valA - valB);
  }

  return sum;
}

function computeSimilarityMatrix(
  embeddings: Float32Array,
  nDocs: number,
  embeddingDim: number,
  usePseudologit: boolean = false
): Float32Array {
  const similarityMatrix = new Float32Array(nDocs * nDocs);

  for (let i = 0; i < nDocs; i++) {
    for (let j = 0; j < nDocs; j++) {
      if (i === j) {
        // For L1 distance, diagonal is 0 (distance to self)
        // For cosine similarity, diagonal is 1 (perfect similarity)
        similarityMatrix[i * nDocs + j] = usePseudologit ? 0.0 : 1.0;
      } else {
        const value = usePseudologit
          ? l1Distance(embeddings, embeddings, embeddingDim, i * embeddingDim, j * embeddingDim)
          : cosineSimilarity(embeddings, embeddings, embeddingDim, i * embeddingDim, j * embeddingDim);
        similarityMatrix[i * nDocs + j] = value;
      }
    }
  }

  return similarityMatrix;
}

function isPrecomputedSimilarityMatrix(embeddings: Float32Array, nDocs: number): boolean {
  // Check if embeddings array is NxN (precomputed matrix) vs NxD (actual embeddings)
  // If the length equals nDocs * nDocs, it's likely a precomputed similarity matrix
  return embeddings.length === nDocs * nDocs;
}

export const HeatmapView: React.FC<HeatmapViewProps> = ({
  embeddings,
  documents,
  triplets,
  manifest,
  width = 800,
  height = 800,
  maxDocs = 100,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [isComputing, setIsComputing] = useState(true);
  const [selectedTripletIndex, setSelectedTripletIndex] = useState<number>(0);
  const [showTriplets, setShowTriplets] = useState<boolean>(true);
  const [hoveredTripletRole, setHoveredTripletRole] = useState<'anchor' | 'positive' | 'negative' | null>(null);
  const hasTriplets = triplets && triplets.length > 0;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Limit number of documents for performance
    const nDocs = Math.min(documents.length, maxDocs);
    if (documents.length > maxDocs) {
      console.warn(`Limiting heatmap to ${maxDocs} documents (original: ${documents.length})`);
    }

    // Detect if we have a precomputed similarity matrix or embeddings
    const isPrecomputed = isPrecomputedSimilarityMatrix(embeddings, documents.length);
    const embeddingDim = isPrecomputed ? documents.length : embeddings.length / documents.length;

    // Check if this is a pseudologit method (use L1 distance instead of cosine similarity)
    const usePseudologit = manifest?.method?.toLowerCase().includes('pseudologit') || false;

    // Show loading message
    const metricName = isPrecomputed ? 'similarity matrix' : (usePseudologit ? 'L1 distance matrix' : 'similarity matrix');
    const loadingText = svg.append('text')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', '#666')
      .text(`Loading ${metricName}...`);

    // Get current triplet if available
    const currentTriplet = hasTriplets && showTriplets && triplets.length > 0
      ? triplets[selectedTripletIndex]
      : null;

    // Compute or extract similarity matrix asynchronously to avoid blocking UI
    setTimeout(() => {
      try {
        let similarityMatrix: Float32Array;

        if (isPrecomputed) {
          // Use precomputed matrix directly (e.g., from BM25)
          console.log('Using precomputed similarity matrix (e.g., BM25)');
          // Extract the subset if we're limiting documents
          if (nDocs < documents.length) {
            similarityMatrix = new Float32Array(nDocs * nDocs);
            for (let i = 0; i < nDocs; i++) {
              for (let j = 0; j < nDocs; j++) {
                similarityMatrix[i * nDocs + j] = embeddings[i * documents.length + j];
              }
            }
          } else {
            similarityMatrix = new Float32Array(embeddings);
          }
        } else {
          // Compute from embeddings
          const metricName = usePseudologit ? 'L1 distance' : 'cosine similarity';
          console.log(`Computing ${metricName} from embeddings`);
          similarityMatrix = computeSimilarityMatrix(embeddings, nDocs, embeddingDim, usePseudologit);
        }

        // Zero out diagonal (self-similarity) for clearer visualization
        // For L1 distance, diagonal is already 0; for cosine similarity, set to 0
        if (!usePseudologit) {
          for (let i = 0; i < nDocs; i++) {
            similarityMatrix[i * nDocs + i] = 0;
          }
        }

        setIsComputing(false);

        // Clear loading message
        loadingText.remove();

        // Create margins
        const margin = { top: 60, right: 60, bottom: 60, left: 60 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        // Create main group
        const g = svg.append('g')
          .attr('transform', `translate(${margin.left},${margin.top})`);

        // Cell size
        const cellSize = Math.min(innerWidth / nDocs, innerHeight / nDocs);

        // Compute min/max for color scale from actual data
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let i = 0; i < similarityMatrix.length; i++) {
          const val = similarityMatrix[i];
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }

        // Color scale using actual data range
        const colorScale = d3.scaleSequential(d3.interpolateViridis)
          .domain([minVal, maxVal]);

        // Helper to check cell highlight type based on triplet hover
        const getCellHighlightType = (i: number, j: number): 'positive' | 'negative' | null => {
          if (!currentTriplet || !hoveredTripletRole) return null;

          const anchorIdx = currentTriplet.anchor_id;
          const posIdx = currentTriplet.positive_id;
          const negIdx = currentTriplet.negative_id;

          // Check both (i,j) and (j,i) for symmetry
          const matchesAnchPos = (i === anchorIdx && j === posIdx) || (i === posIdx && j === anchorIdx);
          const matchesAnchNeg = (i === anchorIdx && j === negIdx) || (i === negIdx && j === anchorIdx);

          if (hoveredTripletRole === 'anchor') {
            if (matchesAnchPos) return 'positive';
            if (matchesAnchNeg) return 'negative';
          } else if (hoveredTripletRole === 'positive') {
            if (matchesAnchPos) return 'positive';
          } else if (hoveredTripletRole === 'negative') {
            if (matchesAnchNeg) return 'negative';
          }
          return null;
        };

        // Create cells
        const cells = g.selectAll('.cell')
          .data(Array.from({ length: nDocs * nDocs }, (_, idx) => {
            const i = Math.floor(idx / nDocs);
            const j = idx % nDocs;
            return {
              i,
              j,
              similarity: similarityMatrix[i * nDocs + j],
            };
          }))
          .join('rect')
          .attr('class', 'cell')
          .attr('x', (d) => d.j * cellSize)
          .attr('y', (d) => d.i * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', (d) => colorScale(d.similarity))
          .attr('stroke', (d) => {
            const highlightType = getCellHighlightType(d.i, d.j);
            if (highlightType === 'positive') return '#00aa00';
            if (highlightType === 'negative') return '#cc0000';
            return 'none';
          })
          .attr('stroke-width', (d) => getCellHighlightType(d.i, d.j) ? 3 : 0)
          .attr('opacity', (d) => {
            if (!hoveredTripletRole) return 1;
            return getCellHighlightType(d.i, d.j) ? 1 : 0.3;
          })
          .style('cursor', 'pointer')
          .on('mouseenter', function (event, d) {
            if (!getCellHighlightType(d.i, d.j)) {
              d3.select(this)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            }

            const docI = documents[d.i].length > 100 ? documents[d.i].slice(0, 100) + '...' : documents[d.i];
            const docJ = documents[d.j].length > 100 ? documents[d.j].slice(0, 100) + '...' : documents[d.j];

            const metricLabel = usePseudologit ? 'Distance' : 'Similarity';
            setTooltip({
              x: event.pageX,
              y: event.pageY,
              text: `Doc ${d.i} ↔ Doc ${d.j}\n${metricLabel}: ${d.similarity.toFixed(3)}\n\n[${d.i}]: ${docI}\n\n[${d.j}]: ${docJ}`,
            });
          })
          .on('mouseleave', function (event, d) {
            if (!getCellHighlightType(d.i, d.j)) {
              d3.select(this)
                .attr('stroke', 'none');
            }
            setTooltip(null);
          });

        // Add title
        let titleText: string;
        if (isPrecomputed) {
          titleText = `Pairwise Similarity Matrix (${nDocs} documents)`;
        } else if (usePseudologit) {
          titleText = `L1 Distance Matrix (${nDocs} documents)`;
        } else {
          titleText = `Cosine Similarity Matrix (${nDocs} documents)`;
        }

        svg.append('text')
          .attr('x', width / 2)
          .attr('y', 25)
          .attr('text-anchor', 'middle')
          .attr('font-size', '16px')
          .attr('font-weight', 'bold')
          .text(titleText);

        // Add color scale legend
        const legendWidth = 200;
        const legendHeight = 15;
        const legendMargin = 10;

        const legendScale = d3.scaleLinear()
          .domain([minVal, maxVal])
          .range([0, legendWidth]);

        // Use appropriate tick format based on value range
        const tickFormat = maxVal > 10 ? d3.format('.0f') : d3.format('.2f');
        const legendAxis = d3.axisBottom(legendScale)
          .ticks(5)
          .tickFormat(tickFormat);

        const legend = svg.append('g')
          .attr('transform', `translate(${width / 2 - legendWidth / 2},${height - 30})`);

        // Legend gradient
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
          .attr('id', 'legend-gradient')
          .attr('x1', '0%')
          .attr('x2', '100%');

        // Create gradient stops across the data range
        const numStops = 11;
        const stopValues = Array.from({ length: numStops }, (_, i) => {
          const t = i / (numStops - 1);
          return minVal + t * (maxVal - minVal);
        });

        gradient.selectAll('stop')
          .data(stopValues)
          .join('stop')
          .attr('offset', (d, i) => `${(i / (numStops - 1)) * 100}%`)
          .attr('stop-color', (d) => colorScale(d));

        legend.append('rect')
          .attr('width', legendWidth)
          .attr('height', legendHeight)
          .style('fill', 'url(#legend-gradient)');

        legend.append('g')
          .attr('transform', `translate(0,${legendHeight})`)
          .call(legendAxis);

        let legendLabel: string;
        if (isPrecomputed) {
          legendLabel = 'Similarity Score';
        } else if (usePseudologit) {
          legendLabel = 'L1 Distance';
        } else {
          legendLabel = 'Cosine Similarity';
        }

        legend.append('text')
          .attr('x', legendWidth / 2)
          .attr('y', -5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .text(legendLabel);

      } catch (error) {
        console.error('Error rendering heatmap:', error);
        svg.selectAll('*').remove();
        svg.append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('fill', '#f00')
          .text('Error rendering heatmap');
      }
    }, 100);

  }, [embeddings, documents, width, height, maxDocs, triplets, selectedTripletIndex, showTriplets, hoveredTripletRole, manifest]);

  // Get current triplet for display
  const currentTriplet = hasTriplets && showTriplets && triplets.length > 0
    ? triplets[selectedTripletIndex]
    : null;

  // Helper to get document text from triplet element
  const getDocText = (doc: any): string => {
    if (typeof doc === 'string') return doc;
    if (doc && typeof doc === 'object' && doc.text) return doc.text;
    return JSON.stringify(doc);
  };

  return (
    <div style={{ display: 'flex', gap: '20px' }}>
      <div style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ border: '1px solid #ccc', background: '#fafafa' }}
        />
        {tooltip && (
          <div
            style={{
              position: 'fixed',
              left: tooltip.x + 10,
              top: tooltip.y + 10,
              background: 'rgba(0, 0, 0, 0.9)',
              color: 'white',
              padding: '12px 16px',
              borderRadius: '4px',
              fontSize: '11px',
              fontFamily: 'monospace',
              maxWidth: '400px',
              pointerEvents: 'none',
              zIndex: 1000,
              whiteSpace: 'pre-wrap',
              wordWrap: 'break-word',
            }}
          >
            {tooltip.text}
          </div>
        )}
        {documents.length > maxDocs && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(255, 193, 7, 0.9)',
            color: '#000',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 'bold',
          }}>
            Showing {maxDocs} of {documents.length} documents
          </div>
        )}
      </div>

      {hasTriplets && (
        <div style={{
          width: '350px',
          maxHeight: `${height}px`,
          overflowY: 'auto',
          padding: '15px',
          background: '#f9f9f9',
          borderRadius: '4px',
          border: '1px solid #ddd',
        }}>
          <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600, flex: 1 }}>
              Triplets
            </h3>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showTriplets}
                onChange={(e) => setShowTriplets(e.target.checked)}
              />
              Show
            </label>
          </div>

          {showTriplets && triplets.length > 0 && (
            <>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '15px', alignItems: 'center', justifyContent: 'center' }}>
                <button
                  onClick={() => setSelectedTripletIndex(Math.max(0, selectedTripletIndex - 1))}
                  disabled={selectedTripletIndex === 0}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    background: '#fff',
                    cursor: selectedTripletIndex === 0 ? 'not-allowed' : 'pointer',
                    fontSize: '13px',
                  }}
                >
                  ← Prev
                </button>
                <span style={{ fontSize: '13px', color: '#666' }}>
                  {selectedTripletIndex + 1} / {triplets.length}
                </span>
                <button
                  onClick={() => setSelectedTripletIndex(Math.min(triplets.length - 1, selectedTripletIndex + 1))}
                  disabled={selectedTripletIndex === triplets.length - 1}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    background: '#fff',
                    cursor: selectedTripletIndex === triplets.length - 1 ? 'not-allowed' : 'pointer',
                    fontSize: '13px',
                  }}
                >
                  Next →
                </button>
              </div>

              {currentTriplet && (
                <div style={{ fontSize: '13px' }}>
                  {/* Margin and Correctness Info */}
                  {currentTriplet.margin !== undefined && (
                    <div style={{
                      marginBottom: '12px',
                      padding: '10px',
                      background: currentTriplet.correct ? '#e8f5e9' : '#ffebee',
                      border: `2px solid ${currentTriplet.correct ? '#4caf50' : '#f44336'}`,
                      borderRadius: '4px',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <span style={{
                          fontWeight: 600,
                          color: currentTriplet.correct ? '#2e7d32' : '#c62828',
                          fontSize: '13px',
                        }}>
                          {currentTriplet.correct ? '✓ CORRECT' : '✗ INCORRECT'}
                        </span>
                        {currentTriplet.is_tie && (
                          <span style={{
                            padding: '2px 6px',
                            borderRadius: '3px',
                            fontSize: '11px',
                            fontWeight: 600,
                            background: '#ff9800',
                            color: '#fff',
                          }}>
                            TIE
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: '12px', color: '#555' }}>
                        <strong>Margin:</strong>{' '}
                        <span style={{
                          fontWeight: 600,
                          color: currentTriplet.margin > 0 ? '#2e7d32' : '#c62828',
                          fontFamily: 'monospace',
                        }}>
                          {currentTriplet.margin > 0 ? '+' : ''}{currentTriplet.margin.toFixed(4)}
                        </span>
                      </div>
                      <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
                        Positive: {currentTriplet.positive_score?.toFixed(4)} | Negative: {currentTriplet.negative_score?.toFixed(4)}
                      </div>
                    </div>
                  )}

                  <div
                    style={{
                      marginBottom: '15px',
                      padding: '10px',
                      background: hoveredTripletRole === 'anchor' ? '#d1e7fd' : '#e3f2fd',
                      borderRadius: '4px',
                      borderLeft: hoveredTripletRole === 'anchor' ? '4px solid #0066cc' : '3px solid #0066cc',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                    onMouseEnter={() => setHoveredTripletRole('anchor')}
                    onMouseLeave={() => setHoveredTripletRole(null)}
                  >
                    <strong style={{ color: '#0066cc' }}>Anchor {hoveredTripletRole === 'anchor' && '👈'}</strong>
                    <div style={{ marginTop: '5px', maxHeight: '100px', overflowY: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px' }}>
                      {getDocText(currentTriplet.anchor).slice(0, 300)}
                      {getDocText(currentTriplet.anchor).length > 300 ? '...' : ''}
                    </div>
                  </div>

                  <div
                    style={{
                      marginBottom: '15px',
                      padding: '10px',
                      background: hoveredTripletRole === 'positive' ? '#d4f4d7' : '#e8f5e9',
                      borderRadius: '4px',
                      borderLeft: hoveredTripletRole === 'positive' ? '4px solid #00aa00' : '3px solid #00aa00',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                    onMouseEnter={() => setHoveredTripletRole('positive')}
                    onMouseLeave={() => setHoveredTripletRole(null)}
                  >
                    <strong style={{ color: '#00aa00' }}>Positive {hoveredTripletRole === 'positive' && '👈'}</strong>
                    <div style={{ marginTop: '5px', maxHeight: '100px', overflowY: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px' }}>
                      {getDocText(currentTriplet.positive).slice(0, 300)}
                      {getDocText(currentTriplet.positive).length > 300 ? '...' : ''}
                    </div>
                  </div>

                  <div
                    style={{
                      marginBottom: '15px',
                      padding: '10px',
                      background: hoveredTripletRole === 'negative' ? '#fdd4d4' : '#ffebee',
                      borderRadius: '4px',
                      borderLeft: hoveredTripletRole === 'negative' ? '4px solid #cc0000' : '3px solid #cc0000',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                    onMouseEnter={() => setHoveredTripletRole('negative')}
                    onMouseLeave={() => setHoveredTripletRole(null)}
                  >
                    <strong style={{ color: '#cc0000' }}>Negative {hoveredTripletRole === 'negative' && '👈'}</strong>
                    <div style={{ marginTop: '5px', maxHeight: '100px', overflowY: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px' }}>
                      {getDocText(currentTriplet.negative).slice(0, 300)}
                      {getDocText(currentTriplet.negative).length > 300 ? '...' : ''}
                    </div>
                  </div>

                  {currentTriplet.quality_assessment_with_annotations && (
                    <div style={{ padding: '10px', background: '#fff3cd', borderRadius: '4px', fontSize: '12px' }}>
                      <strong>Quality: </strong>
                      {currentTriplet.quality_assessment_with_annotations.label} ({currentTriplet.quality_assessment_with_annotations.rating}/5)
                      {currentTriplet.quality_assessment_with_annotations.reasoning && (
                        <div style={{ marginTop: '8px', maxHeight: '150px', overflowY: 'auto', fontSize: '11px', color: '#555' }}>
                          {currentTriplet.quality_assessment_with_annotations.reasoning.text.slice(0, 500)}
                          {currentTriplet.quality_assessment_with_annotations.reasoning.text.length > 500 ? '...' : ''}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};
