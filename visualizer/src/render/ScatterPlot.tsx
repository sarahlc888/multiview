/**
 * ScatterPlot renderer for t-SNE, SOM, UMAP, and PCA visualizations.
 */

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { TripletData } from '../types/manifest';

interface ScatterPlotProps {
  coords: Float32Array;
  documents: string[];
  thumbnailUrls?: (string | null)[];
  triplets?: TripletData[];
  displayMode?: 'points' | 'thumbnails';
  width?: number;
  height?: number;
  onSelectDocument?: (index: number | null) => void;
  highlightedIndex?: number | null;
  onHoverDocument?: (index: number | null) => void;
}

export const ScatterPlot: React.FC<ScatterPlotProps> = ({
  coords,
  documents,
  thumbnailUrls,
  triplets,
  displayMode = 'thumbnails',
  width = 800,
  height = 600,
  onSelectDocument,
  highlightedIndex,
  onHoverDocument,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [selectedPointIndex, setSelectedPointIndex] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [selectedTripletIndex, setSelectedTripletIndex] = useState<number>(0);
  const [showTriplets, setShowTriplets] = useState<boolean>(true);
  const [hoveredTripletRole, setHoveredTripletRole] = useState<'anchor' | 'positive' | 'negative' | null>(null);
  const hasThumbnailsAvailable = thumbnailUrls && thumbnailUrls.some(url => url !== null);
  const useThumbnails = displayMode === 'thumbnails' && hasThumbnailsAvailable;
  const hasTriplets = triplets && triplets.length > 0;

  // Reset triplet index when selected point changes
  useEffect(() => {
    setSelectedTripletIndex(0);
  }, [selectedPointIndex]);

  useEffect(() => {
    if (!svgRef.current || coords.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Filter triplets by selected point if a point is selected
    let filteredTriplets = triplets || [];
    if (selectedPointIndex !== null && hasTriplets) {
      filteredTriplets = triplets.filter(
        t => t.anchor_id === selectedPointIndex ||
             t.positive_id === selectedPointIndex ||
             t.negative_id === selectedPointIndex
      );
    }

    // Get current triplet if available
    const currentTriplet = hasTriplets && showTriplets && filteredTriplets.length > 0
      ? filteredTriplets[selectedTripletIndex]
      : null;

    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Parse coordinates into 2D points
    const points: [number, number][] = [];
    for (let i = 0; i < coords.length; i += 2) {
      points.push([coords[i], coords[i + 1]]);
    }

    // Create scales
    const xExtent = d3.extent(points, d => d[0]) as [number, number];
    const yExtent = d3.extent(points, d => d[1]) as [number, number];

    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([height - margin.bottom, margin.top]);

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create main group
    const g = svg.append('g');

    // Render points
    if (useThumbnails && thumbnailUrls) {
      // Render as images
      const imageSize = 40;
      const images = g
        .selectAll('image')
        .data(points)
        .join('image')
        .attr('x', (d) => {
          const size = selectedPointIndex !== null && d === points[selectedPointIndex] ? imageSize * 1.3 : imageSize;
          return xScale(d[0]) - size / 2;
        })
        .attr('y', (d) => {
          const size = selectedPointIndex !== null && d === points[selectedPointIndex] ? imageSize * 1.3 : imageSize;
          return yScale(d[1]) - size / 2;
        })
        .attr('width', (d, i) => i === selectedPointIndex ? imageSize * 1.3 : imageSize)
        .attr('height', (d, i) => i === selectedPointIndex ? imageSize * 1.3 : imageSize)
        .attr('href', (d, i) => thumbnailUrls![i] || '')
        .attr('opacity', (d, i) => i === selectedPointIndex ? 1 : 0.9)
        .style('cursor', 'pointer')
        .style('filter', (d, i) => i === selectedPointIndex ? 'drop-shadow(0 0 8px rgba(74, 144, 226, 0.8))' : 'none')
        .on('mouseenter', function (event, d) {
          const index = points.indexOf(d);
          if (index !== selectedPointIndex) {
            d3.select(this)
              .attr('opacity', 1)
              .attr('width', imageSize * 1.2)
              .attr('height', imageSize * 1.2)
              .attr('x', xScale(d[0]) - (imageSize * 1.2) / 2)
              .attr('y', yScale(d[1]) - (imageSize * 1.2) / 2)
              .raise();
          }
          setHoveredIndex(index);
          onHoverDocument?.(index);

          setTooltip({
            x: event.pageX,
            y: event.pageY,
            text: documents[index] || `Point ${index}`,
          });
        })
        .on('mouseleave', function (event, d) {
          const index = points.indexOf(d);
          if (index !== selectedPointIndex) {
            d3.select(this)
              .attr('opacity', 0.9)
              .attr('width', imageSize)
              .attr('height', imageSize)
              .attr('x', xScale(d[0]) - imageSize / 2)
              .attr('y', yScale(d[1]) - imageSize / 2);
          }
          setHoveredIndex(null);
          onHoverDocument?.(null);
          setTooltip(null);
        })
        .on('click', function (event, d) {
          const index = points.indexOf(d);
          const nextIndex = index === selectedPointIndex ? null : index;
          setSelectedPointIndex(nextIndex);
          onSelectDocument?.(nextIndex);
          setSelectedTripletIndex(0); // Reset to first triplet
        });
    } else {
      // Render as circles with larger hit area
      const circles = g
        .selectAll('circle')
        .data(points)
        .join('circle')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', (d, i) => i === selectedPointIndex ? 9 : 6)
        .attr('fill', (d, i) => i === selectedPointIndex ? '#ff6b6b' : '#4a90e2')
        .attr('opacity', (d, i) => i === selectedPointIndex ? 1 : 0.7)
        .attr('stroke', (d, i) => i === selectedPointIndex ? '#ff6b6b' : '#fff')
        .attr('stroke-width', (d, i) => i === selectedPointIndex ? 2.5 : 1)
        .style('cursor', 'pointer')
        .style('filter', (d, i) => i === selectedPointIndex ? 'drop-shadow(0 0 6px rgba(255, 107, 107, 0.8))' : 'none')
        .on('mouseenter', function (event, d) {
          const index = points.indexOf(d);
          if (index !== selectedPointIndex) {
            d3.select(this)
              .attr('r', 8)
              .attr('fill', '#ff6b6b')
              .attr('opacity', 1);
          }
          setHoveredIndex(index);
          onHoverDocument?.(index);

          setTooltip({
            x: event.pageX,
            y: event.pageY,
            text: documents[index] || `Point ${index}`,
          });
        })
        .on('mouseleave', function (event, d) {
          const index = points.indexOf(d);
          if (index !== selectedPointIndex) {
            d3.select(this)
              .attr('r', 6)
              .attr('fill', '#4a90e2')
              .attr('opacity', 0.7);
          }
          setHoveredIndex(null);
          onHoverDocument?.(null);
          setTooltip(null);
        })
        .on('click', function (event, d) {
          const index = points.indexOf(d);
          const nextIndex = index === selectedPointIndex ? null : index;
          setSelectedPointIndex(nextIndex);
          onSelectDocument?.(nextIndex);
          setSelectedTripletIndex(0); // Reset to first triplet
        });
    }

    // Draw triplet highlights if available
    if (currentTriplet && showTriplets) {
      const tripletGroup = g.append('g').attr('class', 'triplet-highlight');

      const anchorIdx = currentTriplet.anchor_id;
      const posIdx = currentTriplet.positive_id;
      const negIdx = currentTriplet.negative_id;

      const anchorPoint = points[anchorIdx];
      const posPoint = points[posIdx];
      const negPoint = points[negIdx];

      if (anchorPoint && posPoint && negPoint) {
        // Draw connecting lines
        tripletGroup
          .append('line')
          .attr('x1', xScale(anchorPoint[0]))
          .attr('y1', yScale(anchorPoint[1]))
          .attr('x2', xScale(posPoint[0]))
          .attr('y2', yScale(posPoint[1]))
          .attr('stroke', '#00aa00')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.6);

        tripletGroup
          .append('line')
          .attr('x1', xScale(anchorPoint[0]))
          .attr('y1', yScale(anchorPoint[1]))
          .attr('x2', xScale(negPoint[0]))
          .attr('y2', yScale(negPoint[1]))
          .attr('stroke', '#cc0000')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.6);

        // Highlight anchor, positive, negative points
        const highlightPoints = [
          { point: anchorPoint, color: '#0066cc', label: 'Anchor', role: 'anchor' as const },
          { point: posPoint, color: '#00aa00', label: 'Positive', role: 'positive' as const },
          { point: negPoint, color: '#cc0000', label: 'Negative', role: 'negative' as const }
        ];

        highlightPoints.forEach(({ point, color, label, role }) => {
          const isHovered = hoveredTripletRole === role;

          // Add outer glow ring for hovered element
          if (isHovered) {
            tripletGroup
              .append('circle')
              .attr('cx', xScale(point[0]))
              .attr('cy', yScale(point[1]))
              .attr('r', 16)
              .attr('fill', 'none')
              .attr('stroke', color)
              .attr('stroke-width', 3)
              .attr('opacity', 0.5)
              .style('pointer-events', 'none');
          }

          tripletGroup
            .append('circle')
            .attr('cx', xScale(point[0]))
            .attr('cy', yScale(point[1]))
            .attr('r', isHovered ? 10 : 8)
            .attr('fill', color)
            .attr('stroke', 'white')
            .attr('stroke-width', isHovered ? 3 : 2)
            .attr('opacity', isHovered ? 1 : (hoveredTripletRole ? 0.4 : 1))
            .style('pointer-events', 'none');

          // Add label
          tripletGroup
            .append('text')
            .attr('x', xScale(point[0]))
            .attr('y', yScale(point[1]) - (isHovered ? 20 : 15))
            .attr('text-anchor', 'middle')
            .attr('fill', color)
            .attr('font-size', isHovered ? '14px' : '12px')
            .attr('font-weight', 'bold')
            .attr('stroke', 'white')
            .attr('stroke-width', isHovered ? 4 : 3)
            .attr('paint-order', 'stroke')
            .attr('opacity', isHovered ? 1 : (hoveredTripletRole ? 0.4 : 1))
            .text(label);
        });
      }
    }

    // Draw highlighted point ring (from external hover, e.g. compare view)
    if (highlightedIndex != null && highlightedIndex >= 0 && highlightedIndex < points.length) {
      const hp = points[highlightedIndex];
      const hlGroup = g.append('g').attr('class', 'external-highlight');
      hlGroup
        .append('circle')
        .attr('cx', xScale(hp[0]))
        .attr('cy', yScale(hp[1]))
        .attr('r', 14)
        .attr('fill', 'none')
        .attr('stroke', '#ff6b6b')
        .attr('stroke-width', 3)
        .attr('opacity', 0.8)
        .style('pointer-events', 'none');
      hlGroup
        .append('circle')
        .attr('cx', xScale(hp[0]))
        .attr('cy', yScale(hp[1]))
        .attr('r', 20)
        .attr('fill', 'none')
        .attr('stroke', '#ff6b6b')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.4)
        .style('pointer-events', 'none');
    }

    // Draw axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    svg
      .append('g')
      .attr('transform', `translate(0, ${height - 40})`)
      .call(xAxis)
      .style('font-size', '10px');

    svg
      .append('g')
      .attr('transform', `translate(40, 0)`)
      .call(yAxis)
      .style('font-size', '10px');

  }, [coords, documents, thumbnailUrls, useThumbnails, triplets, selectedTripletIndex, selectedPointIndex, showTriplets, hoveredTripletRole, width, height, highlightedIndex, onHoverDocument]);

  // Filter triplets by selected point for sidebar display
  let displayTriplets = triplets || [];
  if (selectedPointIndex !== null && hasTriplets) {
    displayTriplets = triplets.filter(
      t => t.anchor_id === selectedPointIndex ||
           t.positive_id === selectedPointIndex ||
           t.negative_id === selectedPointIndex
    );
  }

  const currentTriplet = hasTriplets && showTriplets && displayTriplets.length > 0
    ? displayTriplets[selectedTripletIndex]
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
        {coords.length === 0 ? (
          <div
            style={{
              width,
              height,
              border: '1px solid #ddd',
              borderRadius: '4px',
              background: '#fafafa',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#666',
              fontSize: '14px',
              padding: '20px',
              textAlign: 'center',
            }}
          >
            <div>
              <div style={{ marginBottom: '8px', fontWeight: 500 }}>No visualization available</div>
              <div style={{ fontSize: '12px' }}>
                This view is not supported for this method.
                <br />
                Try using the Heatmap or Graph view instead.
              </div>
            </div>
          </div>
        ) : (
          <svg
            ref={svgRef}
            width={width}
            height={height}
            style={{
              border: '1px solid #ddd',
              borderRadius: '4px',
              background: '#fafafa',
            }}
          />
        )}
        {tooltip && (
          <div
            style={{
              position: 'fixed',
              left: tooltip.x + 10,
              top: tooltip.y + 10,
              background: 'rgba(0, 0, 0, 0.85)',
              color: 'white',
              padding: '8px 12px',
              borderRadius: '4px',
              fontSize: '12px',
              maxWidth: '300px',
              pointerEvents: 'none',
              zIndex: 1000,
              whiteSpace: 'pre-wrap',
            }}
          >
            {tooltip.text.slice(0, 200)}
            {tooltip.text.length > 200 ? '...' : ''}
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
              {selectedPointIndex !== null ? 'Point Info' : 'Triplets'}
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

          {selectedPointIndex !== null && (
            <div style={{ marginBottom: '15px', padding: '10px', background: '#e3f2fd', borderRadius: '4px', fontSize: '13px' }}>
              <strong>Selected Point: {selectedPointIndex}</strong>
              <div style={{ marginTop: '6px', maxHeight: '100px', overflowY: 'auto', fontSize: '11px', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                {documents[selectedPointIndex].slice(0, 300)}
                {documents[selectedPointIndex].length > 300 ? '...' : ''}
              </div>
              {displayTriplets.length > 0 && (
                <div style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                  <strong>{displayTriplets.length}</strong> triplet{displayTriplets.length !== 1 ? 's' : ''} involving this point
                </div>
              )}
            </div>
          )}

          {showTriplets && displayTriplets.length === 0 && selectedPointIndex !== null && (
            <div style={{
              padding: '20px',
              textAlign: 'center',
              color: '#999',
              fontSize: '13px',
            }}>
              No triplets involve this point
            </div>
          )}

          {showTriplets && displayTriplets.length > 0 && (
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
                  {selectedTripletIndex + 1} / {displayTriplets.length}
                </span>
                <button
                  onClick={() => setSelectedTripletIndex(Math.min(displayTriplets.length - 1, selectedTripletIndex + 1))}
                  disabled={selectedTripletIndex === displayTriplets.length - 1}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    background: '#fff',
                    cursor: selectedTripletIndex === displayTriplets.length - 1 ? 'not-allowed' : 'pointer',
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

                  {selectedPointIndex !== null && (
                    <div style={{ marginBottom: '12px', fontSize: '12px', color: '#666', fontStyle: 'italic' }}>
                      Selected point is:{' '}
                      <strong>
                        {currentTriplet.anchor_id === selectedPointIndex && 'Anchor'}
                        {currentTriplet.positive_id === selectedPointIndex && 'Positive'}
                        {currentTriplet.negative_id === selectedPointIndex && 'Negative'}
                      </strong>
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
