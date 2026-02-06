/**
 * Force-directed graph visualization with physics simulation and controls.
 * Inspired by gpt2-explorer's graph view.
 */

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { TripletData } from '../types/manifest';

interface ForceDirectedGraphProps {
  coords: Float32Array;
  documents: string[];
  embeddings: Float32Array;
  triplets?: TripletData[];
  width?: number;
  height?: number;
}

interface Node extends d3.SimulationNodeDatum {
  id: number;
  label: string;
  color: string;
}

interface Link {
  source: number | Node;
  target: number | Node;
  strength: number;
}

export const ForceDirectedGraph: React.FC<ForceDirectedGraphProps> = ({
  coords,
  documents,
  embeddings,
  triplets,
  width = 1000,
  height = 700,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewport, setViewport] = useState({ x: 0, y: 0, k: 1 });
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null);
  const dragging = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null);

  // Controls state
  const [threshold, setThreshold] = useState(0.1);
  const [kCap, setKCap] = useState(8);
  const [showLabels, setShowLabels] = useState(true);
  const [metric, setMetric] = useState<'cosine' | 'dot'>('cosine');

  // Triplet state
  const [selectedTripletIndex, setSelectedTripletIndex] = useState<number>(0);
  const [showTriplets, setShowTriplets] = useState<boolean>(true);
  const hasTriplets = triplets && triplets.length > 0;

  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null);
  const nodesRef = useRef<Node[]>([]);
  const linksRef = useRef<Link[]>([]);

  // Parse embeddings into 2D array
  const embeddingsData = React.useMemo(() => {
    const dim = embeddings.length / documents.length;
    const result: number[][] = [];
    for (let i = 0; i < documents.length; i++) {
      const vec: number[] = [];
      for (let j = 0; j < dim; j++) {
        vec.push(embeddings[i * dim + j]);
      }
      result.push(vec);
    }
    return result;
  }, [embeddings, documents]);

  // Compute similarity and create edges
  const edges = React.useMemo(() => {
    const computeSimilarity = (a: number[], b: number[]) => {
      if (metric === 'dot') {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
          sum += a[i] * b[i];
        }
        return sum;
      } else {
        // cosine
        let dot = 0;
        let magA = 0;
        let magB = 0;
        for (let i = 0; i < a.length; i++) {
          dot += a[i] * b[i];
          magA += a[i] * a[i];
          magB += b[i] * b[i];
        }
        return dot / (Math.sqrt(magA) * Math.sqrt(magB));
      }
    };

    const n = embeddingsData.length;
    const edges: { source: number; target: number; strength: number }[] = [];

    // Compute all pairwise similarities
    for (let i = 0; i < n; i++) {
      const neighbors: { idx: number; sim: number }[] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const sim = computeSimilarity(embeddingsData[i], embeddingsData[j]);
        if (sim >= threshold) {
          neighbors.push({ idx: j, sim });
        }
      }

      // Keep top-k neighbors
      neighbors.sort((a, b) => b.sim - a.sim);
      const topK = neighbors.slice(0, kCap);

      for (const { idx, sim } of topK) {
        // Add edge (only if not already added from the other direction)
        if (i < idx) {
          edges.push({ source: i, target: idx, strength: sim });
        }
      }
    }

    return edges;
  }, [embeddingsData, threshold, kCap, metric]);

  // Initialize nodes and simulation
  useEffect(() => {
    const nodes: Node[] = documents.map((label, i) => ({
      id: i,
      label,
      x: (Math.random() - 0.5) * 200,
      y: (Math.random() - 0.5) * 200,
      color: `hsl(${(i / documents.length) * 360}, 70%, 60%)`,
    }));

    const links: Link[] = edges.map((e) => ({
      source: e.source,
      target: e.target,
      strength: e.strength,
    }));

    nodesRef.current = nodes;
    linksRef.current = links;

    // Create D3 force simulation
    const simulation = d3
      .forceSimulation(nodes)
      .force('charge', d3.forceManyBody().strength(-40))
      .force('center', d3.forceCenter(0, 0))
      .force(
        'link',
        d3
          .forceLink(links)
          .id((d: any) => d.id)
          .distance(40)
          .strength((d: any) => d.strength * 0.5)
      )
      .alpha(0.7);

    simulationRef.current = simulation;

    return () => {
      simulation.stop();
    };
  }, [documents, edges]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d')!;
    let raf = 0;

    function resize() {
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function render() {
      if (!canvas) return;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      ctx.save();
      ctx.translate(w / 2 + viewport.x, h / 2 + viewport.y);
      ctx.scale(viewport.k, viewport.k);

      // Draw links
      ctx.lineWidth = 1.5;
      for (const link of linksRef.current) {
        const source = typeof link.source === 'number' ? nodesRef.current[link.source] : link.source;
        const target = typeof link.target === 'number' ? nodesRef.current[link.target] : link.target;
        if (!source || !target) continue;

        const alpha = Math.max(0.2, Math.min(0.9, link.strength));
        ctx.strokeStyle = `rgba(60,60,60,${alpha})`;
        ctx.beginPath();
        ctx.moveTo(source.x!, source.y!);
        ctx.lineTo(target.x!, target.y!);
        ctx.stroke();
      }

      // Draw nodes
      const currentTriplet = hasTriplets && showTriplets && triplets!.length > 0
        ? triplets![selectedTripletIndex]
        : null;

      for (const node of nodesRef.current) {
        ctx.beginPath();

        // Determine color based on triplet role
        let fillColor = node.color; // default colorful
        if (currentTriplet) {
          if (node.id === currentTriplet.anchor_id) {
            fillColor = '#0066cc'; // blue
          } else if (node.id === currentTriplet.positive_id) {
            fillColor = '#00aa00'; // green
          } else if (node.id === currentTriplet.negative_id) {
            fillColor = '#cc0000'; // red
          } else {
            fillColor = '#888'; // gray out non-triplet nodes
          }
        }

        ctx.fillStyle = fillColor;
        ctx.arc(node.x!, node.y!, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw labels
      if (showLabels) {
        ctx.font = '11px system-ui, sans-serif';
        ctx.fillStyle = 'rgba(20,20,20,0.9)';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        for (const node of nodesRef.current) {
          const label = node.label.length > 48 ? node.label.slice(0, 48) + '…' : node.label;
          ctx.fillText(label, node.x! + 7, node.y!);
        }
      }

      ctx.restore();
      raf = requestAnimationFrame(render);
    }
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [viewport, showLabels, selectedTripletIndex, showTriplets, hasTriplets, triplets]);

  // Pan/zoom interactions
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { offsetX, offsetY, deltaY } = e;
      const scale = Math.exp(-deltaY * 0.001);
      const rect = canvas.getBoundingClientRect();
      const cx = (offsetX - rect.width / 2 - viewport.x) / viewport.k;
      const cy = (offsetY - rect.height / 2 - viewport.y) / viewport.k;
      const k = Math.max(0.2, Math.min(6, viewport.k * scale));
      const x = viewport.x + (1 - scale) * (cx * viewport.k);
      const y = viewport.y + (1 - scale) * (cy * viewport.k);
      setViewport({ x, y, k });
    };

    const onPointerDown = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      dragging.current = {
        x: viewport.x,
        y: viewport.y,
        startX: e.clientX - rect.left,
        startY: e.clientY - rect.top,
      };
      (e.target as Element).setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (dragging.current) {
        const dx = e.clientX - rect.left - dragging.current.startX;
        const dy = e.clientY - rect.top - dragging.current.startY;
        setViewport({ ...viewport, x: dragging.current.x + dx, y: dragging.current.y + dy });
      }

      // Tooltip
      const offsetX = e.clientX - rect.left;
      const offsetY = e.clientY - rect.top;
      const worldX = (offsetX - rect.width / 2 - viewport.x) / viewport.k;
      const worldY = (offsetY - rect.height / 2 - viewport.y) / viewport.k;
      const searchRadius = 20 / viewport.k;

      let closest: { id: number; dist: number } | null = null;
      for (const node of nodesRef.current) {
        const dx = node.x! - worldX;
        const dy = node.y! - worldY;
        const dist = Math.hypot(dx, dy);
        if (dist < searchRadius && (!closest || dist < closest.dist)) {
          closest = { id: node.id, dist };
        }
      }

      if (!closest) {
        setTooltip(null);
      } else {
        const node = nodesRef.current[closest.id];
        const text = node.label;
        const x = Math.min(Math.max(offsetX + 12, 8), rect.width - 220);
        const y = Math.min(Math.max(offsetY + 12, 8), rect.height - 40);
        setTooltip({ text, x, y });
      }
    };

    const onPointerUp = () => {
      dragging.current = null;
    };

    const onPointerLeave = () => {
      dragging.current = null;
      setTooltip(null);
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', onPointerLeave);

    return () => {
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointerleave', onPointerLeave);
    };
  }, [viewport]);

  return (
    <div style={{ display: 'flex', gap: '20px' }}>
      {/* Controls sidebar */}
      <div style={{
        width: '280px',
        padding: '20px',
        background: '#f5f5f5',
        borderRadius: '8px',
        fontSize: '14px',
      }}>
        <h3 style={{ marginTop: 0 }}>Graph Controls</h3>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontWeight: 500, marginBottom: '8px' }}>
            Similarity Metric
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setMetric('cosine')}
              style={{
                flex: 1,
                padding: '6px',
                background: metric === 'cosine' ? '#4a90e2' : '#fff',
                color: metric === 'cosine' ? '#fff' : '#000',
                border: '1px solid #ccc',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Cosine
            </button>
            <button
              onClick={() => setMetric('dot')}
              style={{
                flex: 1,
                padding: '6px',
                background: metric === 'dot' ? '#4a90e2' : '#fff',
                color: metric === 'dot' ? '#fff' : '#000',
                border: '1px solid #ccc',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Dot
            </button>
          </div>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontWeight: 500, marginBottom: '8px' }}>
            Threshold τ (edge if sim ≥ τ)
          </label>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ textAlign: 'center', fontSize: '12px', color: '#666' }}>
            τ = {threshold.toFixed(2)}
          </div>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontWeight: 500, marginBottom: '8px' }}>
            Max neighbors per node (k)
          </label>
          <input
            type="range"
            min="2"
            max="24"
            step="1"
            value={kCap}
            onChange={(e) => setKCap(parseInt(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ textAlign: 'center', fontSize: '12px', color: '#666' }}>
            k = {kCap}
          </div>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
            <span>Show labels</span>
          </label>
        </div>

        <div style={{ fontSize: '12px', color: '#666', paddingTop: '12px', borderTop: '1px solid #ddd' }}>
          <div style={{ marginBottom: '4px' }}>
            <strong>{documents.length}</strong> nodes
          </div>
          <div style={{ marginBottom: '4px' }}>
            <strong>{linksRef.current.length}</strong> edges
          </div>
          <div style={{ marginTop: '12px', lineHeight: 1.5 }}>
            Scroll to zoom<br/>
            Drag to pan
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, position: 'relative' }}>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: height,
            border: '1px solid #ccc',
            background: '#fafafa',
            cursor: dragging.current ? 'grabbing' : 'grab',
            borderRadius: '4px',
          }}
        />
        {tooltip && (
          <div
            style={{
              position: 'absolute',
              top: tooltip.y,
              left: tooltip.x,
              maxWidth: 320,
              maxHeight: 220,
              padding: '8px 10px',
              background: 'rgba(20,20,20,0.92)',
              color: '#fff',
              fontSize: 12,
              borderRadius: 6,
              pointerEvents: 'none',
              whiteSpace: 'pre-wrap',
              overflowY: 'auto',
              boxShadow: '0 2px 10px rgba(0,0,0,0.4)',
            }}
          >
            {tooltip.text}
          </div>
        )}
      </div>

      {/* Triplets sidebar */}
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

          {showTriplets && triplets!.length > 0 && (
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
                  {selectedTripletIndex + 1} / {triplets!.length}
                </span>
                <button
                  onClick={() => setSelectedTripletIndex(Math.min(triplets!.length - 1, selectedTripletIndex + 1))}
                  disabled={selectedTripletIndex === triplets!.length - 1}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    background: '#fff',
                    cursor: selectedTripletIndex === triplets!.length - 1 ? 'not-allowed' : 'pointer',
                    fontSize: '13px',
                  }}
                >
                  Next →
                </button>
              </div>

              {(() => {
                const currentTriplet = triplets![selectedTripletIndex];
                const getDocText = (doc: any): string => {
                  if (typeof doc === 'string') return doc;
                  if (doc && typeof doc === 'object' && doc.text) return doc.text;
                  return JSON.stringify(doc);
                };

                return (
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
                        background: '#e3f2fd',
                        borderRadius: '4px',
                        borderLeft: '3px solid #0066cc',
                      }}
                    >
                      <strong style={{ color: '#0066cc' }}>Anchor</strong>
                      <div style={{ marginTop: '5px', maxHeight: '100px', overflowY: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px' }}>
                        {getDocText(currentTriplet.anchor).slice(0, 300)}
                        {getDocText(currentTriplet.anchor).length > 300 ? '...' : ''}
                      </div>
                    </div>

                    <div
                      style={{
                        marginBottom: '15px',
                        padding: '10px',
                        background: '#e8f5e9',
                        borderRadius: '4px',
                        borderLeft: '3px solid #00aa00',
                      }}
                    >
                      <strong style={{ color: '#00aa00' }}>Positive</strong>
                      <div style={{ marginTop: '5px', maxHeight: '100px', overflowY: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px' }}>
                        {getDocText(currentTriplet.positive).slice(0, 300)}
                        {getDocText(currentTriplet.positive).length > 300 ? '...' : ''}
                      </div>
                    </div>

                    <div
                      style={{
                        marginBottom: '15px',
                        padding: '10px',
                        background: '#ffebee',
                        borderRadius: '4px',
                        borderLeft: '3px solid #cc0000',
                      }}
                    >
                      <strong style={{ color: '#cc0000' }}>Negative</strong>
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
                );
              })()}
            </>
          )}
        </div>
      )}
    </div>
  );
};
