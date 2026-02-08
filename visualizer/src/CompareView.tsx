/**
 * CompareView: side-by-side comparison of two criteria for the same dataset.
 * Renders two ScatterPlots with shared hover highlighting, plus a Pareto
 * accuracy scatter when results.json is available.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { DatasetIndex, VisualizationMode, VisualizationData } from './types/manifest';
import { loadDatasetIndex, loadVisualizationData, loadBenchmarkResults } from './utils/dataLoader';
import { ScatterPlot } from './render/ScatterPlot';
import './App.css';

/* ---------- helpers (same logic as App.tsx) ---------- */

function getBenchmarkRoots(index: DatasetIndex): string[] {
  const roots = new Set<string>();
  for (const key of Object.keys(index)) {
    const [root] = key.split('/');
    if (root) roots.add(root);
  }
  return Array.from(roots).sort();
}

function getViewKinds(index: DatasetIndex, benchmarkRoot: string): string[] {
  const kinds = new Set<string>();
  if (index[benchmarkRoot]) kinds.add('triples');
  for (const key of Object.keys(index)) {
    if (!key.startsWith(`${benchmarkRoot}/`)) continue;
    const parts = key.split('/');
    if (parts.length >= 2 && parts[1]) kinds.add(parts[1]);
  }
  return Array.from(kinds).sort((a, b) => {
    const order = ['triples', 'corpus'];
    const ai = order.indexOf(a);
    const bi = order.indexOf(b);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return a.localeCompare(b);
  });
}

function resolveBenchmarkKey(
  index: DatasetIndex,
  benchmarkRoot: string,
  viewKind: string,
): string {
  const namespaced = `${benchmarkRoot}/${viewKind}`;
  if (index[namespaced]) return namespaced;
  if (viewKind === 'triples' && index[benchmarkRoot]) return benchmarkRoot;
  const fallback = Object.keys(index).find(
    (key) => key === benchmarkRoot || key.startsWith(`${benchmarkRoot}/`),
  );
  return fallback || '';
}

function preferredMode(modes: string[]): VisualizationMode {
  if (modes.includes('tsne')) return 'tsne';
  return (modes[0] || 'tsne') as VisualizationMode;
}

/* ---------- neighbor ranking (copied from App.tsx) ---------- */

type NeighborItem = { index: number; score: number; text: string };

function cosineSimilarity(
  embeddings: Float32Array, dim: number, idxA: number, idxB: number,
): number {
  const offsetA = idxA * dim;
  const offsetB = idxB * dim;
  let dot = 0, nA = 0, nB = 0;
  for (let i = 0; i < dim; i++) {
    const a = embeddings[offsetA + i], b = embeddings[offsetB + i];
    dot += a * b; nA += a * a; nB += b * b;
  }
  if (nA === 0 || nB === 0) return 0;
  return dot / (Math.sqrt(nA) * Math.sqrt(nB));
}

function l1Distance(
  embeddings: Float32Array, dim: number, idxA: number, idxB: number,
): number {
  const offsetA = idxA * dim;
  const offsetB = idxB * dim;
  let sum = 0;
  for (let i = 0; i < dim; i++) sum += Math.abs(embeddings[offsetA + i] - embeddings[offsetB + i]);
  return sum;
}

function rankNeighborsForDoc(
  embeddings: Float32Array, documents: string[], queryIndex: number,
  usePseudologit: boolean, embeddingDimHint?: number,
): { neighbors: NeighborItem[]; metricLabel: string } {
  const nDocs = documents.length;
  if (queryIndex < 0 || queryIndex >= nDocs)
    return { neighbors: [], metricLabel: usePseudologit ? 'L1 distance' : 'Cosine similarity' };

  const isPrecomputedMatrix =
    embeddings.length === nDocs * nDocs &&
    (embeddingDimHint === undefined || embeddingDimHint === nDocs) &&
    (() => {
      if (nDocs <= 1) return true;
      let checked = 0, symmetricPairs = 0;
      const step = Math.max(1, Math.floor(nDocs / 11));
      for (let i = 0; i < nDocs && checked < 24; i += step) {
        const j = (i + step) % nDocs;
        if (i === j) continue;
        const a = embeddings[i * nDocs + j], b = embeddings[j * nDocs + i];
        if (Number.isFinite(a) && Number.isFinite(b)) { checked++; if (Math.abs(a - b) <= 1e-4) symmetricPairs++; }
      }
      return checked > 0 && symmetricPairs / checked >= 0.9;
    })();

  const scored: Array<{ index: number; score: number }> = [];

  if (isPrecomputedMatrix) {
    for (let j = 0; j < nDocs; j++) {
      if (j === queryIndex) continue;
      scored.push({ index: j, score: embeddings[queryIndex * nDocs + j] });
    }
    scored.sort((a, b) => b.score - a.score);
    return { neighbors: scored.map(({ index, score }) => ({ index, score, text: documents[index] })), metricLabel: 'Similarity score' };
  }

  const dim = embeddings.length / nDocs;
  for (let j = 0; j < nDocs; j++) {
    if (j === queryIndex) continue;
    scored.push({ index: j, score: usePseudologit ? l1Distance(embeddings, dim, queryIndex, j) : cosineSimilarity(embeddings, dim, queryIndex, j) });
  }
  scored.sort(usePseudologit ? (a, b) => a.score - b.score : (a, b) => b.score - a.score);
  return { neighbors: scored.map(({ index, score }) => ({ index, score, text: documents[index] })), metricLabel: usePseudologit ? 'L1 distance (lower is closer)' : 'Cosine similarity' };
}

/* ---------- Pareto scatter (inline SVG) ---------- */

interface ParetoPoint {
  method: string;
  accA: number;
  accB: number;
}

const ParetoScatter: React.FC<{
  points: ParetoPoint[];
  labelA: string;
  labelB: string;
}> = ({ points, labelA, labelB }) => {
  const w = 420;
  const h = 420;
  const pad = { top: 20, right: 20, bottom: 50, left: 60 };
  const iw = w - pad.left - pad.right;
  const ih = h - pad.top - pad.bottom;

  if (points.length === 0) return null;

  const xMin = Math.min(...points.map((p) => p.accA));
  const xMax = Math.max(...points.map((p) => p.accA));
  const yMin = Math.min(...points.map((p) => p.accB));
  const yMax = Math.max(...points.map((p) => p.accB));

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const xPad = xRange * 0.1;
  const yPad = yRange * 0.1;

  const sx = (v: number) => pad.left + ((v - (xMin - xPad)) / (xRange + 2 * xPad)) * iw;
  const sy = (v: number) => pad.top + ih - ((v - (yMin - yPad)) / (yRange + 2 * yPad)) * ih;

  const colors = [
    '#4a90e2', '#e24a4a', '#4ae24a', '#e2a64a', '#a64ae2',
    '#4ae2e2', '#e24aa6', '#8b8b00', '#008b8b', '#8b008b',
  ];

  return (
    <div className="visualization" style={{ marginTop: 20 }}>
      <h3 style={{ margin: '0 0 8px', fontSize: 16, fontWeight: 600, color: '#333' }}>
        Pareto: accuracy under criterion A vs B
      </h3>
      <svg width={w} height={h} style={{ border: '1px solid #ddd', borderRadius: 4, background: '#fafafa' }}>
        <line x1={pad.left} y1={pad.top + ih} x2={pad.left + iw} y2={pad.top + ih} stroke="#999" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + ih} stroke="#999" />
        <text x={pad.left + iw / 2} y={h - 8} textAnchor="middle" fontSize={12} fill="#333">
          {labelA} accuracy
        </text>
        <text x={14} y={pad.top + ih / 2} textAnchor="middle" fontSize={12} fill="#333"
          transform={`rotate(-90, 14, ${pad.top + ih / 2})`}>
          {labelB} accuracy
        </text>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = (xMin - xPad) + t * (xRange + 2 * xPad);
          return (
            <g key={`xt-${t}`}>
              <line x1={sx(v)} y1={pad.top + ih} x2={sx(v)} y2={pad.top + ih + 4} stroke="#999" />
              <text x={sx(v)} y={pad.top + ih + 16} textAnchor="middle" fontSize={10} fill="#666">{v.toFixed(2)}</text>
            </g>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = (yMin - yPad) + t * (yRange + 2 * yPad);
          return (
            <g key={`yt-${t}`}>
              <line x1={pad.left - 4} y1={sy(v)} x2={pad.left} y2={sy(v)} stroke="#999" />
              <text x={pad.left - 8} y={sy(v) + 3} textAnchor="end" fontSize={10} fill="#666">{v.toFixed(2)}</text>
            </g>
          );
        })}
        {points.map((p, i) => (
          <g key={p.method}>
            <circle cx={sx(p.accA)} cy={sy(p.accB)} r={6} fill={colors[i % colors.length]} opacity={0.85} />
            <text x={sx(p.accA) + 8} y={sy(p.accB) + 4} fontSize={10} fill="#333">
              {p.method.length > 25 ? p.method.slice(0, 22) + '...' : p.method}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

/* ---------- main component ---------- */

const CompareView: React.FC = () => {
  const [index, setIndex] = useState<DatasetIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedBenchmark, setSelectedBenchmark] = useState('');
  const [selectedViewKind, setSelectedViewKind] = useState('triples');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [criterionA, setCriterionA] = useState('');
  const [criterionB, setCriterionB] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('');
  const [selectedMode, setSelectedMode] = useState<VisualizationMode>('tsne');
  const [displayMode, setDisplayMode] = useState<'points' | 'thumbnails'>('thumbnails');

  const [vizA, setVizA] = useState<VisualizationData | null>(null);
  const [vizB, setVizB] = useState<VisualizationData | null>(null);
  const [loadingViz, setLoadingViz] = useState(false);

  const [results, setResults] = useState<any>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState<number | null>(null);
  const [topK, setTopK] = useState<number>(5);

  /* derived */
  const benchmarkRoots = useMemo(() => (index ? getBenchmarkRoots(index) : []), [index]);

  const availableViewKinds = useMemo(
    () => (index && selectedBenchmark ? getViewKinds(index, selectedBenchmark) : []),
    [index, selectedBenchmark],
  );

  const activeBenchmarkKey = useMemo(
    () =>
      index && selectedBenchmark
        ? resolveBenchmarkKey(index, selectedBenchmark, selectedViewKind)
        : '',
    [index, selectedBenchmark, selectedViewKind],
  );

  const benchmarkData = activeBenchmarkKey && index ? index[activeBenchmarkKey] : undefined;

  const datasets = useMemo(
    () => (benchmarkData ? Object.keys(benchmarkData) : []),
    [benchmarkData],
  );

  const criteria = useMemo(
    () =>
      selectedDataset && benchmarkData?.[selectedDataset]
        ? Object.keys(benchmarkData[selectedDataset])
        : [],
    [benchmarkData, selectedDataset],
  );

  const methods = useMemo(() => {
    if (!benchmarkData || !selectedDataset || !criterionA || !criterionB) return [];
    const mA = benchmarkData[selectedDataset]?.[criterionA]
      ? Object.keys(benchmarkData[selectedDataset][criterionA])
      : [];
    const mB = new Set(
      benchmarkData[selectedDataset]?.[criterionB]
        ? Object.keys(benchmarkData[selectedDataset][criterionB])
        : [],
    );
    return mA.filter((m) => mB.has(m));
  }, [benchmarkData, selectedDataset, criterionA, criterionB]);

  const modes = useMemo(() => {
    if (!benchmarkData || !selectedDataset || !criterionA || !selectedMethod) return [];
    return benchmarkData[selectedDataset]?.[criterionA]?.[selectedMethod]?.modes || [];
  }, [benchmarkData, selectedDataset, criterionA, selectedMethod]);

  const hasThumbnailsA = vizA?.thumbnailUrls && vizA.thumbnailUrls.some(u => u !== null);
  const hasThumbnailsB = vizB?.thumbnailUrls && vizB.thumbnailUrls.some(u => u !== null);
  const hasThumbnails = hasThumbnailsA || hasThumbnailsB;

  /* compute neighbors for both criteria */
  const neighborsA = useMemo(() => {
    if (selectedDocumentIndex === null || !vizA) return null;
    const usePl = selectedMethod.toLowerCase().includes('pseudologit');
    const ranked = rankNeighborsForDoc(vizA.embeddings, vizA.documents, selectedDocumentIndex, usePl, vizA.manifest.embedding_dim);
    return { neighbors: ranked.neighbors.slice(0, topK), metricLabel: ranked.metricLabel };
  }, [selectedDocumentIndex, vizA, selectedMethod, topK]);

  const neighborsB = useMemo(() => {
    if (selectedDocumentIndex === null || !vizB) return null;
    const usePl = selectedMethod.toLowerCase().includes('pseudologit');
    const ranked = rankNeighborsForDoc(vizB.embeddings, vizB.documents, selectedDocumentIndex, usePl, vizB.manifest.embedding_dim);
    return { neighbors: ranked.neighbors.slice(0, topK), metricLabel: ranked.metricLabel };
  }, [selectedDocumentIndex, vizB, selectedMethod, topK]);

  /* reset selection on parameter change */
  useEffect(() => {
    setSelectedDocumentIndex(null);
  }, [selectedBenchmark, selectedViewKind, selectedDataset, criterionA, criterionB, selectedMethod]);

  /* load index */
  useEffect(() => {
    loadDatasetIndex()
      .then((idx) => {
        setIndex(idx);
        const roots = getBenchmarkRoots(idx);
        if (roots.length > 0) {
          const bm = roots[0];
          const views = getViewKinds(idx, bm);
          const view = views.includes('triples') ? 'triples' : views[0] || 'triples';
          const key = resolveBenchmarkKey(idx, bm, view);
          const bd = key ? idx[key] : undefined;
          setSelectedBenchmark(bm);
          setSelectedViewKind(view);
          if (bd) {
            const ds = Object.keys(bd);
            if (ds.length > 0) {
              setSelectedDataset(ds[0]);
              const crits = Object.keys(bd[ds[0]]);
              if (crits.length > 0) setCriterionA(crits[0]);
              if (crits.length > 1) setCriterionB(crits[1]);
              else if (crits.length > 0) setCriterionB(crits[0]);
            }
          }
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(`Failed to load dataset index: ${err.message}`);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (methods.length > 0 && !methods.includes(selectedMethod)) {
      setSelectedMethod(methods[0]);
    }
  }, [methods, selectedMethod]);

  useEffect(() => {
    if (modes.length > 0 && !modes.includes(selectedMode)) {
      setSelectedMode(preferredMode(modes));
    }
  }, [modes, selectedMode]);

  useEffect(() => {
    if (activeBenchmarkKey) {
      loadBenchmarkResults(activeBenchmarkKey).then(setResults);
    }
  }, [activeBenchmarkKey]);

  useEffect(() => {
    if (!index || !activeBenchmarkKey || !selectedDataset || !criterionA || !criterionB || !selectedMethod || !selectedMode) {
      setVizA(null);
      setVizB(null);
      return;
    }

    const bd = index[activeBenchmarkKey];
    const mdA = bd?.[selectedDataset]?.[criterionA]?.[selectedMethod];
    const mdB = bd?.[selectedDataset]?.[criterionB]?.[selectedMethod];

    if (!mdA || !mdB || mdA.has_embeddings === false || mdB.has_embeddings === false) {
      setVizA(null);
      setVizB(null);
      return;
    }

    setLoadingViz(true);
    setError(null);

    Promise.all([
      loadVisualizationData(mdA.path, selectedMode),
      loadVisualizationData(mdB.path, selectedMode),
    ])
      .then(([a, b]) => { setVizA(a); setVizB(b); setLoadingViz(false); })
      .catch((err) => { setError(`Failed to load visualization data: ${err.message}`); setLoadingViz(false); });
  }, [index, activeBenchmarkKey, selectedDataset, criterionA, criterionB, selectedMethod, selectedMode]);

  /* Pareto */
  const paretoPoints = useMemo((): ParetoPoint[] => {
    if (!results || !criterionA || !criterionB || !selectedDataset || !benchmarkData) return [];
    const cmA = benchmarkData[selectedDataset]?.[criterionA] || {};
    const cmB = benchmarkData[selectedDataset]?.[criterionB] || {};
    const getTask = (md: { path: string }) => { const p = md.path.split('/'); return p.length >= 2 ? p[p.length - 2] : ''; };
    const taskA = Object.values(cmA)[0] ? getTask(Object.values(cmA)[0]) : '';
    const taskB = Object.values(cmB)[0] ? getTask(Object.values(cmB)[0]) : '';
    if (!taskA || !taskB || !results[taskA] || !results[taskB]) return [];
    const pts: ParetoPoint[] = [];
    for (const method of Object.keys(results[taskA])) {
      const rA = results[taskA][method], rB = results[taskB]?.[method];
      if (rA && rB && typeof rA.accuracy === 'number' && typeof rB.accuracy === 'number')
        pts.push({ method, accA: rA.accuracy, accB: rB.accuracy });
    }
    return pts;
  }, [results, criterionA, criterionB, selectedDataset, benchmarkData]);

  /* handle click from either scatter plot */
  const handleSelectDocument = (idx: number | null) => {
    setSelectedDocumentIndex(idx);
  };

  /* ---------- render ---------- */

  if (loading) {
    return <div className="app"><div className="loading">Loading dataset index...</div></div>;
  }

  if (!index || Object.keys(index).length === 0) {
    return (
      <div className="app">
        <div className="error"><h2>No visualizations found</h2><p>Please run evaluation first.</p></div>
      </div>
    );
  }

  const selectedDocText = vizA && selectedDocumentIndex !== null
    ? (vizA.documents[selectedDocumentIndex] || `Point ${selectedDocumentIndex}`)
    : '';

  const selectedDocThumb = vizA?.thumbnailUrls?.[selectedDocumentIndex ?? -1] ?? null;

  return (
    <div className="app">
      <header className="header">
        <div className="header-top-row">
          <h1>Compare Criteria</h1>
          <a href="/" className="nav-compare-btn">Back to Visualizer</a>
        </div>
        <p className="header-subtitle">
          Side-by-side scatter plots for <strong>{criterionA || '?'}</strong> vs <strong>{criterionB || '?'}</strong> on <strong>{selectedDataset || '...'}</strong>
        </p>
        <div className="controls">
          <div className="control-group">
            <label>Experiment:</label>
            <select value={selectedBenchmark} onChange={(e) => {
              const bm = e.target.value;
              setSelectedBenchmark(bm);
              const views = getViewKinds(index, bm);
              const view = views.includes(selectedViewKind) ? selectedViewKind : views.includes('triples') ? 'triples' : views[0] || 'triples';
              setSelectedViewKind(view);
              const key = resolveBenchmarkKey(index, bm, view);
              const bd = key ? index[key] : undefined;
              if (bd) {
                const ds = Object.keys(bd);
                if (ds.length > 0) {
                  setSelectedDataset(ds[0]);
                  const crits = Object.keys(bd[ds[0]]);
                  if (crits.length > 0) setCriterionA(crits[0]);
                  if (crits.length > 1) setCriterionB(crits[1]);
                  else if (crits.length > 0) setCriterionB(crits[0]);
                }
              }
            }}>
              {benchmarkRoots.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>View:</label>
            <select value={selectedViewKind} onChange={(e) => {
              const view = e.target.value;
              setSelectedViewKind(view);
              const key = resolveBenchmarkKey(index, selectedBenchmark, view);
              const bd = key ? index[key] : undefined;
              if (bd) {
                const ds = Object.keys(bd);
                if (ds.length > 0) {
                  setSelectedDataset(ds[0]);
                  const crits = Object.keys(bd[ds[0]]);
                  if (crits.length > 0) setCriterionA(crits[0]);
                  if (crits.length > 1) setCriterionB(crits[1]);
                  else if (crits.length > 0) setCriterionB(crits[0]);
                }
              }
            }}>
              {availableViewKinds.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Dataset:</label>
            <select value={selectedDataset} onChange={(e) => {
              setSelectedDataset(e.target.value);
              if (benchmarkData) {
                const crits = Object.keys(benchmarkData[e.target.value] || {});
                if (crits.length > 0) setCriterionA(crits[0]);
                if (crits.length > 1) setCriterionB(crits[1]);
                else if (crits.length > 0) setCriterionB(crits[0]);
              }
            }}>
              {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Criterion A:</label>
            <select value={criterionA} onChange={(e) => setCriterionA(e.target.value)}>
              {criteria.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Criterion B:</label>
            <select value={criterionB} onChange={(e) => setCriterionB(e.target.value)}>
              {criteria.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Method:</label>
            <select value={selectedMethod} onChange={(e) => setSelectedMethod(e.target.value)}>
              {methods.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Mode:</label>
            <div className="mode-buttons">
              {modes.map((m) => (
                <button key={m} className={selectedMode === m ? 'active' : ''} onClick={() => setSelectedMode(m as VisualizationMode)}>
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="control-group">
            <label>Display:</label>
            <div className="mode-buttons">
              <button className={displayMode === 'points' ? 'active' : ''} onClick={() => setDisplayMode('points')}>
                Points
              </button>
              <button
                className={displayMode === 'thumbnails' ? 'active' : ''}
                onClick={() => setDisplayMode('thumbnails')}
                disabled={!hasThumbnails}
                title={!hasThumbnails ? 'Thumbnails not available for this dataset' : ''}
              >
                Thumbnails
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        {error && <div className="error">{error}</div>}
        {loadingViz && <div className="loading">Loading visualizations...</div>}

        {vizA && vizB && !loadingViz && (
          <>
            <div className="visualization">
              <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                <div>
                  <h3 style={{ margin: '0 0 4px', fontSize: 16, fontWeight: 600, color: '#333' }}>
                    Criterion A: {criterionA}
                  </h3>
                  <div className="viz-info" style={{ marginBottom: 8 }}>
                    <span>{vizA.documents.length} documents</span>
                    <span>{vizA.manifest.embedding_dim}D embeddings</span>
                  </div>
                  <ScatterPlot
                    coords={vizA.coords}
                    documents={vizA.documents}
                    thumbnailUrls={vizA.thumbnailUrls}
                    displayMode={displayMode}
                    width={560}
                    height={480}
                    highlightedIndex={hoveredIndex}
                    onHoverDocument={setHoveredIndex}
                    onSelectDocument={handleSelectDocument}
                  />
                </div>

                <div>
                  <h3 style={{ margin: '0 0 4px', fontSize: 16, fontWeight: 600, color: '#333' }}>
                    Criterion B: {criterionB}
                  </h3>
                  <div className="viz-info" style={{ marginBottom: 8 }}>
                    <span>{vizB.documents.length} documents</span>
                    <span>{vizB.manifest.embedding_dim}D embeddings</span>
                  </div>
                  <ScatterPlot
                    coords={vizB.coords}
                    documents={vizB.documents}
                    thumbnailUrls={vizB.thumbnailUrls}
                    displayMode={displayMode}
                    width={560}
                    height={480}
                    highlightedIndex={hoveredIndex}
                    onHoverDocument={setHoveredIndex}
                    onSelectDocument={handleSelectDocument}
                  />
                </div>
              </div>
            </div>

            {/* Neighbors panel */}
            {selectedDocumentIndex !== null && neighborsA && neighborsB && (
              <div className="neighbors-panel" style={{ marginTop: 16, width: '100%', maxWidth: 1200 }}>
                <div className="neighbors-header">
                  <h3>Top-{topK} nearest neighbors for document #{selectedDocumentIndex}</h3>
                  <div className="neighbors-controls">
                    <label>Top K:</label>
                    <select value={topK} onChange={(e) => setTopK(parseInt(e.target.value, 10))}>
                      {[3, 5, 10, 20].map((k) => <option key={k} value={k}>{k}</option>)}
                    </select>
                  </div>
                </div>

                <div className="neighbors-selected-doc">
                  <strong>Selected document #{selectedDocumentIndex}</strong>
                  <div className="neighbors-doc-content">
                    {selectedDocThumb && (
                      <img src={selectedDocThumb} alt={`Doc ${selectedDocumentIndex}`} className="neighbor-thumb" />
                    )}
                    <div className="neighbors-doc-text">
                      {selectedDocText.slice(0, 400)}
                      {selectedDocText.length > 400 ? '...' : ''}
                    </div>
                  </div>
                </div>

                <div className="neighbors-columns" style={{ gridTemplateColumns: '1fr 1fr' }}>
                  {/* Criterion A neighbors */}
                  <div>
                    <div className="neighbors-subtitle">
                      {criterionA} ({neighborsA.metricLabel})
                    </div>
                    <ol className="neighbors-list">
                      {neighborsA.neighbors.map((neighbor) => (
                        <li key={`a-${neighbor.index}`}>
                          <div className="neighbor-topline">
                            <span>Doc #{neighbor.index}</span>
                            <span>{neighbor.score.toFixed(4)}</span>
                          </div>
                          <div className="neighbor-content">
                            {vizA.thumbnailUrls?.[neighbor.index] && (
                              <img src={vizA.thumbnailUrls[neighbor.index] || ''} alt={`Doc ${neighbor.index}`} className="neighbor-thumb" />
                            )}
                            <div className="neighbor-text">
                              {neighbor.text.slice(0, 180)}
                              {neighbor.text.length > 180 ? '...' : ''}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>

                  {/* Criterion B neighbors */}
                  <div>
                    <div className="neighbors-subtitle">
                      {criterionB} ({neighborsB.metricLabel})
                    </div>
                    <ol className="neighbors-list">
                      {neighborsB.neighbors.map((neighbor) => (
                        <li key={`b-${neighbor.index}`}>
                          <div className="neighbor-topline">
                            <span>Doc #{neighbor.index}</span>
                            <span>{neighbor.score.toFixed(4)}</span>
                          </div>
                          <div className="neighbor-content">
                            {vizB.thumbnailUrls?.[neighbor.index] && (
                              <img src={vizB.thumbnailUrls[neighbor.index] || ''} alt={`Doc ${neighbor.index}`} className="neighbor-thumb" />
                            )}
                            <div className="neighbor-text">
                              {neighbor.text.slice(0, 180)}
                              {neighbor.text.length > 180 ? '...' : ''}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              </div>
            )}

            {paretoPoints.length > 0 && (
              <ParetoScatter points={paretoPoints} labelA={criterionA} labelB={criterionB} />
            )}
          </>
        )}

        {!vizA && !vizB && !loadingViz && !error && selectedMethod && (
          <div className="error">
            <strong>{selectedMethod}</strong> does not have visualization data available for the selected criteria combination.
          </div>
        )}
      </main>
    </div>
  );
};

export default CompareView;
