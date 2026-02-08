/**
 * Main App component for Multiview Visualizer.
 */

import React, { useState, useEffect } from 'react';
import { DatasetIndex, VisualizationMode, VisualizationData, TripletLogEntry } from './types/manifest';
import { loadDatasetIndex, loadVisualizationData, loadBenchmarkResults, loadTripletLogs, loadEmbeddingsData } from './utils/dataLoader';
import { ModeRenderer } from './render/ModeRenderer';
import { Leaderboard, BenchmarkResults } from './components/Leaderboard';
import { TripletList } from './components/TripletList';
import './App.css';

type TaxonomySection = {
  title: string;
  content: string;
};

type NeighborItem = {
  index: number;
  score: number;
  text: string;
};

type CurrentCriterionNeighbors = {
  neighbors: NeighborItem[];
  novelNeighbors: NeighborItem[];
  baselineNeighbors: NeighborItem[];
  metricLabel: string;
  baselineMethodName?: string;
  baselineMetricLabel?: string;
  error?: string;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function normalizeSectionContent(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (isObject(value) && typeof value.summary_guidance === 'string') {
    return value.summary_guidance;
  }
  return JSON.stringify(value, null, 2);
}

function extractTaxonomySectionsFromTriplet(triplet: unknown): TaxonomySection[] {
  if (!isObject(triplet)) {
    return [];
  }

  const keyToTitle: Record<string, string> = {
    category_schema: 'Category taxonomy',
    tag_schema: 'Tag taxonomy',
    spurious_tag_schema: 'Spurious tag taxonomy',
    summary_guidance: 'Summary guidance taxonomy',
  };

  const sections: TaxonomySection[] = [];
  const seen = new Set<string>();

  const visit = (value: unknown) => {
    if (!isObject(value)) {
      return;
    }

    for (const [key, title] of Object.entries(keyToTitle)) {
      if (key in value) {
        const rawSection = value[key];
        const content = normalizeSectionContent(rawSection);
        if (content && !seen.has(content)) {
          seen.add(content);
          sections.push({ title, content });
        }
      }
    }

    for (const nestedValue of Object.values(value)) {
      if (isObject(nestedValue)) {
        visit(nestedValue);
      }
    }
  };

  visit(triplet);
  return sections;
}

function cosineSimilarity(
  embeddings: Float32Array,
  dim: number,
  idxA: number,
  idxB: number
): number {
  const offsetA = idxA * dim;
  const offsetB = idxB * dim;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < dim; i++) {
    const valA = embeddings[offsetA + i];
    const valB = embeddings[offsetB + i];
    dotProduct += valA * valB;
    normA += valA * valA;
    normB += valB * valB;
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function l1Distance(
  embeddings: Float32Array,
  dim: number,
  idxA: number,
  idxB: number
): number {
  const offsetA = idxA * dim;
  const offsetB = idxB * dim;
  let sum = 0;
  for (let i = 0; i < dim; i++) {
    sum += Math.abs(embeddings[offsetA + i] - embeddings[offsetB + i]);
  }
  return sum;
}

function rankNeighborsForDoc(
  embeddings: Float32Array,
  documents: string[],
  queryIndex: number,
  usePseudologit: boolean,
  embeddingDimHint?: number
): { neighbors: NeighborItem[]; metricLabel: string } {
  const nDocs = documents.length;
  if (queryIndex < 0 || queryIndex >= nDocs) {
    return { neighbors: [], metricLabel: usePseudologit ? 'L1 distance' : 'Cosine similarity' };
  }

  const isPrecomputedMatrix =
    embeddings.length === nDocs * nDocs &&
    (embeddingDimHint === undefined || embeddingDimHint === nDocs) &&
    (() => {
      // Avoid misclassifying true NxD embeddings when D == N by requiring near-symmetry.
      if (nDocs <= 1) return true;
      let checked = 0;
      let symmetricPairs = 0;
      const step = Math.max(1, Math.floor(nDocs / 11));
      const maxChecks = 24;
      for (let i = 0; i < nDocs && checked < maxChecks; i += step) {
        const j = (i + step) % nDocs;
        if (i === j) continue;
        const a = embeddings[i * nDocs + j];
        const b = embeddings[j * nDocs + i];
        if (Number.isFinite(a) && Number.isFinite(b)) {
          checked += 1;
          if (Math.abs(a - b) <= 1e-4) {
            symmetricPairs += 1;
          }
        }
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
    return {
      neighbors: scored.map(({ index, score }) => ({
        index,
        score,
        text: documents[index],
      })),
      metricLabel: 'Similarity score',
    };
  }

  const dim = embeddings.length / nDocs;
  for (let j = 0; j < nDocs; j++) {
    if (j === queryIndex) continue;
    const score = usePseudologit
      ? l1Distance(embeddings, dim, queryIndex, j)
      : cosineSimilarity(embeddings, dim, queryIndex, j);
    scored.push({ index: j, score });
  }

  if (usePseudologit) {
    scored.sort((a, b) => a.score - b.score);
  } else {
    scored.sort((a, b) => b.score - a.score);
  }

  return {
    neighbors: scored.map(({ index, score }) => ({
      index,
      score,
      text: documents[index],
    })),
    metricLabel: usePseudologit ? 'L1 distance (lower is closer)' : 'Cosine similarity',
  };
}

function isImagePlaceholder(text: string | undefined): boolean {
  if (!text) return true;
  const normalized = text.trim().toLowerCase();
  return !normalized || normalized === '<image>';
}

function getImageOnlyFallbackText(docIndex: number): string {
  return `Image-only document #${docIndex}`;
}

function getTextFromTripletDoc(doc: unknown): string | null {
  if (typeof doc === 'string') {
    return isImagePlaceholder(doc) ? null : doc;
  }

  if (!isObject(doc)) {
    return null;
  }

  const textValue = doc.text;
  if (typeof textValue === 'string' && !isImagePlaceholder(textValue)) {
    return textValue;
  }

  const ignoredKeys = new Set(['text', 'image_path', 'embedding_viz', '_metadata']);
  const parts: string[] = [];
  for (const [key, value] of Object.entries(doc)) {
    if (ignoredKeys.has(key) || value === null || value === undefined) {
      continue;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (trimmed) {
        parts.push(`${key}: ${trimmed}`);
      }
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      parts.push(`${key}: ${String(value)}`);
    }
  }

  if (parts.length === 0) {
    return null;
  }

  return parts.join(' | ');
}

function getBenchmarkRoots(index: DatasetIndex): string[] {
  const roots = new Set<string>();
  for (const key of Object.keys(index)) {
    const [root] = key.split('/');
    if (root) {
      roots.add(root);
    }
  }
  return Array.from(roots).sort();
}

function getViewKinds(index: DatasetIndex, benchmarkRoot: string): string[] {
  const kinds = new Set<string>();
  if (index[benchmarkRoot]) {
    kinds.add('triples');
  }
  for (const key of Object.keys(index)) {
    if (!key.startsWith(`${benchmarkRoot}/`)) {
      continue;
    }
    const parts = key.split('/');
    if (parts.length >= 2 && parts[1]) {
      kinds.add(parts[1]);
    }
  }
  return Array.from(kinds).sort((a, b) => {
    const preferredOrder = ['triples', 'corpus'];
    const ai = preferredOrder.indexOf(a);
    const bi = preferredOrder.indexOf(b);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return a.localeCompare(b);
  });
}

function resolveBenchmarkKey(
  index: DatasetIndex,
  benchmarkRoot: string,
  viewKind: string
): string {
  const namespaced = `${benchmarkRoot}/${viewKind}`;
  if (index[namespaced]) {
    return namespaced;
  }
  if (viewKind === 'triples' && index[benchmarkRoot]) {
    return benchmarkRoot;
  }
  const fallback = Object.keys(index).find(
    (key) => key === benchmarkRoot || key.startsWith(`${benchmarkRoot}/`)
  );
  return fallback || '';
}

function preferredMode(modes: string[]): VisualizationMode {
  if (modes.includes('tsne')) return 'tsne';
  return (modes[0] || 'tsne') as VisualizationMode;
}

const App: React.FC = () => {
  const [index, setIndex] = useState<DatasetIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selection state
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('');
  const [selectedViewKind, setSelectedViewKind] = useState<string>('triples');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedCriterion, setSelectedCriterion] = useState<string>('');
  const [selectedMethod, setSelectedMethod] = useState<string>('');
  const [selectedMode, setSelectedMode] = useState<VisualizationMode>('tsne');
  const [displayMode, setDisplayMode] = useState<'points' | 'thumbnails'>('thumbnails');

  // Benchmark results
  const [results, setResults] = useState<BenchmarkResults | null>(null);

  // Visualization data
  const [vizData, setVizData] = useState<VisualizationData | null>(null);
  const [loadingViz, setLoadingViz] = useState(false);

  // Triplet logs for methods without embeddings
  const [tripletLogs, setTripletLogs] = useState<TripletLogEntry[] | null>(null);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState<number | null>(null);
  const [topK, setTopK] = useState<number>(5);
  const [currentCriterionNeighbors, setCurrentCriterionNeighbors] = useState<CurrentCriterionNeighbors | null>(null);
  const [loadingNeighbors, setLoadingNeighbors] = useState<boolean>(false);
  const [neighborsError, setNeighborsError] = useState<string | null>(null);

  const isPseudologitSelected = selectedMethod.toLowerCase().includes('pseudologit');
  const isCorpusView = selectedViewKind === 'corpus';
  const benchmarkRoots = React.useMemo(
    () => (index ? getBenchmarkRoots(index) : []),
    [index]
  );
  const availableViewKinds = React.useMemo(
    () => (index && selectedBenchmark ? getViewKinds(index, selectedBenchmark) : []),
    [index, selectedBenchmark]
  );
  const activeBenchmarkKey = React.useMemo(
    () =>
      index && selectedBenchmark
        ? resolveBenchmarkKey(index, selectedBenchmark, selectedViewKind)
        : '',
    [index, selectedBenchmark, selectedViewKind]
  );

  useEffect(() => {
    if (!selectedBenchmark || availableViewKinds.length === 0) {
      return;
    }
    if (!availableViewKinds.includes(selectedViewKind)) {
      const preferredView = availableViewKinds.includes('triples')
        ? 'triples'
        : availableViewKinds[0];
      setSelectedViewKind(preferredView);
    }
  }, [selectedBenchmark, availableViewKinds, selectedViewKind]);

  const isDocumentRewriteSelected = React.useMemo(() => {
    if (isCorpusView) {
      return false;
    }
    if (selectedMethod.toLowerCase().includes('document_rewrite')) {
      return true;
    }
    if (!vizData?.triplets?.length) {
      return false;
    }
    return vizData.triplets.some((triplet) =>
      triplet.method_type === 'document_rewrite' ||
      typeof triplet.anchor_summary === 'string' ||
      typeof triplet.positive_summary === 'string' ||
      typeof triplet.negative_summary === 'string'
    );
  }, [isCorpusView, selectedMethod, vizData?.triplets]);

  const documentRewriteSummaryById = React.useMemo(() => {
    const summaryById = new Map<number, string>();
    if (!isDocumentRewriteSelected || !vizData?.triplets?.length) {
      return summaryById;
    }

    for (const triplet of vizData.triplets) {
      if (typeof triplet.anchor_summary === 'string' && triplet.anchor_summary.trim()) {
        summaryById.set(triplet.anchor_id, triplet.anchor_summary);
      }
      if (typeof triplet.positive_summary === 'string' && triplet.positive_summary.trim()) {
        summaryById.set(triplet.positive_id, triplet.positive_summary);
      }
      if (typeof triplet.negative_summary === 'string' && triplet.negative_summary.trim()) {
        summaryById.set(triplet.negative_id, triplet.negative_summary);
      }
    }

    return summaryById;
  }, [isDocumentRewriteSelected, vizData?.triplets]);

  const tripletSourceSnippetById = React.useMemo(() => {
    const sourceById = new Map<number, string>();
    if (!vizData?.triplets?.length) {
      return sourceById;
    }

    const assign = (docId: number, doc: unknown) => {
      if (sourceById.has(docId)) {
        return;
      }
      const text = getTextFromTripletDoc(doc);
      if (text && text.trim()) {
        sourceById.set(docId, text);
      }
    };

    for (const triplet of vizData.triplets) {
      assign(triplet.anchor_id, triplet.anchor);
      assign(triplet.positive_id, triplet.positive);
      assign(triplet.negative_id, triplet.negative);
    }

    return sourceById;
  }, [vizData?.triplets]);

  const displayDocuments = React.useMemo(() => {
    if (!vizData) {
      return [] as string[];
    }

    return vizData.documents.map((docText, docIdx) => {
      if (!isImagePlaceholder(docText)) {
        return docText;
      }

      const rewriteSummary = documentRewriteSummaryById.get(docIdx);
      if (rewriteSummary && rewriteSummary.trim()) {
        return rewriteSummary;
      }

      const sourceSnippet = tripletSourceSnippetById.get(docIdx);
      if (sourceSnippet && sourceSnippet.trim()) {
        return sourceSnippet;
      }

      return getImageOnlyFallbackText(docIdx);
    });
  }, [vizData, documentRewriteSummaryById, tripletSourceSnippetById]);

  const getDisplayTextForDoc = (docIndex: number, fallbackText: string): string => {
    const resolvedText = displayDocuments[docIndex];
    if (resolvedText && resolvedText.trim() && !isImagePlaceholder(resolvedText)) {
      return resolvedText;
    }
    if (fallbackText && fallbackText.trim() && !isImagePlaceholder(fallbackText)) {
      return fallbackText;
    }
    return getImageOnlyFallbackText(docIndex);
  };

  const selectedTaxonomySections = React.useMemo(() => {
    if (isCorpusView || !isPseudologitSelected || !vizData?.triplets?.length) {
      return [] as TaxonomySection[];
    }

    for (const triplet of vizData.triplets.slice(0, 10)) {
      const sections = extractTaxonomySectionsFromTriplet(triplet);
      if (sections.length > 0) {
        return sections;
      }
    }

    return [] as TaxonomySection[];
  }, [isCorpusView, isPseudologitSelected, vizData?.triplets]);

  useEffect(() => {
    setSelectedDocumentIndex(null);
    setCurrentCriterionNeighbors(null);
    setNeighborsError(null);
  }, [selectedBenchmark, selectedViewKind, selectedDataset, selectedCriterion, selectedMethod]);

  useEffect(() => {
    if (
      selectedDocumentIndex === null ||
      !index ||
      !activeBenchmarkKey ||
      !selectedDataset ||
      !selectedCriterion ||
      !selectedMethod
    ) {
      setCurrentCriterionNeighbors(null);
      setLoadingNeighbors(false);
      setNeighborsError(null);
      return;
    }

    const methodsForCriterion =
      index[activeBenchmarkKey]?.[selectedDataset]?.[selectedCriterion];
    if (!methodsForCriterion) {
      setCurrentCriterionNeighbors(null);
      setLoadingNeighbors(false);
      setNeighborsError('No methods found for selected criterion.');
      return;
    }

    const methodData = methodsForCriterion[selectedMethod];
    if (!methodData || methodData.has_embeddings === false) {
      setCurrentCriterionNeighbors(null);
      setLoadingNeighbors(false);
      setNeighborsError(`Selected method '${selectedMethod}' has no embeddings for this criterion.`);
      return;
    }

    const baselineMethodName = Object.keys(methodsForCriterion).find(
      (name) =>
        name !== selectedMethod &&
        methodsForCriterion[name]?.has_embeddings !== false &&
        /no[_-]?(instructions?|criteria|criterion|annotation|annotations?)|voyage_multimodal_3_5/i.test(name)
    );

    let cancelled = false;
    setLoadingNeighbors(true);
    setNeighborsError(null);

    Promise.all([
      loadEmbeddingsData(methodData.path),
      baselineMethodName ? loadEmbeddingsData(methodsForCriterion[baselineMethodName].path) : Promise.resolve(null),
    ])
      .then(([currentData, baselineData]) => {
        if (cancelled) return;

        const { embeddings, documents, manifest } = currentData;
        if (selectedDocumentIndex >= documents.length) {
          setCurrentCriterionNeighbors({
            neighbors: [],
            novelNeighbors: [],
            baselineNeighbors: [],
            metricLabel: 'Unavailable',
            baselineMethodName,
            error: `Selected document index ${selectedDocumentIndex} is out of range for this criterion.`,
          });
          setLoadingNeighbors(false);
          return;
        }

        const usePseudologit =
          manifest.method?.toLowerCase().includes('pseudologit') ||
          selectedMethod.toLowerCase().includes('pseudologit');
        const ranked = rankNeighborsForDoc(
          embeddings,
          documents,
          selectedDocumentIndex,
          usePseudologit,
          manifest.embedding_dim
        );

        const neighbors = ranked.neighbors.slice(0, topK);
        let novelNeighbors: NeighborItem[] = neighbors;
        let baselineNeighbors: NeighborItem[] = [];
        let baselineMetricLabel: string | undefined;

        if (baselineData && selectedDocumentIndex < baselineData.documents.length) {
          const baselineUsePseudologit =
            baselineData.manifest.method?.toLowerCase().includes('pseudologit') ||
            (baselineMethodName ? baselineMethodName.toLowerCase().includes('pseudologit') : false);
          const baselineRanked = rankNeighborsForDoc(
            baselineData.embeddings,
            baselineData.documents,
            selectedDocumentIndex,
            baselineUsePseudologit,
            baselineData.manifest.embedding_dim
          );
          baselineNeighbors = baselineRanked.neighbors.slice(0, topK);
          const baselineTopKSet = new Set(
            baselineNeighbors.map((n) => n.index)
          );
          novelNeighbors = ranked.neighbors
            .filter((n) => !baselineTopKSet.has(n.index))
            .slice(0, topK);
          baselineMetricLabel = baselineRanked.metricLabel;
        }

        setCurrentCriterionNeighbors({
          neighbors,
          novelNeighbors,
          baselineNeighbors,
          metricLabel: ranked.metricLabel,
          baselineMethodName,
          baselineMetricLabel,
        });
        setLoadingNeighbors(false);
      })
      .catch((err: any) => {
        if (cancelled) return;
        setNeighborsError(err?.message || 'Failed to compute neighbors.');
        setCurrentCriterionNeighbors(null);
        setLoadingNeighbors(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    selectedDocumentIndex,
    index,
    activeBenchmarkKey,
    selectedDataset,
    selectedCriterion,
    selectedMethod,
    topK,
  ]);

  // Load index on mount
  useEffect(() => {
    loadDatasetIndex()
      .then((idx) => {
        setIndex(idx);

        // Auto-select first available benchmark/dataset/criterion/method/mode
        const roots = getBenchmarkRoots(idx);
        if (roots.length > 0) {
          const firstBenchmark = roots[0];
          const views = getViewKinds(idx, firstBenchmark);
          const firstView = views.includes('triples') ? 'triples' : (views[0] || 'triples');
          const firstBenchmarkKey = resolveBenchmarkKey(idx, firstBenchmark, firstView);
          const benchmarkData = firstBenchmarkKey ? idx[firstBenchmarkKey] : undefined;
          const datasets = benchmarkData ? Object.keys(benchmarkData) : [];
          if (datasets.length > 0) {
            const firstDataset = datasets[0];
            const criteria = Object.keys(benchmarkData![firstDataset]);
            if (criteria.length > 0) {
              const firstCriterion = criteria[0];
              const methods = Object.keys(benchmarkData![firstDataset][firstCriterion]);
              if (methods.length > 0) {
                const firstMethod = methods[0];
                const modes = benchmarkData![firstDataset][firstCriterion][firstMethod].modes;
                if (modes.length > 0) {
                  setSelectedBenchmark(firstBenchmark);
                  setSelectedViewKind(firstView);
                  setSelectedDataset(firstDataset);
                  setSelectedCriterion(firstCriterion);
                  setSelectedMethod(firstMethod);
                  setSelectedMode(preferredMode(modes));

                  // Load results for this benchmark
                  loadBenchmarkResults(firstBenchmarkKey).then(setResults);
                }
              }
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

  // Load results when benchmark changes
  useEffect(() => {
    if (activeBenchmarkKey) {
      loadBenchmarkResults(activeBenchmarkKey).then(setResults);
    }
  }, [activeBenchmarkKey]);

  // Load visualization data when selection changes
  useEffect(() => {
    if (!index || !activeBenchmarkKey || !selectedDataset || !selectedCriterion || !selectedMethod || !selectedMode) {
      setVizData(null);
      setTripletLogs(null);
      return;
    }

    const benchmarkData = index[activeBenchmarkKey];
    if (!benchmarkData || !benchmarkData[selectedDataset]) {
      setVizData(null);
      setTripletLogs(null);
      return;
    }

    const methodData = benchmarkData[selectedDataset][selectedCriterion]?.[selectedMethod];
    if (!methodData) {
      // Method doesn't exist in index
      setVizData(null);
      setTripletLogs(null);
      setLoadingViz(false);
      setError(null);
      return;
    }

    setLoadingViz(true);
    setError(null);

    // Check if method has embeddings
    if (methodData.has_embeddings === false) {
      if (isCorpusView) {
        setTripletLogs(null);
        setVizData(null);
        setLoadingViz(false);
        return;
      }
      // Load triplet logs for methods without embeddings
      loadTripletLogs(methodData.path)
        .then((logs) => {
          setTripletLogs(logs || null);
          setVizData(null);
          setLoadingViz(false);
        })
        .catch((err) => {
          setError(`Failed to load triplet logs: ${err.message}`);
          setLoadingViz(false);
          setVizData(null);
          setTripletLogs(null);
        });
    } else {
      // Load visualization data for methods with embeddings
      setTripletLogs(null);
      loadVisualizationData(methodData.path, selectedMode)
        .then((data) => {
          setVizData(
            isCorpusView
              ? {
                  ...data,
                  triplets: undefined,
                }
              : data
          );
          setLoadingViz(false);
        })
        .catch((err) => {
          setError(`Failed to load visualization: ${err.message}`);
          setLoadingViz(false);
          setVizData(null);
        });
    }
  }, [index, activeBenchmarkKey, selectedDataset, selectedCriterion, selectedMethod, selectedMode, isCorpusView]);

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading dataset index...</div>
      </div>
    );
  }

  if (!index || Object.keys(index).length === 0) {
    return (
      <div className="app">
        <div className="error">
          <h2>No visualizations found</h2>
          <p>Please run evaluation first using:</p>
          <pre>
            uv run python scripts/run_eval.py --config-name benchmark_debug
          </pre>
          <p>Visualizations are automatically generated after evaluation.</p>
        </div>
      </div>
    );
  }

  const benchmarkData = activeBenchmarkKey ? index[activeBenchmarkKey] : undefined;
  const datasets = benchmarkData
    ? Object.keys(benchmarkData)
    : [];
  const criteria = selectedDataset && benchmarkData?.[selectedDataset]
    ? Object.keys(benchmarkData[selectedDataset])
    : [];
  const methods = selectedDataset && selectedCriterion && benchmarkData?.[selectedDataset]?.[selectedCriterion]
    ? Object.keys(benchmarkData[selectedDataset][selectedCriterion])
    : [];
  const modes = selectedDataset && selectedCriterion && selectedMethod && benchmarkData?.[selectedDataset]?.[selectedCriterion]?.[selectedMethod]
    ? benchmarkData[selectedDataset][selectedCriterion][selectedMethod].modes
    : [];

  // Get task name for leaderboard - extract from ANY method's path in this dataset/criterion
  // (Even if selected method doesn't have embeddings, we want to show the leaderboard)
  // Path format: "benchmark_fuzzy_debug2/gsm8k__final_expression__tag__5/method_name"
  const taskName = (() => {
    if (!benchmarkData || !selectedDataset || !selectedCriterion) return '';

    const criterionMethods = benchmarkData[selectedDataset]?.[selectedCriterion];
    if (!criterionMethods) return '';

    // Get task name from first available method
    const firstMethod = Object.values(criterionMethods)[0];
    if (!firstMethod?.path) return '';
    const pathParts = firstMethod.path.split('/');
    return pathParts.length >= 2 ? pathParts[pathParts.length - 2] : '';
  })();

  return (
    <div className="app">
      <header className="header">
        <div className="header-top-row">
          <h1>🔬 View X by Y</h1>
          <a href="/compare.html" className="nav-compare-btn">Compare Criteria</a>
        </div>
        <p className="header-subtitle">
          Represent documents <strong>{selectedDataset || 'X'}</strong> according to criteria <strong>{selectedCriterion || 'Y'}</strong>
        </p>
        <div className="controls">
          <div className="control-group">
            <label>Experiment:</label>
            <select
              value={selectedBenchmark}
              onChange={(e) => {
                const newBenchmark = e.target.value;
                const newViews = getViewKinds(index, newBenchmark);
                const newView = newViews.includes(selectedViewKind)
                  ? selectedViewKind
                  : (newViews.includes('triples') ? 'triples' : (newViews[0] || 'triples'));
                const benchmarkKey = resolveBenchmarkKey(index, newBenchmark, newView);
                const benchmarkSelectionData = benchmarkKey ? index[benchmarkKey] : undefined;
                setSelectedBenchmark(newBenchmark);
                setSelectedViewKind(newView);
                // Reset downstream selections
                const newDatasets = benchmarkSelectionData ? Object.keys(benchmarkSelectionData) : [];
                if (newDatasets.length > 0) {
                  const newDataset = newDatasets[0];
                  setSelectedDataset(newDataset);
                  const newCriteria = Object.keys(benchmarkSelectionData![newDataset]);
                  if (newCriteria.length > 0) {
                    const newCriterion = newCriteria[0];
                    setSelectedCriterion(newCriterion);
                    const newMethods = Object.keys(benchmarkSelectionData![newDataset][newCriterion]);
                    if (newMethods.length > 0) {
                      const newMethod = newMethods[0];
                      setSelectedMethod(newMethod);
                      const newModes = benchmarkSelectionData![newDataset][newCriterion][newMethod].modes;
                      if (newModes.length > 0) {
                        setSelectedMode(preferredMode(newModes));
                      }
                    }
                  }
                }
              }}
            >
              {benchmarkRoots.map((benchmark) => (
                <option key={benchmark} value={benchmark}>
                  {benchmark}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>View:</label>
            <select
              value={selectedViewKind}
              onChange={(e) => {
                const newView = e.target.value;
                const benchmarkKey = resolveBenchmarkKey(index, selectedBenchmark, newView);
                const benchmarkSelectionData = benchmarkKey ? index[benchmarkKey] : undefined;
                setSelectedViewKind(newView);
                // Reset downstream selections
                const newDatasets = benchmarkSelectionData ? Object.keys(benchmarkSelectionData) : [];
                if (newDatasets.length > 0) {
                  const newDataset = newDatasets[0];
                  setSelectedDataset(newDataset);
                  const newCriteria = Object.keys(benchmarkSelectionData![newDataset]);
                  if (newCriteria.length > 0) {
                    const newCriterion = newCriteria[0];
                    setSelectedCriterion(newCriterion);
                    const newMethods = Object.keys(benchmarkSelectionData![newDataset][newCriterion]);
                    if (newMethods.length > 0) {
                      const newMethod = newMethods[0];
                      setSelectedMethod(newMethod);
                      const newModes = benchmarkSelectionData![newDataset][newCriterion][newMethod].modes;
                      if (newModes.length > 0) {
                        setSelectedMode(preferredMode(newModes));
                      }
                    }
                  }
                }
              }}
            >
              {availableViewKinds.map((view) => (
                <option key={view} value={view}>
                  {view}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Dataset:</label>
            <select
              value={selectedDataset}
              onChange={(e) => {
                setSelectedDataset(e.target.value);
                // Reset downstream selections
                const newCriteria = benchmarkData ? Object.keys(benchmarkData[e.target.value]) : [];
                if (newCriteria.length > 0) {
                  const newCriterion = newCriteria[0];
                  setSelectedCriterion(newCriterion);
                  const newMethods = benchmarkData ? Object.keys(benchmarkData[e.target.value][newCriterion]) : [];
                  if (newMethods.length > 0) {
                    const newMethod = newMethods[0];
                    setSelectedMethod(newMethod);
                    const newModes = benchmarkData ? benchmarkData[e.target.value][newCriterion][newMethod].modes : [];
                    if (newModes.length > 0) {
                      setSelectedMode(preferredMode(newModes));
                    }
                  }
                }
              }}
            >
              {datasets.map((dataset) => (
                <option key={dataset} value={dataset}>
                  {dataset}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Criterion:</label>
            <select
              value={selectedCriterion}
              onChange={(e) => {
                setSelectedCriterion(e.target.value);
                // Reset downstream selections
                const newMethods = benchmarkData ? Object.keys(benchmarkData[selectedDataset][e.target.value]) : [];
                if (newMethods.length > 0) {
                  const newMethod = newMethods[0];
                  setSelectedMethod(newMethod);
                  const newModes = benchmarkData ? benchmarkData[selectedDataset][e.target.value][newMethod].modes : [];
                  if (newModes.length > 0) {
                    setSelectedMode(preferredMode(newModes));
                  }
                }
              }}
            >
              {criteria.map((criterion) => (
                <option key={criterion} value={criterion}>
                  {criterion}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Method:</label>
            <select
              value={selectedMethod}
              onChange={(e) => {
                setSelectedMethod(e.target.value);
                // Reset mode
                const newModes = benchmarkData ? benchmarkData[selectedDataset][selectedCriterion][e.target.value].modes : [];
                if (newModes.length > 0) {
                  setSelectedMode(preferredMode(newModes));
                }
              }}
            >
              {methods.map((method) => (
                <option key={method} value={method}>
                  {method}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Mode:</label>
            <div className="mode-buttons">
              {/* Show GRAPH and HEATMAP if not already in modes */}
              {!modes.includes('heatmap') && (
                <button
                  className={selectedMode === 'heatmap' ? 'active' : ''}
                  onClick={() => setSelectedMode('heatmap')}
                >
                  HEATMAP
                </button>
              )}
              {!modes.includes('graph') && (
                <button
                  className={selectedMode === 'graph' ? 'active' : ''}
                  onClick={() => setSelectedMode('graph')}
                >
                  GRAPH
                </button>
              )}
              {[...modes].reverse().map((mode) => (
                <button
                  key={mode}
                  className={selectedMode === mode ? 'active' : ''}
                  onClick={() => setSelectedMode(mode as VisualizationMode)}
                >
                  {mode.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="control-group">
            <label>Display:</label>
            <div className="mode-buttons">
              <button
                className={displayMode === 'points' ? 'active' : ''}
                onClick={() => setDisplayMode('points')}
              >
                Points
              </button>
              <button
                className={displayMode === 'thumbnails' ? 'active' : ''}
                onClick={() => setDisplayMode('thumbnails')}
                disabled={!vizData?.thumbnailUrls || vizData.thumbnailUrls.every(url => url === null)}
                title={!vizData?.thumbnailUrls || vizData.thumbnailUrls.every(url => url === null) ? 'Thumbnails not available for this dataset' : ''}
              >
                Thumbnails
              </button>
            </div>
          </div>
        </div>

        <div className="regen-command-panel">
          <span>To regenerate results, run:</span>{' '}
          <code>{`python scripts/run_eval.py --config-name ${selectedBenchmark}`}</code>
        </div>
      </header>

      <main className="main">
        {/* Leaderboard (hidden in corpus view) */}
        {!isCorpusView && (
          <Leaderboard
            results={results}
            currentTask={taskName}
            currentMethod={selectedMethod}
            methodsWithEmbeddings={
              benchmarkData && selectedDataset && selectedCriterion
                ? Object.entries(benchmarkData[selectedDataset]?.[selectedCriterion] || {})
                    .filter(([_, data]) => data.has_embeddings !== false)
                    .map(([name, _]) => name)
                : []
            }
            onMethodSelect={(method) => {
              setSelectedMethod(method);
              // Reset mode for new method (if it has embeddings)
              const methodData = benchmarkData?.[selectedDataset]?.[selectedCriterion]?.[method];
              if (methodData && methodData.modes.length > 0) {
                setSelectedMode(preferredMode(methodData.modes));
              }
            }}
          />
        )}

        {error && <div className="error">{error}</div>}
        {loadingViz && <div className="loading">Loading...</div>}
        {!vizData && !tripletLogs && !loadingViz && !error && selectedMethod && (
          <div className="loading" style={{
            background: '#fff3cd',
            color: '#856404',
            border: '1px solid #ffc107',
            borderRadius: '4px',
            padding: '20px'
          }}>
            <strong>{selectedMethod}</strong> does not have data available.
            <br />
            <span style={{ fontSize: '14px', marginTop: '8px', display: 'block' }}>
              This may indicate an error during evaluation or that the method was skipped.
            </span>
          </div>
        )}
        {!isCorpusView && tripletLogs && !loadingViz && (
          <div className="visualization">
            <TripletList tripletLogs={tripletLogs} />
          </div>
        )}
        {vizData && !loadingViz && (
          <div className="visualization">
            {!isCorpusView && isPseudologitSelected && (
              <div className="taxonomy-panel">
                <h3>Taxonomy (selected pseudologit method)</h3>
                {selectedTaxonomySections.length > 0 ? (
                  selectedTaxonomySections.map((section, idx) => (
                    <div key={`${section.title}-${idx}`} className="taxonomy-section">
                      <h4>{section.title}</h4>
                      <pre>{section.content}</pre>
                    </div>
                  ))
                ) : (
                  <p className="taxonomy-empty">
                    Taxonomy metadata is not present in this visualization artifact for{' '}
                    <code>{selectedMethod}</code>.
                  </p>
                )}
              </div>
            )}

            <div className="viz-info">
              <span>{vizData.documents.length} documents</span>
              <span>{vizData.manifest.embedding_dim}D embeddings</span>
              {!isCorpusView && vizData.triplets && <span>{vizData.triplets.length} triplets</span>}
            </div>
            <ModeRenderer
              mode={selectedMode}
              data={{ ...vizData, documents: displayDocuments }}
              displayMode={displayMode}
              width={1000}
              height={700}
              onSelectDocument={setSelectedDocumentIndex}
            />
          </div>
        )}

        {vizData && selectedDocumentIndex !== null && (
          <div className="neighbors-panel">
            <div className="neighbors-header">
              <h3>Top-{topK} nearest neighbors ({selectedCriterion})</h3>
              <div className="neighbors-controls">
                <label>Top K:</label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                >
                  {[3, 5, 10, 20].map((k) => (
                    <option key={k} value={k}>{k}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="neighbors-selected-doc">
              <strong>Selected document #{selectedDocumentIndex}</strong>
              <div className="neighbors-doc-content">
                {vizData.thumbnailUrls?.[selectedDocumentIndex] && (
                  <img
                    src={vizData.thumbnailUrls[selectedDocumentIndex] || ''}
                    alt={`Doc ${selectedDocumentIndex}`}
                    className="neighbor-thumb"
                  />
                )}
                <div className="neighbors-doc-text">
                  {(() => {
                    const rawText = (vizData.rawDocuments?.[selectedDocumentIndex] ?? vizData.documents[selectedDocumentIndex]) || '';
                    const displayText = getDisplayTextForDoc(
                      selectedDocumentIndex,
                      vizData.documents[selectedDocumentIndex] || ''
                    );
                    return (
                      <>
                        <div className="neighbor-doc-preview">
                          {rawText.slice(0, 400)}
                          {rawText.length > 400 ? '...' : ''}
                        </div>
                        {displayText !== rawText && (
                          <div className="neighbor-summary">
                            {displayText.slice(0, 400)}
                            {displayText.length > 400 ? '...' : ''}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              </div>
            </div>

            {loadingNeighbors && <div className="loading">Computing nearest neighbors...</div>}
            {neighborsError && <div className="error">{neighborsError}</div>}

            {!loadingNeighbors && !neighborsError && currentCriterionNeighbors && (
              <div className="neighbors-card is-selected-criterion">
                {currentCriterionNeighbors.error ? (
                  <div className="neighbors-card-error">{currentCriterionNeighbors.error}</div>
                ) : (
                  <div className="neighbors-columns">
                    <div>
                      <div className="neighbors-subtitle">
                        Top-{topK} ({currentCriterionNeighbors.metricLabel})
                      </div>
                      <ol className="neighbors-list">
                        {currentCriterionNeighbors.neighbors.map((neighbor) => (
                          <li key={`top-${neighbor.index}`}>
                            <div className="neighbor-topline">
                              <span>Doc #{neighbor.index}</span>
                              <span>{neighbor.score.toFixed(4)}</span>
                            </div>
                            <div className="neighbor-content">
                              {vizData.thumbnailUrls?.[neighbor.index] && (
                                <img
                                  src={vizData.thumbnailUrls[neighbor.index] || ''}
                                  alt={`Doc ${neighbor.index}`}
                                  className="neighbor-thumb"
                                />
                              )}
                              <div className="neighbor-text">
                                {(() => {
                                  const rawText = (vizData.rawDocuments?.[neighbor.index] ?? vizData.documents[neighbor.index]) || '';
                                  const displayText = getDisplayTextForDoc(neighbor.index, neighbor.text);
                                  return (
                                    <>
                                      <div className="neighbor-doc-preview">
                                        {rawText.slice(0, 80)}
                                        {rawText.length > 80 ? '...' : ''}
                                      </div>
                                      {displayText !== rawText && (
                                        <div className="neighbor-summary">
                                          {displayText.slice(0, 120)}
                                          {displayText.length > 120 ? '...' : ''}
                                        </div>
                                      )}
                                    </>
                                  );
                                })()}
                              </div>
                            </div>
                          </li>
                        ))}
                      </ol>
                    </div>

                    <div>
                      <div className="neighbors-subtitle">
                        Baseline Top-{topK}
                        {currentCriterionNeighbors.baselineMethodName
                          ? ` (${currentCriterionNeighbors.baselineMethodName})`
                          : ' (no baseline found)'}
                      </div>
                      {currentCriterionNeighbors.baselineMethodName ? (
                        <>
                          {currentCriterionNeighbors.baselineMetricLabel && (
                            <div className="neighbors-baseline-metric">
                              {currentCriterionNeighbors.baselineMetricLabel}
                            </div>
                          )}
                          <ol className="neighbors-list">
                            {currentCriterionNeighbors.baselineNeighbors.map((neighbor) => (
                              <li key={`baseline-${neighbor.index}`}>
                                <div className="neighbor-topline">
                                  <span>Doc #{neighbor.index}</span>
                                  <span>{neighbor.score.toFixed(4)}</span>
                                </div>
                                <div className="neighbor-content">
                                  {vizData.thumbnailUrls?.[neighbor.index] && (
                                    <img
                                      src={vizData.thumbnailUrls[neighbor.index] || ''}
                                      alt={`Doc ${neighbor.index}`}
                                      className="neighbor-thumb"
                                    />
                                  )}
                                  <div className="neighbor-text">
                                    {(() => {
                                      const rawText = (vizData.rawDocuments?.[neighbor.index] ?? vizData.documents[neighbor.index]) || '';
                                      const displayText = getDisplayTextForDoc(neighbor.index, neighbor.text);
                                      return (
                                        <>
                                          <div className="neighbor-doc-preview">
                                            {rawText.slice(0, 80)}
                                            {rawText.length > 80 ? '...' : ''}
                                          </div>
                                          {displayText !== rawText && (
                                            <div className="neighbor-summary">
                                              {displayText.slice(0, 120)}
                                              {displayText.length > 120 ? '...' : ''}
                                            </div>
                                          )}
                                        </>
                                      );
                                    })()}
                                  </div>
                                </div>
                              </li>
                            ))}
                          </ol>
                        </>
                      ) : (
                        <div className="neighbors-card-error">
                          No baseline method found for this criterion.
                        </div>
                      )}
                    </div>

                    <div>
                      <div className="neighbors-subtitle">
                        Top-{topK} excluding baseline Top-{topK}
                        {currentCriterionNeighbors.baselineMethodName
                          ? ` (${currentCriterionNeighbors.baselineMethodName})`
                          : ' (no baseline found)'}
                      </div>
                      {currentCriterionNeighbors.baselineMethodName ? (
                        <>
                          {currentCriterionNeighbors.baselineMetricLabel && (
                            <div className="neighbors-baseline-metric">
                              Baseline metric: {currentCriterionNeighbors.baselineMetricLabel}
                            </div>
                          )}
                          <ol className="neighbors-list">
                            {currentCriterionNeighbors.novelNeighbors.map((neighbor) => (
                              <li key={`novel-${neighbor.index}`}>
                                <div className="neighbor-topline">
                                  <span>Doc #{neighbor.index}</span>
                                  <span>{neighbor.score.toFixed(4)}</span>
                                </div>
                                <div className="neighbor-content">
                                  {vizData.thumbnailUrls?.[neighbor.index] && (
                                    <img
                                      src={vizData.thumbnailUrls[neighbor.index] || ''}
                                      alt={`Doc ${neighbor.index}`}
                                      className="neighbor-thumb"
                                    />
                                  )}
                                  <div className="neighbor-text">
                                    {(() => {
                                      const rawText = (vizData.rawDocuments?.[neighbor.index] ?? vizData.documents[neighbor.index]) || '';
                                      const displayText = getDisplayTextForDoc(neighbor.index, neighbor.text);
                                      return (
                                        <>
                                          <div className="neighbor-doc-preview">
                                            {rawText.slice(0, 80)}
                                            {rawText.length > 80 ? '...' : ''}
                                          </div>
                                          {displayText !== rawText && (
                                            <div className="neighbor-summary">
                                              {displayText.slice(0, 120)}
                                              {displayText.length > 120 ? '...' : ''}
                                            </div>
                                          )}
                                        </>
                                      );
                                    })()}
                                  </div>
                                </div>
                              </li>
                            ))}
                          </ol>
                          {currentCriterionNeighbors.novelNeighbors.length === 0 && (
                            <div className="neighbors-card-error">
                              No additional neighbors outside baseline Top-{topK}.
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="neighbors-card-error">
                          No baseline no-criteria/no-instructions method found for this criterion.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
