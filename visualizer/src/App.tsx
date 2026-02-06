/**
 * Main App component for Multiview Visualizer.
 */

import React, { useState, useEffect } from 'react';
import { DatasetIndex, VisualizationMode, VisualizationData, TripletLogEntry } from './types/manifest';
import { loadDatasetIndex, loadVisualizationData, loadBenchmarkResults, loadTripletLogs } from './utils/dataLoader';
import { ModeRenderer } from './render/ModeRenderer';
import { Leaderboard, BenchmarkResults } from './components/Leaderboard';
import { TripletList } from './components/TripletList';
import './App.css';

const App: React.FC = () => {
  const [index, setIndex] = useState<DatasetIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selection state
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('');
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

  // Load index on mount
  useEffect(() => {
    loadDatasetIndex()
      .then((idx) => {
        setIndex(idx);

        // Auto-select first available benchmark/dataset/criterion/method/mode
        const benchmarks = Object.keys(idx);
        if (benchmarks.length > 0) {
          const firstBenchmark = benchmarks[0];
          const datasets = Object.keys(idx[firstBenchmark]);
          if (datasets.length > 0) {
            const firstDataset = datasets[0];
            const criteria = Object.keys(idx[firstBenchmark][firstDataset]);
            if (criteria.length > 0) {
              const firstCriterion = criteria[0];
              const methods = Object.keys(idx[firstBenchmark][firstDataset][firstCriterion]);
              if (methods.length > 0) {
                const firstMethod = methods[0];
                const modes = idx[firstBenchmark][firstDataset][firstCriterion][firstMethod].modes;
                if (modes.length > 0) {
                  setSelectedBenchmark(firstBenchmark);
                  setSelectedDataset(firstDataset);
                  setSelectedCriterion(firstCriterion);
                  setSelectedMethod(firstMethod);
                  setSelectedMode(modes[0] as VisualizationMode);

                  // Load results for this benchmark
                  loadBenchmarkResults(firstBenchmark).then(setResults);
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
    if (selectedBenchmark) {
      loadBenchmarkResults(selectedBenchmark).then(setResults);
    }
  }, [selectedBenchmark]);

  // Load visualization data when selection changes
  useEffect(() => {
    if (!index || !selectedBenchmark || !selectedDataset || !selectedCriterion || !selectedMethod || !selectedMode) {
      setVizData(null);
      setTripletLogs(null);
      return;
    }

    const benchmarkData = index[selectedBenchmark];
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
          setVizData(data);
          setLoadingViz(false);
        })
        .catch((err) => {
          setError(`Failed to load visualization: ${err.message}`);
          setLoadingViz(false);
          setVizData(null);
        });
    }
  }, [index, selectedBenchmark, selectedDataset, selectedCriterion, selectedMethod, selectedMode]);

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

  const benchmarks = Object.keys(index);
  const datasets = selectedBenchmark && index[selectedBenchmark]
    ? Object.keys(index[selectedBenchmark])
    : [];
  const criteria = selectedBenchmark && selectedDataset && index[selectedBenchmark]?.[selectedDataset]
    ? Object.keys(index[selectedBenchmark][selectedDataset])
    : [];
  const methods = selectedBenchmark && selectedDataset && selectedCriterion && index[selectedBenchmark]?.[selectedDataset]?.[selectedCriterion]
    ? Object.keys(index[selectedBenchmark][selectedDataset][selectedCriterion])
    : [];
  const modes = selectedBenchmark && selectedDataset && selectedCriterion && selectedMethod && index[selectedBenchmark]?.[selectedDataset]?.[selectedCriterion]?.[selectedMethod]
    ? index[selectedBenchmark][selectedDataset][selectedCriterion][selectedMethod].modes
    : [];

  // Get task name for leaderboard - extract from ANY method's path in this dataset/criterion
  // (Even if selected method doesn't have embeddings, we want to show the leaderboard)
  // Path format: "benchmark_fuzzy_debug2/gsm8k__final_expression__tag__5/method_name"
  const taskName = (() => {
    if (!selectedBenchmark || !selectedDataset || !selectedCriterion) return '';

    const criterionMethods = index[selectedBenchmark]?.[selectedDataset]?.[selectedCriterion];
    if (!criterionMethods) return '';

    // Get task name from first available method
    const firstMethod = Object.values(criterionMethods)[0];
    return firstMethod?.path ? firstMethod.path.split('/')[1] : '';
  })();

  return (
    <div className="app">
      <header className="header">
        <h1>🔬 Multiview Visualizer</h1>
        <div className="controls">
          <div className="control-group">
            <label>Experiment:</label>
            <select
              value={selectedBenchmark}
              onChange={(e) => {
                const newBenchmark = e.target.value;
                setSelectedBenchmark(newBenchmark);
                // Reset downstream selections
                const newDatasets = Object.keys(index[newBenchmark]);
                if (newDatasets.length > 0) {
                  const newDataset = newDatasets[0];
                  setSelectedDataset(newDataset);
                  const newCriteria = Object.keys(index[newBenchmark][newDataset]);
                  if (newCriteria.length > 0) {
                    const newCriterion = newCriteria[0];
                    setSelectedCriterion(newCriterion);
                    const newMethods = Object.keys(index[newBenchmark][newDataset][newCriterion]);
                    if (newMethods.length > 0) {
                      const newMethod = newMethods[0];
                      setSelectedMethod(newMethod);
                      const newModes = index[newBenchmark][newDataset][newCriterion][newMethod].modes;
                      if (newModes.length > 0) {
                        setSelectedMode(newModes[0] as VisualizationMode);
                      }
                    }
                  }
                }
              }}
            >
              {benchmarks.map((benchmark) => (
                <option key={benchmark} value={benchmark}>
                  {benchmark}
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
                const newCriteria = Object.keys(index[selectedBenchmark][e.target.value]);
                if (newCriteria.length > 0) {
                  const newCriterion = newCriteria[0];
                  setSelectedCriterion(newCriterion);
                  const newMethods = Object.keys(index[selectedBenchmark][e.target.value][newCriterion]);
                  if (newMethods.length > 0) {
                    const newMethod = newMethods[0];
                    setSelectedMethod(newMethod);
                    const newModes = index[selectedBenchmark][e.target.value][newCriterion][newMethod].modes;
                    if (newModes.length > 0) {
                      setSelectedMode(newModes[0] as VisualizationMode);
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
                const newMethods = Object.keys(index[selectedBenchmark][selectedDataset][e.target.value]);
                if (newMethods.length > 0) {
                  const newMethod = newMethods[0];
                  setSelectedMethod(newMethod);
                  const newModes = index[selectedBenchmark][selectedDataset][e.target.value][newMethod].modes;
                  if (newModes.length > 0) {
                    setSelectedMode(newModes[0] as VisualizationMode);
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
                const newModes = index[selectedBenchmark][selectedDataset][selectedCriterion][e.target.value].modes;
                if (newModes.length > 0) {
                  setSelectedMode(newModes[0] as VisualizationMode);
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
              {modes.map((mode) => (
                <button
                  key={mode}
                  className={selectedMode === mode ? 'active' : ''}
                  onClick={() => setSelectedMode(mode as VisualizationMode)}
                >
                  {mode.toUpperCase()}
                </button>
              ))}
              {/* Show GRAPH and HEATMAP if not already in modes */}
              {!modes.includes('graph') && (
                <button
                  className={selectedMode === 'graph' ? 'active' : ''}
                  onClick={() => setSelectedMode('graph')}
                >
                  GRAPH
                </button>
              )}
              {!modes.includes('heatmap') && (
                <button
                  className={selectedMode === 'heatmap' ? 'active' : ''}
                  onClick={() => setSelectedMode('heatmap')}
                >
                  HEATMAP
                </button>
              )}
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
      </header>

      <main className="main">
        {/* Leaderboard */}
        <Leaderboard
          results={results}
          currentTask={taskName}
          currentMethod={selectedMethod}
          methodsWithEmbeddings={
            selectedBenchmark && selectedDataset && selectedCriterion
              ? Object.entries(index[selectedBenchmark]?.[selectedDataset]?.[selectedCriterion] || {})
                  .filter(([_, data]) => data.has_embeddings !== false)
                  .map(([name, _]) => name)
              : []
          }
          onMethodSelect={(method) => {
            setSelectedMethod(method);
            // Reset mode for new method (if it has embeddings)
            const methodData = index[selectedBenchmark]?.[selectedDataset]?.[selectedCriterion]?.[method];
            if (methodData && methodData.modes.length > 0) {
              setSelectedMode(methodData.modes[0] as VisualizationMode);
            }
          }}
        />

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
        {tripletLogs && !loadingViz && (
          <div className="visualization">
            <TripletList tripletLogs={tripletLogs} />
          </div>
        )}
        {vizData && !loadingViz && (
          <div className="visualization">
            <div className="viz-info">
              <span>{vizData.documents.length} documents</span>
              <span>{vizData.manifest.embedding_dim}D embeddings</span>
              {vizData.triplets && <span>{vizData.triplets.length} triplets</span>}
            </div>
            <ModeRenderer
              mode={selectedMode}
              data={vizData}
              displayMode={displayMode}
              width={1000}
              height={700}
            />
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
