/**
 * Leaderboard component - displays benchmark results for all methods.
 */

import React from 'react';
import { BarChart } from './BarChart';

export interface BenchmarkResults {
  [taskName: string]: {
    [methodName: string]: {
      accuracy: number;
      n_correct: number;
      n_total: number;
      n_incorrect?: number;
      n_ties?: number;
      mean_accuracy?: number;
      std_accuracy?: number;
    };
  };
}

interface LeaderboardProps {
  results: BenchmarkResults | null;
  currentTask: string;
  currentMethod: string;
  methodsWithEmbeddings: string[];
  onMethodSelect: (method: string) => void;
}

/**
 * Compute 95% confidence interval for binomial proportion using Wilson score interval.
 * More accurate than normal approximation, especially for small sample sizes.
 */
function computeConfidenceInterval(
  n_correct: number,
  n_total: number
): { lower: number; upper: number; margin: number } {
  if (n_total === 0) {
    return { lower: 0, upper: 0, margin: 0 };
  }

  const p = n_correct / n_total;
  const z = 1.96; // 95% confidence
  const z2 = z * z;

  // Wilson score interval
  const denominator = 1 + z2 / n_total;
  const center = (p + z2 / (2 * n_total)) / denominator;
  const margin =
    (z * Math.sqrt((p * (1 - p)) / n_total + z2 / (4 * n_total * n_total))) /
    denominator;

  return {
    lower: Math.max(0, center - margin),
    upper: Math.min(1, center + margin),
    margin: margin,
  };
}

/**
 * Group methods by base name (e.g., "pseudologit_proposed_trial1" -> "pseudologit_proposed")
 * Returns a map of base names to trial results.
 */
function groupTrialMethods(taskResults: BenchmarkResults[string]): Map<string, Array<{
  method: string;
  accuracy: number;
  n_correct: number;
  n_total: number;
}>> {
  const trialPattern = /^(.+)_trial\d+$/;
  const groups = new Map<string, Array<{
    method: string;
    accuracy: number;
    n_correct: number;
    n_total: number;
  }>>();

  Object.entries(taskResults).forEach(([method, data]) => {
    const match = method.match(trialPattern);
    if (match) {
      const baseName = match[1];
      if (!groups.has(baseName)) {
        groups.set(baseName, []);
      }
      groups.get(baseName)!.push({
        method,
        accuracy: data.accuracy,
        n_correct: data.n_correct,
        n_total: data.n_total,
      });
    }
  });

  // Only keep groups with multiple trials
  const multiTrialGroups = new Map<string, Array<{
    method: string;
    accuracy: number;
    n_correct: number;
    n_total: number;
  }>>();
  groups.forEach((trials, baseName) => {
    if (trials.length > 1) {
      multiTrialGroups.set(baseName, trials);
    }
  });

  return multiTrialGroups;
}

/**
 * Compute aggregate statistics across trials.
 */
function computeAggregateStats(trials: Array<{
  method: string;
  accuracy: number;
  n_correct: number;
  n_total: number;
}>): {
  mean_accuracy: number;
  std_accuracy: number;
  total_correct: number;
  total_n: number;
} {
  const accuracies = trials.map(t => t.accuracy).filter(a => !isNaN(a) && isFinite(a));

  if (accuracies.length === 0) {
    return {
      mean_accuracy: NaN,
      std_accuracy: NaN,
      total_correct: 0,
      total_n: 0,
    };
  }

  const mean = accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length;
  const variance = accuracies.reduce((sum, acc) => sum + Math.pow(acc - mean, 2), 0) / accuracies.length;
  const std = Math.sqrt(variance);

  const total_correct = trials.reduce((sum, t) => sum + t.n_correct, 0);
  const total_n = trials.reduce((sum, t) => sum + t.n_total, 0);

  return {
    mean_accuracy: mean,
    std_accuracy: std,
    total_correct,
    total_n,
  };
}

export const Leaderboard: React.FC<LeaderboardProps> = ({
  results,
  currentTask,
  currentMethod,
  methodsWithEmbeddings,
  onMethodSelect,
}) => {

  if (!results || !results[currentTask]) {
    return (
      <div style={{
        padding: '20px',
        background: '#f9f9f9',
        borderRadius: '8px',
        marginBottom: '20px',
      }}>
        <h3 style={{ marginTop: 0 }}>Leaderboard</h3>
        <p style={{ color: '#666' }}>No results available for this task.</p>
      </div>
    );
  }

  const taskResults = results[currentTask];

  // Group trial methods and compute aggregates
  const trialGroups = groupTrialMethods(taskResults);
  const aggregateRows = new Map<string, {
    method: string;
    accuracy: number;
    n_correct: number;
    n_total: number;
    std_accuracy: number;
    isAggregate: boolean;
    firstTrialMethod: string;
  }>();

  trialGroups.forEach((trials, baseName) => {
    const stats = computeAggregateStats(trials);
    aggregateRows.set(baseName, {
      method: `${baseName} (avg)`,
      accuracy: stats.mean_accuracy,
      n_correct: stats.total_correct,
      n_total: stats.total_n,
      std_accuracy: stats.std_accuracy,
      isAggregate: true,
      firstTrialMethod: trials[0].method,
    });
  });

  // Build methods list with individual methods and aggregates
  const allMethods = Object.entries(taskResults)
    .map(([method, data]) => ({
      method,
      accuracy: data.accuracy,
      n_correct: data.n_correct,
      n_total: data.n_total,
      isAggregate: false,
      std_accuracy: undefined,
      firstTrialMethod: undefined,
    }));

  // Add aggregate rows
  aggregateRows.forEach(aggRow => {
    allMethods.push(aggRow as any);
  });

  // Sort all methods
  const methods = allMethods.sort((a, b) => {
    // Put NaN/invalid values last
    const aIsValid = !isNaN(a.accuracy) && isFinite(a.accuracy);
    const bIsValid = !isNaN(b.accuracy) && isFinite(b.accuracy);

    if (!aIsValid && !bIsValid) return 0;
    if (!aIsValid) return 1;
    if (!bIsValid) return -1;

    return b.accuracy - a.accuracy;
  });

  return (
    <div style={{
      padding: '20px',
      background: '#f9f9f9',
      borderRadius: '8px',
      marginBottom: '20px',
    }}>
      <h3 style={{ margin: 0, marginBottom: '16px' }}>
        📊 Leaderboard: {currentTask.replace('__', ' / ')}
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Table View */}
        <div>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '14px',
            background: '#fff',
            borderRadius: '4px',
            overflow: 'hidden',
          }}>
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600 }}>Rank</th>
                <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600 }}>Method</th>
                <th style={{ padding: '12px', textAlign: 'right', fontWeight: 600 }}>Accuracy</th>
                <th style={{ padding: '12px', textAlign: 'right', fontWeight: 600 }}>Correct</th>
              </tr>
            </thead>
            <tbody>
              {methods.map((row, idx) => {
                const isSelected = row.method === currentMethod;
                const isAggregate = (row as any).isAggregate || false;

                return (
                  <tr
                    key={row.method}
                    onClick={() => !isAggregate && onMethodSelect(row.method)}
                    style={{
                      background: isSelected ? '#e3f2fd' : isAggregate ? '#fff8e1' : idx % 2 === 0 ? '#fff' : '#fafafa',
                      cursor: isAggregate ? 'default' : 'pointer',
                      borderLeft: isSelected ? '3px solid #4a90e2' : isAggregate ? '3px solid #ffa726' : '3px solid transparent',
                      fontStyle: isAggregate ? 'italic' : 'normal',
                      opacity: isAggregate ? 0.8 : 1,
                    }}
                    title={isAggregate ? 'Aggregate rows show average statistics and have no individual embeddings' : ''}
                    onMouseEnter={(e) => {
                      if (!isSelected && !isAggregate) {
                        e.currentTarget.style.background = '#f0f0f0';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.background = isAggregate ? '#fff8e1' : idx % 2 === 0 ? '#fff' : '#fafafa';
                      }
                    }}
                  >
                    <td style={{ padding: '12px', fontWeight: 500 }}>
                      {idx + 1 === 1 && '🥇'}
                      {idx + 1 === 2 && '🥈'}
                      {idx + 1 === 3 && '🥉'}
                      {idx + 1 > 3 && `#${idx + 1}`}
                    </td>
                    <td style={{ padding: '12px', fontWeight: isSelected ? 600 : 400 }}>
                      {row.method}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 600 }}>
                      {isNaN(row.accuracy) || !isFinite(row.accuracy) ? 'N/A' : (
                        <>
                          {`${(row.accuracy * 100).toFixed(1)}%`}
                          {isAggregate && (row as any).std_accuracy !== undefined && (
                            <span style={{ fontSize: '11px', color: '#666', marginLeft: '4px' }}>
                              ± {((row as any).std_accuracy * 100).toFixed(1)}%
                            </span>
                          )}
                        </>
                      )}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right', color: '#666' }}>
                      {row.n_correct} / {row.n_total}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Chart View */}
        <BarChart
          data={methods.map((row, idx) => {
            const isAggregate = (row as any).isAggregate || false;

            // Only show error bars for aggregate rows
            let ci = { lower: 0, upper: 0, margin: 0 };
            if (isAggregate && !isNaN(row.accuracy) && isFinite(row.accuracy)) {
              const std = (row as any).std_accuracy;
              if (std !== undefined) {
                const margin = 1.96 * std;
                ci = {
                  lower: Math.max(0, row.accuracy - margin),
                  upper: Math.min(1, row.accuracy + margin),
                  margin: margin,
                };
              }
            }

            return {
              method: row.method,
              accuracy: row.accuracy,
              n_correct: row.n_correct,
              n_total: row.n_total,
              ci_lower: ci.lower,
              ci_upper: ci.upper,
              rank: idx + 1,
              isSelected: row.method === currentMethod,
              isAggregate: isAggregate,
            };
          })}
          onMethodSelect={(method) => {
            // Don't allow selecting aggregate rows
            const row = methods.find(m => m.method === method);
            if (row && !(row as any).isAggregate) {
              onMethodSelect(method);
            }
          }}
          width={600}
          height={Math.max(300, methods.length * 40 + 100)}
        />
      </div>

      <div style={{
        marginTop: '12px',
        fontSize: '12px',
        color: '#666',
        textAlign: 'center',
      }}>
        Click a row or bar to view that method's results
        {methodsWithEmbeddings.length < methods.length && (
          <span style={{ display: 'block', marginTop: '4px', fontStyle: 'italic' }}>
            Methods marked "(no viz)" show triplet-level results instead of embedding visualizations
          </span>
        )}
      </div>
    </div>
  );
};
