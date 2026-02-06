/**
 * TripletList component - displays triplets with margin and correctness.
 */

import React, { useState } from 'react';
import { TripletLogEntry } from '../types/manifest';

interface TripletListProps {
  tripletLogs: TripletLogEntry[];
}

export const TripletList: React.FC<TripletListProps> = ({ tripletLogs }) => {
  const [sortBy, setSortBy] = useState<'index' | 'margin' | 'correctness'>('index');
  const [filterCorrect, setFilterCorrect] = useState<'all' | 'correct' | 'incorrect'>('all');

  // Calculate margin for each triplet
  const tripletsWithMargin = tripletLogs.map(log => ({
    ...log,
    margin: log.positive_score - log.negative_score,
  }));

  // Filter
  let filtered = tripletsWithMargin;
  if (filterCorrect === 'correct') {
    filtered = filtered.filter(t => t.correct);
  } else if (filterCorrect === 'incorrect') {
    filtered = filtered.filter(t => !t.correct);
  }

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === 'index') {
      return a.triplet_idx - b.triplet_idx;
    } else if (sortBy === 'margin') {
      return Math.abs(b.margin) - Math.abs(a.margin); // Sort by absolute margin (hardest first)
    } else if (sortBy === 'correctness') {
      return (a.correct === b.correct) ? 0 : (a.correct ? 1 : -1); // Incorrect first
    }
    return 0;
  });

  const correctCount = tripletLogs.filter(t => t.correct).length;
  const incorrectCount = tripletLogs.filter(t => !t.correct).length;
  const accuracy = tripletLogs.length > 0 ? (correctCount / tripletLogs.length * 100).toFixed(1) : '0.0';

  return (
    <div style={{
      padding: '20px',
      background: '#fff',
      borderRadius: '8px',
      maxWidth: '1400px',
      margin: '0 auto',
    }}>
      {/* Header */}
      <div style={{
        marginBottom: '20px',
        paddingBottom: '16px',
        borderBottom: '2px solid #e0e0e0',
      }}>
        <h2 style={{ margin: '0 0 12px 0', fontSize: '24px' }}>
          Triplet Evaluation Results
        </h2>
        <div style={{
          display: 'flex',
          gap: '24px',
          fontSize: '14px',
          color: '#666',
        }}>
          <div>
            <strong>Total:</strong> {tripletLogs.length} triplets
          </div>
          <div style={{ color: '#4caf50' }}>
            <strong>Correct:</strong> {correctCount} ({((correctCount / tripletLogs.length) * 100).toFixed(1)}%)
          </div>
          <div style={{ color: '#f44336' }}>
            <strong>Incorrect:</strong> {incorrectCount} ({((incorrectCount / tripletLogs.length) * 100).toFixed(1)}%)
          </div>
          <div>
            <strong>Accuracy:</strong> {accuracy}%
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{
        display: 'flex',
        gap: '16px',
        marginBottom: '20px',
        alignItems: 'center',
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontWeight: 500, fontSize: '14px' }}>Sort by:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            style={{
              padding: '6px 12px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '14px',
            }}
          >
            <option value="index">Triplet Index</option>
            <option value="margin">Margin (Hardest First)</option>
            <option value="correctness">Correctness</option>
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontWeight: 500, fontSize: '14px' }}>Filter:</label>
          <select
            value={filterCorrect}
            onChange={(e) => setFilterCorrect(e.target.value as any)}
            style={{
              padding: '6px 12px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '14px',
            }}
          >
            <option value="all">All Triplets</option>
            <option value="correct">Correct Only</option>
            <option value="incorrect">Incorrect Only</option>
          </select>
        </div>

        <div style={{
          marginLeft: 'auto',
          fontSize: '13px',
          color: '#666',
          fontStyle: 'italic',
        }}>
          Showing {sorted.length} of {tripletLogs.length} triplets
        </div>
      </div>

      {/* Triplet List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {sorted.map((triplet) => (
          <div
            key={triplet.triplet_idx}
            style={{
              border: `2px solid ${triplet.correct ? '#4caf50' : '#f44336'}`,
              borderRadius: '8px',
              padding: '16px',
              background: triplet.correct ? '#f1f8f4' : '#fff5f5',
            }}
          >
            {/* Triplet Header */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '12px',
              paddingBottom: '8px',
              borderBottom: '1px solid #ddd',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{
                  fontWeight: 600,
                  fontSize: '16px',
                  color: '#333',
                }}>
                  Triplet #{triplet.triplet_idx}
                </span>
                <span style={{
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontWeight: 600,
                  background: triplet.correct ? '#4caf50' : '#f44336',
                  color: '#fff',
                }}>
                  {triplet.correct ? '✓ CORRECT' : '✗ INCORRECT'}
                </span>
                {triplet.is_tie && (
                  <span style={{
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 600,
                    background: '#ff9800',
                    color: '#fff',
                  }}>
                    TIE
                  </span>
                )}
              </div>
              <div style={{
                fontSize: '14px',
                color: '#666',
              }}>
                <strong>Margin:</strong> <span style={{
                  fontWeight: 600,
                  color: triplet.margin > 0 ? '#4caf50' : '#f44336',
                  fontFamily: 'monospace',
                }}>
                  {triplet.margin > 0 ? '+' : ''}{triplet.margin.toFixed(4)}
                </span>
                {' '}
                (pos: {triplet.positive_score.toFixed(4)}, neg: {triplet.negative_score.toFixed(4)})
              </div>
            </div>

            {/* Triplet Content */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '12px',
            }}>
              {/* Anchor */}
              <div style={{
                padding: '12px',
                background: '#fff',
                borderRadius: '4px',
                border: '1px solid #ddd',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#666',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  Anchor (ID: {triplet.anchor_id})
                </div>
                <div style={{
                  fontSize: '13px',
                  lineHeight: '1.5',
                  color: '#333',
                  maxHeight: '200px',
                  overflowY: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}>
                  {truncateText(triplet.anchor, 500)}
                </div>
              </div>

              {/* Positive */}
              <div style={{
                padding: '12px',
                background: '#e8f5e9',
                borderRadius: '4px',
                border: '2px solid #4caf50',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#2e7d32',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  Positive (ID: {triplet.positive_id})
                </div>
                <div style={{
                  fontSize: '13px',
                  lineHeight: '1.5',
                  color: '#333',
                  maxHeight: '200px',
                  overflowY: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}>
                  {truncateText(triplet.positive, 500)}
                </div>
              </div>

              {/* Negative */}
              <div style={{
                padding: '12px',
                background: '#ffebee',
                borderRadius: '4px',
                border: '2px solid #f44336',
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#c62828',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  Negative (ID: {triplet.negative_id})
                </div>
                <div style={{
                  fontSize: '13px',
                  lineHeight: '1.5',
                  color: '#333',
                  maxHeight: '200px',
                  overflowY: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}>
                  {truncateText(triplet.negative, 500)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}
