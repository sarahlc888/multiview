/**
 * SOMGridView - Interactive Self-Organizing Map grid visualization.
 * Displays documents as a grid of thumbnails/points with triplet highlighting.
 */

import React, { useState, useMemo } from 'react';
import { TripletData } from '../types/manifest';

interface SOMGridViewProps {
  coords: Float32Array;
  documents: string[];
  thumbnailUrls?: (string | null)[];
  triplets?: TripletData[];
  displayMode?: 'points' | 'thumbnails';
  width?: number;
  height?: number;
  onSelectDocument?: (index: number | null) => void;
}

export const SOMGridView: React.FC<SOMGridViewProps> = ({
  coords,
  documents,
  thumbnailUrls,
  triplets,
  displayMode = 'thumbnails',
  width = 800,
  height = 600,
  onSelectDocument,
}) => {
  const [selectedPointIndex, setSelectedPointIndex] = useState<number | null>(null);
  const [selectedTripletIndex, setSelectedTripletIndex] = useState<number>(0);
  const [showTriplets, setShowTriplets] = useState<boolean>(true);
  const [hoveredTripletRole, setHoveredTripletRole] = useState<'anchor' | 'positive' | 'negative' | null>(null);
  const [hoveredCellIndex, setHoveredCellIndex] = useState<number | null>(null);

  const hasThumbnailsAvailable = thumbnailUrls && thumbnailUrls.some(url => url !== null);
  const useThumbnails = displayMode === 'thumbnails' && hasThumbnailsAvailable;
  const hasTriplets = triplets && triplets.length > 0;

  // Parse grid positions and determine grid size
  const { gridData, gridRows, gridCols } = useMemo(() => {
    const positions: Array<{ row: number; col: number; index: number }> = [];
    let maxRow = 0;
    let maxCol = 0;

    console.log('SOMGridView: coords length =', coords.length);

    for (let i = 0; i < coords.length; i += 2) {
      const row = Math.round(coords[i]);
      const col = Math.round(coords[i + 1]);
      positions.push({ row, col, index: i / 2 });
      maxRow = Math.max(maxRow, row);
      maxCol = Math.max(maxCol, col);
    }

    console.log('SOMGridView: gridRows =', maxRow + 1, 'gridCols =', maxCol + 1, 'positions =', positions.length);

    return {
      gridData: positions,
      gridRows: maxRow + 1,
      gridCols: maxCol + 1,
    };
  }, [coords]);

  // Create a map from (row, col) to document index
  const gridMap = useMemo(() => {
    const map = new Map<string, number>();
    gridData.forEach(({ row, col, index }) => {
      map.set(`${row},${col}`, index);
    });
    return map;
  }, [gridData]);

  // Filter triplets by selected point if a point is selected
  let filteredTriplets = triplets || [];
  if (selectedPointIndex !== null && hasTriplets) {
    filteredTriplets = triplets.filter(
      t => t.anchor_id === selectedPointIndex ||
           t.positive_id === selectedPointIndex ||
           t.negative_id === selectedPointIndex
    );
  }

  const currentTriplet = hasTriplets && showTriplets && filteredTriplets.length > 0
    ? filteredTriplets[selectedTripletIndex]
    : null;

  // Get the role of a cell index in the current triplet
  const getCellRole = (cellIndex: number): 'anchor' | 'positive' | 'negative' | null => {
    if (!currentTriplet) return null;
    if (cellIndex === currentTriplet.anchor_id) return 'anchor';
    if (cellIndex === currentTriplet.positive_id) return 'positive';
    if (cellIndex === currentTriplet.negative_id) return 'negative';
    return null;
  };

  // Get border styling for triplet highlighting
  const getTripletBorder = (cellIndex: number): string => {
    const role = getCellRole(cellIndex);
    if (!role) return 'none';

    const colors = {
      anchor: '#0066cc',
      positive: '#00aa00',
      negative: '#cc0000',
    };

    const isHovered = hoveredTripletRole === role;
    const borderWidth = isHovered ? '4px' : '3px';

    return `${borderWidth} solid ${colors[role]}`;
  };

  // Get box shadow for triplet highlighting
  const getTripletBoxShadow = (cellIndex: number): string => {
    const role = getCellRole(cellIndex);
    if (!role) return 'none';

    const colors = {
      anchor: 'rgba(0, 102, 204, 0.5)',
      positive: 'rgba(0, 170, 0, 0.5)',
      negative: 'rgba(204, 0, 0, 0.5)',
    };

    const isHovered = hoveredTripletRole === role;
    return isHovered ? `0 0 12px ${colors[role]}` : `0 0 6px ${colors[role]}`;
  };

  // Get opacity for non-hovered triplet elements
  const getCellOpacity = (cellIndex: number): number => {
    if (!hoveredTripletRole) return 1;
    const role = getCellRole(cellIndex);
    return role === hoveredTripletRole ? 1 : 0.4;
  };

  // Calculate cell size based on grid dimensions and available space
  const cellSize = Math.min(
    Math.floor((width - 40) / gridCols),
    Math.floor((height - 40) / gridRows),
    useThumbnails ? 60 : 30
  );

  // Helper to get document text from triplet element
  const getDocText = (doc: any): string => {
    if (typeof doc === 'string') return doc;
    if (doc && typeof doc === 'object' && doc.text) return doc.text;
    return JSON.stringify(doc);
  };

  // Reset triplet index when selected point changes
  React.useEffect(() => {
    setSelectedTripletIndex(0);
  }, [selectedPointIndex]);

  return (
    <div style={{ display: 'flex', gap: '20px' }}>
      {/* Grid View */}
      <div style={{ position: 'relative' }}>
        {coords.length === 0 || gridRows === 0 || gridCols === 0 ? (
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
              <div style={{ marginBottom: '8px', fontWeight: 500 }}>No SOM grid available</div>
              <div style={{ fontSize: '12px' }}>
                This view requires SOM coordinates.
                <br />
                Coords length: {coords.length}, Grid: {gridRows}x{gridCols}
              </div>
            </div>
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateRows: `repeat(${gridRows}, ${cellSize}px)`,
              gridTemplateColumns: `repeat(${gridCols}, ${cellSize}px)`,
              gap: '2px',
              padding: '20px',
              background: '#fafafa',
              border: '1px solid #ddd',
              borderRadius: '4px',
              overflow: 'auto',
              maxWidth: width,
              maxHeight: height,
            }}
          >
          {Array.from({ length: gridRows }).map((_, row) =>
            Array.from({ length: gridCols }).map((_, col) => {
              const cellIndex = gridMap.get(`${row},${col}`);
              const hasDocument = cellIndex !== undefined;
              const isSelected = cellIndex === selectedPointIndex;
              const isHovered = cellIndex === hoveredCellIndex;
              const role = cellIndex !== undefined ? getCellRole(cellIndex) : null;

              return (
                <div
                  key={`${row}-${col}`}
                  style={{
                    gridRow: row + 1,
                    gridColumn: col + 1,
                    width: cellSize,
                    height: cellSize,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: hasDocument ? (isSelected ? '#e3f2fd' : '#fff') : '#f0f0f0',
                    border: role ? getTripletBorder(cellIndex!) : (hasDocument ? '1px solid #ddd' : '1px solid #e0e0e0'),
                    borderRadius: '4px',
                    cursor: hasDocument ? 'pointer' : 'default',
                    transition: 'all 0.2s',
                    opacity: cellIndex !== undefined ? getCellOpacity(cellIndex) : 1,
                    boxShadow: role ? getTripletBoxShadow(cellIndex!) : (isHovered ? '0 2px 8px rgba(0,0,0,0.1)' : 'none'),
                  }}
                  onClick={() => {
                    if (hasDocument) {
                      const nextIndex = cellIndex === selectedPointIndex ? null : cellIndex!;
                      setSelectedPointIndex(nextIndex);
                      onSelectDocument?.(nextIndex);
                    }
                  }}
                  onMouseEnter={() => {
                    if (hasDocument) {
                      setHoveredCellIndex(cellIndex);
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredCellIndex(null);
                  }}
                >
                  {hasDocument && (
                    <>
                      {useThumbnails && thumbnailUrls?.[cellIndex!] ? (
                        <img
                          src={thumbnailUrls[cellIndex!]!}
                          alt={`Doc ${cellIndex}`}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            borderRadius: '2px',
                          }}
                        />
                      ) : (
                        <div
                          style={{
                            width: '70%',
                            height: '70%',
                            borderRadius: '50%',
                            background: isSelected ? '#4a90e2' : '#4a90e2',
                            opacity: isSelected ? 1 : 0.7,
                          }}
                        />
                      )}
                    </>
                  )}
                </div>
              );
            })
          )}
        </div>
        )}

        {/* Tooltip */}
        {hoveredCellIndex !== null && (
          <div
            style={{
              position: 'fixed',
              left: '50%',
              top: 20,
              transform: 'translateX(-50%)',
              background: 'rgba(0, 0, 0, 0.85)',
              color: 'white',
              padding: '8px 12px',
              borderRadius: '4px',
              fontSize: '12px',
              maxWidth: '400px',
              pointerEvents: 'none',
              zIndex: 1000,
              whiteSpace: 'pre-wrap',
            }}
          >
            {documents[hoveredCellIndex].slice(0, 200)}
            {documents[hoveredCellIndex].length > 200 ? '...' : ''}
          </div>
        )}
      </div>

      {/* Triplet Sidebar (same as ScatterPlot) */}
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
              {filteredTriplets.length > 0 && (
                <div style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                  <strong>{filteredTriplets.length}</strong> triplet{filteredTriplets.length !== 1 ? 's' : ''} involving this point
                </div>
              )}
            </div>
          )}

          {showTriplets && filteredTriplets.length === 0 && selectedPointIndex !== null && (
            <div style={{
              padding: '20px',
              textAlign: 'center',
              color: '#999',
              fontSize: '13px',
            }}>
              No triplets involve this point
            </div>
          )}

          {showTriplets && filteredTriplets.length > 0 && (
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
                  {selectedTripletIndex + 1} / {filteredTriplets.length}
                </span>
                <button
                  onClick={() => setSelectedTripletIndex(Math.min(filteredTriplets.length - 1, selectedTripletIndex + 1))}
                  disabled={selectedTripletIndex === filteredTriplets.length - 1}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    background: '#fff',
                    cursor: selectedTripletIndex === filteredTriplets.length - 1 ? 'not-allowed' : 'pointer',
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
