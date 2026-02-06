/**
 * ModeRenderer - switches between different visualization modes.
 */

import React from 'react';
import { VisualizationMode, VisualizationData } from '../types/manifest';
import { ScatterPlot } from './ScatterPlot';
import { SOMGridView } from './SOMGridView';
import { DendrogramView } from './DendrogramView';
import { ForceDirectedGraph } from './ForceDirectedGraph';
import { HeatmapView } from './HeatmapView';

interface ModeRendererProps {
  mode: VisualizationMode;
  data: VisualizationData;
  displayMode?: 'points' | 'thumbnails';
  width?: number;
  height?: number;
}

export const ModeRenderer: React.FC<ModeRendererProps> = ({
  mode,
  data,
  displayMode = 'thumbnails',
  width = 800,
  height = 600,
}) => {
  switch (mode) {
    case 'tsne':
    case 'umap':
    case 'pca':
      return (
        <ScatterPlot
          coords={data.coords}
          documents={data.documents}
          thumbnailUrls={data.thumbnailUrls}
          triplets={data.triplets}
          displayMode={displayMode}
          width={width}
          height={height}
        />
      );

    case 'som':
      return (
        <SOMGridView
          coords={data.coords}
          documents={data.documents}
          thumbnailUrls={data.thumbnailUrls}
          triplets={data.triplets}
          displayMode={displayMode}
          width={width}
          height={height}
        />
      );

    case 'dendrogram':
      return (
        <DendrogramView
          coords={data.coords}
          documents={data.documents}
          thumbnailUrls={data.thumbnailUrls}
          displayMode={displayMode}
          linkageMatrix={data.linkageMatrix}
          dendrogramImageUrl={data.dendrogramImageUrl}
          width={width}
          height={height}
        />
      );

    case 'graph':
      return (
        <ForceDirectedGraph
          coords={data.coords}
          documents={data.documents}
          embeddings={data.embeddings}
          triplets={data.triplets}
          width={width}
          height={height}
        />
      );

    case 'heatmap':
      return (
        <HeatmapView
          embeddings={data.embeddings}
          documents={data.documents}
          triplets={data.triplets}
          manifest={data.manifest}
          width={width}
          height={height}
        />
      );

    default:
      return (
        <div style={{ padding: '20px', textAlign: 'center' }}>
          <p>Unsupported visualization mode: {mode}</p>
        </div>
      );
  }
};
