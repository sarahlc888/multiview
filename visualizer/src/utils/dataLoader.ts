/**
 * Data loader for visualization data.
 */

import { MultiviewManifest, VisualizationData, VisualizationMode, DatasetIndex, TripletData, TripletLogEntry } from '../types/manifest';
import { loadNpy, loadDocuments } from './npyLoader';

export async function loadManifest(basePath: string): Promise<MultiviewManifest> {
  const manifestUrl = `/${basePath}/manifest.json`;
  const response = await fetch(manifestUrl);
  if (!response.ok) {
    throw new Error(`Failed to load manifest from ${manifestUrl}: ${response.statusText}`);
  }
  return await response.json();
}

export async function loadVisualizationData(
  basePath: string,
  mode: VisualizationMode
): Promise<VisualizationData> {
  // Load manifest
  const manifest = await loadManifest(basePath);

  // Modes that don't require layouts (coords)
  const noLayoutModes = ['heatmap', 'graph'];

  // Check if mode is available
  // Special cases: dendrogram with image, SOM with grid image
  const hasDendrogramImage = mode === 'dendrogram' && manifest.dendrogram_image;
  const hasSomGridImage = mode === 'som' && manifest.som_grid_image;
  const hasLayout = manifest.layouts[mode];

  if (!noLayoutModes.includes(mode) && !hasLayout && !hasDendrogramImage && !hasSomGridImage) {
    throw new Error(`Mode '${mode}' not available at ${basePath}`);
  }

  // Load documents
  const documentsUrl = `/${basePath}/${manifest.documents_path}`;
  const documents = await loadDocuments(documentsUrl);

  // Load embeddings
  const embeddingsUrl = `/${basePath}/${manifest.embeddings_path}`;
  const embeddings = await loadNpy(embeddingsUrl);

  // Load coordinates for the selected mode (if needed)
  let coords: Float32Array;
  if (noLayoutModes.includes(mode) || hasDendrogramImage) {
    // For modes that use images or don't need coords, use empty array
    coords = new Float32Array(0);
  } else if (hasLayout) {
    try {
      const coordsUrl = `/${basePath}/${manifest.layouts[mode]}`;
      coords = await loadNpy(coordsUrl);
    } catch (error) {
      console.warn(`Failed to load layout for ${mode}, using empty coords:`, error);
      coords = new Float32Array(0);
    }
  } else {
    coords = new Float32Array(0);
  }

  // Load linkage matrix if available (for dendrogram)
  let linkageMatrix: Float32Array | undefined;
  if (manifest.linkage_matrix) {
    const linkageUrl = `/${basePath}/${manifest.linkage_matrix}`;
    linkageMatrix = await loadNpy(linkageUrl);
  }

  // Build thumbnail URLs if available
  let thumbnailUrls: (string | null)[] | undefined;
  if (manifest.thumbnails) {
    thumbnailUrls = manifest.thumbnails.map(thumbPath =>
      thumbPath ? `/${basePath}/${thumbPath}` : null
    );
  }

  // Build dendrogram image URL if available
  let dendrogramImageUrl: string | undefined;
  if (manifest.dendrogram_image) {
    dendrogramImageUrl = `/${basePath}/${manifest.dendrogram_image}`;
  }

  // Build SOM grid image URL if available
  let somGridImageUrl: string | undefined;
  if (manifest.som_grid_image) {
    somGridImageUrl = `/${basePath}/${manifest.som_grid_image}`;
  }

  // Try to load triplets
  const triplets = await loadTriplets(basePath);

  // Try to load triplet logs (for margin/correctness info)
  const tripletLogs = await loadTripletLogs(basePath);

  // Merge triplet logs with triplets if both exist
  let enrichedTriplets = triplets;
  if (triplets && tripletLogs) {
    enrichedTriplets = triplets.map(triplet => {
      const log = tripletLogs.find(l => l.triplet_idx === triplet.triplet_id);
      if (log) {
        return {
          ...triplet,
          positive_score: log.positive_score,
          negative_score: log.negative_score,
          margin: log.positive_score - log.negative_score,
          correct: log.correct,
          is_tie: log.is_tie,
        };
      }
      return triplet;
    });
  }

  return {
    documents,
    embeddings,
    coords,
    linkageMatrix,
    dendrogramImageUrl,
    somGridImageUrl,
    thumbnailUrls,
    manifest,
    triplets: enrichedTriplets,
  };
}

export async function loadDatasetIndex(): Promise<DatasetIndex> {
  const indexUrl = '/index.json';
  const response = await fetch(indexUrl);
  if (!response.ok) {
    throw new Error(`Failed to load dataset index from ${indexUrl}: ${response.statusText}`);
  }
  return await response.json();
}

export async function loadTriplets(basePath: string): Promise<TripletData[] | undefined> {
  // Try to find triplets.json file relative to the base path
  const tripletsUrl = `/${basePath}/triplets.json`;

  try {
    const response = await fetch(tripletsUrl);
    if (response.ok) {
      const triplets = await response.json();
      console.log(`Loaded ${triplets.length} triplets from ${tripletsUrl}`);
      return triplets;
    }
  } catch (e) {
    console.log(`No triplets found at ${tripletsUrl}`);
  }

  return undefined;
}

export async function loadBenchmarkResults(benchmarkRun: string): Promise<any> {
  const resultsUrl = `/${benchmarkRun}/results.json`;

  try {
    const response = await fetch(resultsUrl);
    if (!response.ok) {
      throw new Error(`Failed to load results from ${resultsUrl}: ${response.statusText}`);
    }
    return await response.json();
  } catch (e) {
    console.warn(`No results found at ${resultsUrl}`);
    return null;
  }
}

export async function loadTripletLogs(basePath: string): Promise<TripletLogEntry[] | undefined> {
  const logsUrl = `/${basePath}/triplet_logs.jsonl`;

  try {
    const response = await fetch(logsUrl);
    if (!response.ok) {
      console.log(`No triplet logs found at ${logsUrl}`);
      return undefined;
    }

    const text = await response.text();
    const lines = text.trim().split('\n');
    const logs = lines.map(line => JSON.parse(line));
    console.log(`Loaded ${logs.length} triplet log entries from ${logsUrl}`);
    return logs;
  } catch (e) {
    console.log(`Error loading triplet logs from ${logsUrl}:`, e);
    return undefined;
  }
}
