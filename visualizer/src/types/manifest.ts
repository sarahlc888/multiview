/**
 * Type definitions for multiview visualizer data formats.
 */

export type VisualizationMode = 'tsne' | 'som' | 'dendrogram' | 'umap' | 'pca' | 'graph' | 'heatmap';

export interface MultiviewManifest {
  version: number;
  dataset: string;
  criterion: string;
  method?: string;  // Method name (e.g., 'pseudologit_proposed_trial2', 'qwen3_8b_baseline')
  n_docs: number;
  embedding_dim: number;
  documents_path: string;
  raw_documents_path?: string;
  embeddings_path: string;
  layouts: {
    tsne?: string;
    som?: string;
    dendrogram?: string;
    umap?: string;
    pca?: string;
    graph?: string;
    heatmap?: string;
  };
  linkage_matrix?: string;
  dendrogram_image?: string;  // Path to matplotlib-generated dendrogram PNG
  som_grid_image?: string;    // Path to SOM grid composite PNG
  thumbnails?: (string | null)[];
}

export interface DatasetIndex {
  [benchmarkRun: string]: {
    [dataset: string]: {
      [criterion: string]: {
        [method: string]: {
          modes: string[];
          path: string;
          has_embeddings?: boolean;
        };
      };
    };
  };
}

export interface VisualizationData {
  documents: string[];
  rawDocuments?: string[];
  embeddings: Float32Array;
  coords: Float32Array;
  linkageMatrix?: Float32Array;
  dendrogramImageUrl?: string;  // URL to matplotlib-generated dendrogram image
  somGridImageUrl?: string;      // URL to SOM grid composite image
  thumbnailUrls?: (string | null)[];
  manifest: MultiviewManifest;
  triplets?: TripletData[];
}

export interface TripletData {
  triplet_id: number;
  triplet_idx?: number;
  method_type?: string;
  anchor_id: number;
  positive_id: number;
  negative_id: number;
  anchor: any;
  positive: any;
  negative: any;
  anchor_summary?: string;
  positive_summary?: string;
  negative_summary?: string;
  anchor_summaries?: string[];
  positive_summaries?: string[];
  negative_summaries?: string[];
  quality_assessment?: QualityAssessment;
  quality_assessment_with_annotations?: QualityAssessment;
  quality_assessment_without_annotations?: QualityAssessment;
  // Fields from triplet logs (if available)
  positive_score?: number;
  negative_score?: number;
  margin?: number;
  correct?: boolean;
  is_tie?: boolean;
}

export interface QualityAssessment {
  rating: number;
  label: string;
  class: string;
  reasoning?: {
    text: string;
  };
}

export interface TripletLogEntry {
  triplet_idx: number;
  method_type: string;
  preset: string;
  anchor_id: number;
  positive_id: number;
  negative_id: number;
  anchor: string;
  positive: string;
  negative: string;
  positive_score: number;
  negative_score: number;
  outcome: number;
  correct: boolean;
  is_tie: boolean;
}
