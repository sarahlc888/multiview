#!/usr/bin/env node
/**
 * Build dataset index for web visualizer.
 *
 * Scans outputs/viz/ directory for manifest.json files and builds an index
 * of available datasets, criteria, and visualization modes.
 *
 * Output: outputs/viz/index.json
 */

const fs = require('fs');
const path = require('path');

const VIZ_DIR = path.join(__dirname, '../../outputs/viz');
const INDEX_FILE = path.join(VIZ_DIR, 'index.json');

function scanDirectory() {
  const index = {};

  if (!fs.existsSync(VIZ_DIR)) {
    console.log('outputs/viz/ directory does not exist, creating empty index');
    return index;
  }

  // Supports:
  //   - outputs/viz/{benchmark}/{task}/{method}/manifest.json
  //   - outputs/viz/{benchmark}/{namespace}/{task}/{method}/manifest.json
  // Index structure: index[benchmarkKey][dataset][criterion][method]
  // where benchmarkKey may include namespace (e.g. "benchmark_fuzzy_debug2/corpus").
  const manifestPaths = [];

  function collectManifests(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith('.')) {
        continue;
      }
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === 'assets') {
          continue;
        }
        collectManifests(fullPath);
      } else if (entry.isFile() && entry.name === 'manifest.json') {
        manifestPaths.push(fullPath);
      }
    }
  }

  collectManifests(VIZ_DIR);

  for (const manifestPath of manifestPaths) {
    const methodPath = path.dirname(manifestPath);
    const relPath = path.relative(VIZ_DIR, manifestPath);
    const segments = relPath.split(path.sep);

    // Need at least {benchmark}/{task}/{method}/manifest.json
    if (segments.length < 4) {
      continue;
    }

    const method = segments[segments.length - 2];
    const task = segments[segments.length - 3];
    const benchmark = segments.slice(0, segments.length - 3).join('/');
    const basePath = `${benchmark}/${task}/${method}`;

    if (!index[benchmark]) {
      index[benchmark] = {};
    }

    try {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
      const modesSet = new Set(Object.keys(manifest.layouts || {}));
      // Match python index builder behavior for non-layout modes.
      modesSet.add('heatmap');
      modesSet.add('graph');
      if (manifest.dendrogram_image) modesSet.add('dendrogram');
      if (manifest.som_grid_image) modesSet.add('som');
      const modes = Array.from(modesSet).sort();

      const dataset = manifest.dataset;
      const criterion = manifest.criterion;

      if (!index[benchmark][dataset]) {
        index[benchmark][dataset] = {};
      }
      if (!index[benchmark][dataset][criterion]) {
        index[benchmark][dataset][criterion] = {};
      }

      index[benchmark][dataset][criterion][method] = {
        modes: modes,
        path: basePath,
      };

      console.log(`✓ Found ${benchmark}/${dataset}/${criterion}/${method}: ${modes.join(', ')}`);
    } catch (err) {
      console.error(`Error reading manifest in ${methodPath}:`, err.message);
    }
  }

  return index;
}

function main() {
  console.log('Building dataset index...');
  console.log(`Scanning: ${VIZ_DIR}`);
  console.log('');

  const index = scanDirectory();

  // Ensure outputs/viz directory exists
  if (!fs.existsSync(VIZ_DIR)) {
    fs.mkdirSync(VIZ_DIR, { recursive: true });
  }

  // Write index
  fs.writeFileSync(INDEX_FILE, JSON.stringify(index, null, 2));
  console.log('');
  console.log(`✓ Index built: ${INDEX_FILE}`);

  // Print summary
  const benchmarkCount = Object.keys(index).length;
  let totalMethods = 0;
  let totalModes = 0;

  for (const benchmark in index) {
    for (const dataset in index[benchmark]) {
      for (const criterion in index[benchmark][dataset]) {
        const methods = Object.keys(index[benchmark][dataset][criterion]);
        totalMethods += methods.length;
        for (const method of methods) {
          totalModes += index[benchmark][dataset][criterion][method].modes.length;
        }
      }
    }
  }

  console.log('');
  console.log('Summary:');
  console.log(`  Benchmarks: ${benchmarkCount}`);
  console.log(`  Methods: ${totalMethods}`);
  console.log(`  Total visualizations: ${totalModes}`);
}

if (require.main === module) {
  main();
}
