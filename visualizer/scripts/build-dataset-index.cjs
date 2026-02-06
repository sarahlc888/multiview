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

  // New structure: outputs/viz/{benchmark}/{task}/{method}/manifest.json
  // Index structure: index[benchmark][dataset][criterion][method]

  const benchmarks = fs.readdirSync(VIZ_DIR).filter(name => {
    // Skip special directories that aren't benchmarks
    if (name === 'assets' || name.startsWith('.')) {
      return false;
    }
    const fullPath = path.join(VIZ_DIR, name);
    return fs.statSync(fullPath).isDirectory();
  });

  for (const benchmark of benchmarks) {
    const benchmarkPath = path.join(VIZ_DIR, benchmark);
    const tasks = fs.readdirSync(benchmarkPath).filter(name => {
      const fullPath = path.join(benchmarkPath, name);
      return fs.statSync(fullPath).isDirectory() && !name.endsWith('.json');
    });

    if (!index[benchmark]) {
      index[benchmark] = {};
    }

    for (const task of tasks) {
      const taskPath = path.join(benchmarkPath, task);
      const methods = fs.readdirSync(taskPath).filter(name => {
        const fullPath = path.join(taskPath, name);
        return fs.statSync(fullPath).isDirectory();
      });

      for (const method of methods) {
        const methodPath = path.join(taskPath, method);
        const manifestPath = path.join(methodPath, 'manifest.json');

        if (!fs.existsSync(manifestPath)) {
          continue;
        }

        try {
          const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
          const modes = Object.keys(manifest.layouts || {});
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
            path: `${benchmark}/${task}/${method}`,
          };

          console.log(`✓ Found ${benchmark}/${dataset}/${criterion}/${method}: ${modes.join(', ')}`);
        } catch (err) {
          console.error(`Error reading manifest in ${methodPath}:`, err.message);
        }
      }
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
