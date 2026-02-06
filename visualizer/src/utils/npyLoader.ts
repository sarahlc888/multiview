/**
 * Utilities for loading NumPy .npy files in the browser.
 *
 * Supports loading .npy files and converting them to TypedArrays.
 */

interface NpyHeader {
  dtype: string;
  fortran_order: boolean;
  shape: number[];
}

function parseNpyHeader(headerBytes: Uint8Array): NpyHeader {
  // Convert bytes to string
  const headerStr = new TextDecoder().decode(headerBytes);

  // Parse Python dict format (e.g., "{'descr': '<f4', 'fortran_order': False, 'shape': (100, 2)}")
  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);

  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error('Failed to parse .npy header');
  }

  const dtype = descrMatch[1];
  const fortran_order = fortranMatch[1] === 'True';
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => parseInt(s.trim()))
    .filter((n) => !isNaN(n));

  return { dtype, fortran_order, shape };
}

function getDtypeInfo(dtype: string): {
  constructor: Float32ArrayConstructor | Float64ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor;
  byteSize: number;
} {
  // Parse dtype string (e.g., '<f4', '>i4', '|u1')
  const typeChar = dtype[dtype.length - 2];
  const sizeChar = dtype[dtype.length - 1];
  const byteSize = parseInt(sizeChar);

  if (typeChar === 'f') {
    if (byteSize === 4) {
      return { constructor: Float32Array, byteSize: 4 };
    } else if (byteSize === 8) {
      return { constructor: Float64Array, byteSize: 8 };
    }
  } else if (typeChar === 'i') {
    if (byteSize === 4) {
      return { constructor: Int32Array, byteSize: 4 };
    }
  } else if (typeChar === 'u') {
    if (byteSize === 1) {
      return { constructor: Uint8Array, byteSize: 1 };
    }
  }

  throw new Error(`Unsupported dtype: ${dtype}`);
}

export async function loadNpy(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // Check magic string (first 6 bytes should be '\x93NUMPY')
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== 'NUMPY') {
    throw new Error('Invalid .npy file: bad magic number');
  }

  // Get version (bytes 6-7)
  const majorVersion = bytes[6];
  const minorVersion = bytes[7];

  if (majorVersion !== 1 && majorVersion !== 2) {
    throw new Error(`Unsupported .npy version: ${majorVersion}.${minorVersion}`);
  }

  // Get header length
  let headerLength: number;
  let dataOffset: number;

  if (majorVersion === 1) {
    // Version 1: 2-byte little-endian header length
    headerLength = bytes[8] | (bytes[9] << 8);
    dataOffset = 10 + headerLength;
  } else {
    // Version 2: 4-byte little-endian header length
    headerLength = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    dataOffset = 12 + headerLength;
  }

  // Parse header
  const headerBytes = bytes.slice(dataOffset - headerLength, dataOffset);
  const header = parseNpyHeader(headerBytes);

  // Get dtype info
  const dtypeInfo = getDtypeInfo(header.dtype);

  // Extract data
  const dataBytes = bytes.slice(dataOffset);
  const data = new dtypeInfo.constructor(
    dataBytes.buffer,
    dataBytes.byteOffset,
    dataBytes.byteLength / dtypeInfo.byteSize
  );

  // Convert to Float32Array if needed
  if (data instanceof Float32Array) {
    return data;
  } else {
    return new Float32Array(data);
  }
}

export async function loadDocuments(url: string): Promise<string[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.statusText}`);
  }

  const text = await response.text();
  return text
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => line.replace(/\\n/g, '\n')); // Unescape newlines
}
