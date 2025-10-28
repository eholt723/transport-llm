export function l2Normalize(vec: Float32Array): void {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
  for (let i = 0; i < vec.length; i++) vec[i] *= inv;
}

export function cosineSim(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s; // if both normalized, dot = cosine
}

export function topKCosine(
  query: Float32Array,
  matrix: Float32Array,
  dim: number,
  k: number
): { index: number; score: number }[] {
  // matrix is [N*dim]
  const N = Math.floor(matrix.length / dim);
  const acc: { index: number; score: number }[] = [];
  for (let i = 0; i < N; i++) {
    let s = 0;
    const base = i * dim;
    for (let j = 0; j < dim; j++) s += query[j] * matrix[base + j];
    // insert into small ordered list
    if (acc.length < k) {
      acc.push({ index: i, score: s });
      if (acc.length === k) acc.sort((x, y) => y.score - x.score);
    } else if (s > acc[k - 1].score) {
      acc[k - 1] = { index: i, score: s };
      acc.sort((x, y) => y.score - x.score);
    }
  }
  return acc;
}
