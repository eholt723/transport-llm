const q = document.getElementById('q');
const out = document.getElementById('out');
document.getElementById('go').onclick = async () => {
  out.textContent = "Loading model (first run may take a minute)...";
  // TODO: integrate WebLLM or Transformers.js WebGPU loader here.
  // For now, just echo:
  out.textContent = ">> " + q.value + "\n\n(Demo placeholder — model loading to be wired)";
};
