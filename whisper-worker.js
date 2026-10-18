/* In-browser speech recognition worker (Whisper via transformers.js). */
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";

env.allowLocalModels = false;
env.useBrowserCache = true;

const SR = 16000;
let asr = null;
let asrKey = "";
let device = "wasm";

async function hasWebGPU() {
  try {
    if (!("gpu" in navigator) || !navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

async function load(model, preferGpu) {
  const gpu = preferGpu && (await hasWebGPU());
  const wanted = `${model}|${gpu ? "webgpu" : "wasm"}`;
  if (asr && asrKey === wanted) return device;
  if (asr) {
    try {
      await asr.dispose();
    } catch {}
    asr = null;
  }
  const progress_callback = (p) => self.postMessage({ type: "load-progress", data: p });
  if (gpu) {
    try {
      asr = await pipeline("automatic-speech-recognition", model, {
        device: "webgpu",
        dtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
        progress_callback,
      });
      device = "webgpu";
      asrKey = wanted;
      return device;
    } catch (err) {
      self.postMessage({ type: "log", message: "WebGPU failed, falling back to WASM: " + (err && err.message) });
    }
  }
  asr = await pipeline("automatic-speech-recognition", model, {
    device: "wasm",
    dtype: "q8",
    progress_callback,
  });
  device = "wasm";
  asrKey = `${model}|wasm`;
  return device;
}

function rms(arr, start, end) {
  let sum = 0;
  let n = 0;
  for (let i = start; i < end; i += 4) {
    sum += arr[i] * arr[i];
    n++;
  }
  return n ? Math.sqrt(sum / n) : 0;
}

/** Split audio into <=28s windows, cutting at the quietest moment between 18s and 28s. */
function makeWindows(audio) {
  const maxLen = 28 * SR;
  const minLen = 18 * SR;
  const frame = Math.floor(SR * 0.1);
  const windows = [];
  let start = 0;
  while (start < audio.length) {
    let end = Math.min(audio.length, start + maxLen);
    if (end < audio.length) {
      let best = end;
      let bestE = Infinity;
      for (let s = start + minLen; s + frame <= end; s += Math.floor(frame / 2)) {
        const e = rms(audio, s, s + frame);
        if (e < bestE) {
          bestE = e;
          best = s + Math.floor(frame / 2);
        }
      }
      end = best;
    }
    windows.push([start, end]);
    start = end;
  }
  return windows;
}

const JUNK = /^(\s|[.,!?;:…\-–—"'«»()\[\]♪]|\[BLANK_AUDIO\]|\(.*?\))*$/i;

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type !== "transcribe") return;
  const { audio, model, language, preferGpu } = msg;
  try {
    self.postMessage({ type: "status", stage: "loading" });
    const dev = await load(model, preferGpu);
    self.postMessage({ type: "ready", device: dev });

    const windows = makeWindows(audio);
    const total = audio.length;
    let prevText = "";
    let repeatCount = 0;

    for (let w = 0; w < windows.length; w++) {
      const [s, eIdx] = windows[w];
      const offset = s / SR;
      const winEnd = eIdx / SR;
      const segment = audio.slice(s, eIdx);
      const out = [];

      // Skip near-silent windows to avoid hallucinations
      if (rms(segment, 0, segment.length) > 0.004 && segment.length > SR * 0.3) {
        const opts = { task: "transcribe", return_timestamps: true };
        if (language && language !== "auto") opts.language = language;
        const result = await asr(segment, opts);
        const chunks = (result && result.chunks) || [];
        for (const c of chunks) {
          const text = String(c.text || "").replace(/\s+/g, " ").trim();
          if (!text || JUNK.test(text)) continue;
          const ts = c.timestamp || [0, null];
          let a = offset + (Number(ts[0]) || 0);
          let b = ts[1] == null ? winEnd : offset + Number(ts[1]);
          if (b > winEnd) b = winEnd;
          if (a >= winEnd) a = Math.max(offset, winEnd - 1);
          if (b <= a) b = Math.min(winEnd, a + 1.5);
          if (text === prevText) {
            repeatCount++;
            if (repeatCount >= 2) continue;
          } else {
            repeatCount = 0;
          }
          prevText = text;
          out.push({ start: a, end: b, text });
        }
      }
      self.postMessage({ type: "partial", chunks: out, progress: eIdx / total, window: w + 1, windows: windows.length });
    }
    self.postMessage({ type: "complete" });
  } catch (err) {
    self.postMessage({ type: "error", message: (err && err.message) || String(err) });
  }
};
