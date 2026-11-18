// The Reel browser player: fetches keyframe-aligned segments from the Go
// server, picks a rendition per segment with the hybrid ABR rule, decodes
// with the WebAssembly decoder and paints frames on a canvas at the clip's
// frame rate.
import { hybridRule, type AbrState } from "./abr.js";
import { packets } from "./container.js";
import { ReelDecoder } from "./wasm.js";

interface Rendition {
  id: string;
  kbps: number;
  width: number;
  height: number;
  segment_bytes: number[];
}
interface Manifest {
  name: string;
  fps: number;
  segment_frames: number[];
  renditions: Rendition[];
}
interface Queued {
  rendition: number;
  header: Uint8Array;
  packets: Uint8Array[];
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("screen");
const ctx = canvas.getContext("2d")!;
const log = (s: string) => ($("status").textContent = s);

let decoder: ReelDecoder;
let session = 0; // bumps on every (re)start so stale loops stop
const MAX_BUFFER = 20; // seconds

async function init() {
  decoder = await ReelDecoder.load(await (await fetch("reel.wasm")).arrayBuffer());
  const { streams, traces } = await (await fetch("/api/streams")).json();
  $<HTMLSelectElement>("stream").innerHTML = streams.map((s: string) => `<option>${s}</option>`).join("");
  $<HTMLSelectElement>("network").innerHTML =
    `<option value="">unlimited</option>` +
    [300, 600, 1200, 2500].map((k) => `<option value="kbps=${k}">${k} kbps</option>`).join("") +
    traces.map((t: string) => `<option value="trace=${t}">trace: ${t}</option>`).join("");
  $("play").addEventListener("click", () => play());
  log(streams.length ? "ready" : "no streams on the server (see README: scripts/ladder.py)");
}

async function play() {
  const me = ++session;
  const name = $<HTMLSelectElement>("stream").value;
  const net = $<HTMLSelectElement>("network").value;
  const sid = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const m: Manifest = await (await fetch(`/streams/${name}/manifest.json`)).json();
  const kbps = m.renditions.map((r) => r.kbps);
  canvas.width = m.renditions[0].width;
  canvas.height = m.renditions[0].height;
  const image = ctx.createImageData(canvas.width, canvas.height);
  const headers = new Map<number, Uint8Array>();
  const queue: Queued[] = [];
  const st: AbrState = { buffer: 0, throughput: [], last: -1 };
  const history: { t: number; kbps: number }[] = [];
  const stats = { stalls: 0, stallSeconds: 0, decodeMs: 0, frames: 0, downloaded: 0 };
  let queuedFrames = 0;
  let fetching = true;
  const t0 = performance.now();

  // downloader
  (async () => {
    for (let i = 0; i < m.segment_frames.length && me === session; i++) {
      while (queuedFrames / m.fps > MAX_BUFFER && me === session) await sleep(100);
      st.buffer = queuedFrames / m.fps;
      const r = hybridRule(st, kbps);
      if (!headers.has(r)) {
        headers.set(r, new Uint8Array(await (await fetch(`/streams/${name}/${m.renditions[r].id}/header`)).arrayBuffer()));
      }
      const q = net ? `?${net}&sid=${sid}` : "";
      const start = performance.now();
      const body = new Uint8Array(await (await fetch(`/streams/${name}/${m.renditions[r].id}/${i}${q}`)).arrayBuffer());
      const secs = Math.max((performance.now() - start) / 1000, 1e-3);
      st.throughput.push((body.length * 8) / 1000 / secs);
      st.last = r;
      stats.downloaded += body.length;
      queue.push({ rendition: r, header: headers.get(r)!, packets: packets(body) });
      queuedFrames += m.segment_frames[i];
      history.push({ t: (performance.now() - t0) / 1000, kbps: kbps[r] });
    }
    fetching = false;
  })();

  // playback at the clip's frame rate; packets are in decode order, so the
  // decoder may hold a picture back (B frames) and releases it on flush at
  // the end of each segment
  let current = -1;
  let seg: Queued | undefined;
  let k = 0;
  let flushed = false;
  let next = performance.now();
  let stalledAt = 0;
  const nextPicture = (): Uint8ClampedArray | null | undefined => {
    // a picture, null if the queue ran dry, undefined at the end
    for (;;) {
      if (seg && k < seg.packets.length) {
        const d0 = performance.now();
        const pic = decoder.decode(seg.packets[k++]);
        stats.decodeMs += performance.now() - d0;
        if (pic) return pic;
        continue;
      }
      if (seg && !flushed) {
        flushed = true;
        const pic = decoder.flush();
        if (pic) return pic;
      }
      seg = queue.shift();
      k = 0;
      flushed = false;
      if (!seg) return fetching ? null : undefined;
      if (seg.rendition !== current) {
        decoder.open(seg.header); // segments start with keyframes: switching is clean
        current = seg.rendition;
      }
    }
  };
  const frame = () => {
    if (me !== session) return;
    const now = performance.now();
    if (now >= next) {
      const pic = nextPicture();
      if (pic === undefined) {
        log(`finished: ${stats.frames} frames, ${stats.stalls} stalls (${stats.stallSeconds.toFixed(1)} s)`);
        return;
      }
      if (pic === null) {
        if (!stalledAt && stats.frames > 0) {
          stalledAt = now;
          stats.stalls++;
        }
        requestAnimationFrame(frame);
        return;
      }
      if (stalledAt) {
        stats.stallSeconds += (now - stalledAt) / 1000;
        stalledAt = 0;
        next = now;
      }
      image.data.set(pic);
      ctx.putImageData(image, 0, 0);
      stats.frames++;
      queuedFrames--;
      next += 1000 / m.fps;
      if (now - next > 250) next = now; // fell far behind: resynchronize
      drawStats(m, current, queuedFrames / m.fps, st, stats, history);
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function drawStats(m: Manifest, r: number, buffer: number, st: AbrState, s: { stalls: number; stallSeconds: number; decodeMs: number; frames: number }, history: { t: number; kbps: number }[]) {
  const tp = st.throughput.length ? st.throughput[st.throughput.length - 1] : 0;
  $("stats").innerHTML = [
    ["rendition", r >= 0 ? `${m.renditions[r].id} (${m.renditions[r].kbps.toFixed(0)} kbps)` : "–"],
    ["buffer", `${buffer.toFixed(1)} s`],
    ["last throughput", `${tp.toFixed(0)} kbps`],
    ["decode", `${(s.decodeMs / Math.max(s.frames, 1)).toFixed(2)} ms/frame`],
    ["stalls", `${s.stalls} (${s.stallSeconds.toFixed(1)} s)`],
  ]
    .map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`)
    .join("");
  const c = $<HTMLCanvasElement>("chart");
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, c.width, c.height);
  const top = Math.max(...m.renditions.map((x) => x.kbps)) * 1.1;
  const span = Math.max(10, ...history.map((h) => h.t));
  g.strokeStyle = getComputedStyle(document.body).getPropertyValue("--accent");
  g.lineWidth = 2;
  g.beginPath();
  history.forEach((h, i) => {
    const x = (h.t / span) * c.width;
    const y = c.height - (h.kbps / top) * c.height;
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  });
  g.stroke();
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

init().catch((e) => log(`error: ${e}`));
