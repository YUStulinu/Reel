// The .reel container: a 17-byte header, then [u32 length][type][qp][payload] packets.

export interface StreamInfo {
  width: number;
  height: number;
  fpsNum: number;
  fpsDen: number;
}

export const HEADER_LEN = 17;

export function parseHeader(b: Uint8Array): StreamInfo {
  if (b.length < HEADER_LEN || String.fromCharCode(...b.subarray(0, 4)) !== "REEL" || b[4] !== 1) {
    throw new Error("not a Reel stream");
  }
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  return { width: v.getUint16(5, true), height: v.getUint16(7, true), fpsNum: v.getUint32(9, true), fpsDen: v.getUint32(13, true) };
}

/** splits a run of packets (a segment, or a stream body) into packets: type, qp, payload */
export function packets(b: Uint8Array): Uint8Array[] {
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const out: Uint8Array[] = [];
  let p = 0;
  while (p < b.length) {
    if (p + 4 > b.length) throw new Error("truncated packet length");
    const n = v.getUint32(p, true);
    if (n < 2 || p + 4 + n > b.length) throw new Error("truncated packet");
    out.push(b.subarray(p + 4, p + 4 + n));
    p += 4 + n;
  }
  return out;
}

export const isKeyframe = (packet: Uint8Array) => packet[0] === 0;
