// A thin wrapper over the Reel WebAssembly decoder (player/wasm).

interface Exports {
  memory: WebAssembly.Memory;
  reel_alloc(n: number): number;
  reel_free(p: number, n: number): void;
  reel_open(p: number, n: number): number;
  reel_decode(p: number, n: number): number;
  reel_flush(): number;
  reel_rgba(): number;
  reel_width(): number;
  reel_height(): number;
  reel_checksum(): number;
}

export class ReelDecoder {
  private x: Exports;
  private buf = 0;
  private cap = 0;

  private constructor(x: Exports) {
    this.x = x;
  }

  static async load(bytes: BufferSource): Promise<ReelDecoder> {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new ReelDecoder(instance.exports as unknown as Exports);
  }

  private put(data: Uint8Array): number {
    if (data.length > this.cap) {
      if (this.cap) this.x.reel_free(this.buf, this.cap);
      this.cap = Math.max(data.length, 1 << 16);
      this.buf = this.x.reel_alloc(this.cap);
    }
    // memory may have grown: always take a fresh view
    new Uint8Array(this.x.memory.buffer, this.buf, data.length).set(data);
    return this.buf;
  }

  /** starts a stream (or switches rendition) from a 17-byte header */
  open(header: Uint8Array): void {
    if (this.x.reel_open(this.put(header), header.length) !== 0) throw new Error("bad stream header");
  }

  get width(): number {
    return this.x.reel_width();
  }
  get height(): number {
    return this.x.reel_height();
  }

  private picture(): Uint8ClampedArray {
    return new Uint8ClampedArray(this.x.memory.buffer, this.x.reel_rgba(), this.width * this.height * 4);
  }

  /** decodes one packet; returns the RGBA picture to show now (a view into
   * wasm memory, valid until the next call), or null while the decoder holds
   * a frame back (B frames reorder pictures) */
  decode(packet: Uint8Array): Uint8ClampedArray | null {
    const rc = this.x.reel_decode(this.put(packet), packet.length);
    if (rc < 0) throw new Error(`decode failed (${rc})`);
    return rc === 1 ? this.picture() : null;
  }

  /** the picture held back at the end of a stream or segment, if any */
  flush(): Uint8ClampedArray | null {
    return this.x.reel_flush() === 1 ? this.picture() : null;
  }

  /** FNV-1a of the last shown YUV picture (for tests) */
  checksum(): number {
    return this.x.reel_checksum() >>> 0;
  }
}
