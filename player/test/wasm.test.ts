import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { ReelDecoder } from "../src/wasm.ts";
import { HEADER_LEN, packets, parseHeader } from "../src/container.ts";

// test/fixture.reel and its per-frame checksums from the native decoder
// (scripts/make_fixture.py); reel.wasm comes from `npm run wasm`
const wasm = new URL("../reel.wasm", import.meta.url);

test("the wasm decoder reproduces the native decoder bit for bit", { skip: !existsSync(wasm) && "reel.wasm not built" }, async () => {
  const dec = await ReelDecoder.load(readFileSync(wasm));
  const s = new Uint8Array(readFileSync(new URL("fixture.reel", import.meta.url)));
  const want: number[] = JSON.parse(readFileSync(new URL("fixture.json", import.meta.url), "utf8")).checksums;
  const info = parseHeader(s);
  dec.open(s.subarray(0, HEADER_LEN));
  assert.equal(dec.width, info.width);
  const ps = packets(s.subarray(HEADER_LEN));
  const got: number[] = [];
  for (const p of ps) {
    const rgba = dec.decode(p);
    if (rgba) {
      assert.equal(rgba.length, info.width * info.height * 4);
      got.push(dec.checksum());
    }
  }
  if (dec.flush()) got.push(dec.checksum());
  assert.deepEqual(got, want);
  assert.throws(() => dec.decode(new Uint8Array([7, 30, 1, 2, 3])));
});
