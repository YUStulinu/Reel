import { test } from "node:test";
import assert from "node:assert/strict";
import { packets, parseHeader, isKeyframe, HEADER_LEN } from "../src/container.ts";

function stream(types: number[]): Uint8Array {
  const out: number[] = [...new TextEncoder().encode("REEL"), 1, 64, 0, 48, 0, 25, 0, 0, 0, 1, 0, 0, 0];
  types.forEach((t, i) => {
    out.push(5, 0, 0, 0, t, 30, i, i, i);
  });
  return new Uint8Array(out);
}

test("header and packets", () => {
  const s = stream([0, 1, 1]);
  assert.deepEqual(parseHeader(s), { width: 64, height: 48, fpsNum: 25, fpsDen: 1 });
  const p = packets(s.subarray(HEADER_LEN));
  assert.equal(p.length, 3);
  assert.ok(isKeyframe(p[0]) && !isKeyframe(p[1]));
  assert.deepEqual([...p[2]], [1, 30, 2, 2, 2]);
  assert.throws(() => packets(s.subarray(HEADER_LEN, s.length - 1)));
  assert.throws(() => parseHeader(new TextEncoder().encode("NOPE and more bytes")));
});
