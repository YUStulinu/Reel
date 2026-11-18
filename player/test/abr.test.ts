import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { rules, harmonic, fit } from "../src/abr.ts";

// the same vectors as server/abr_test (Go): both implementations must agree
const v = JSON.parse(readFileSync(new URL("../../server/testdata/abr_vectors.json", import.meta.url), "utf8"));

test("ABR rules match the shared vectors", () => {
  v.cases.forEach((c: any, i: number) => {
    for (const [name, want] of Object.entries(c.expect)) {
      assert.equal(rules[name as keyof typeof rules](c.state, v.kbps), want, `case ${i} ${name}`);
    }
  });
});

test("helpers", () => {
  assert.equal(harmonic([100, 400], 5), 160);
  assert.equal(harmonic([1, 100, 400], 2), 160);
  assert.equal(fit([200, 400, 800], 399), 0);
  assert.equal(fit([200, 400, 800], 10000), 2);
});
