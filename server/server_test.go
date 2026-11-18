package main

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// a synthetic stream: types 0 = keyframe, 1 = P; each payload is `size` bytes
func synth(types []byte, size int) []byte {
	b := []byte("REEL")
	b = append(b, 1)
	b = binary.LittleEndian.AppendUint16(b, 64)
	b = binary.LittleEndian.AppendUint16(b, 48)
	b = binary.LittleEndian.AppendUint32(b, 25)
	b = binary.LittleEndian.AppendUint32(b, 1)
	for i, t := range types {
		b = binary.LittleEndian.AppendUint32(b, uint32(size+2))
		b = append(b, t, 30)
		for k := 0; k < size; k++ {
			b = append(b, byte(i))
		}
	}
	return b
}

func TestSegmentsStartAtKeyframes(t *testing.T) {
	s, err := ParseStream(synth([]byte{0, 1, 1, 0, 1, 1, 0, 1, 0}, 10), 3)
	if err != nil {
		t.Fatal(err)
	}
	got := []int{}
	for _, seg := range s.Segments {
		got = append(got, seg.Frames)
		if s.Data[seg.Offset+4] != 0 {
			t.Fatalf("segment at %d starts with type %d", seg.Offset, s.Data[seg.Offset+4])
		}
	}
	// a segment closes at the first keyframe after 3 frames
	if want := []int{3, 3, 3}; len(got) != 3 || got[0] != 3 || got[1] != 3 || got[2] != 3 {
		t.Fatalf("segments %v, want %v", got, want)
	}
	if _, err := ParseStream(synth([]byte{1, 0}, 4), 1); err == nil {
		t.Fatal("a stream starting with a P frame must be rejected")
	}
	if _, err := ParseStream(synth([]byte{0, 1}, 4)[:30], 1); err == nil {
		t.Fatal("a truncated stream must be rejected")
	}
}

func TestServerManifestSegmentsAndThrottling(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "clip"), 0o755)
	types := []byte{0, 1, 1, 1, 0, 1, 1, 1}
	os.WriteFile(filepath.Join(dir, "clip", "low.reel"), synth(types, 1000), 0o644)
	os.WriteFile(filepath.Join(dir, "clip", "high.reel"), synth(types, 6000), 0o644)
	media, err := LoadMedia(dir, 4.0/25)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer((&Server{media: media, sessions: map[string]time.Time{}}).routes(""))
	defer srv.Close()
	var m Manifest
	res, err := srv.Client().Get(srv.URL + "/streams/clip/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	json.NewDecoder(res.Body).Decode(&m)
	if len(m.Renditions) != 2 || m.Renditions[0].ID != "low" || len(m.SegmentFrames) != 2 {
		t.Fatalf("manifest %+v", m)
	}
	get := func(path string) []byte {
		res, err := srv.Client().Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(res.Body)
		return b
	}
	if h := get("/streams/clip/high/header"); len(h) != headerLen || string(h[:4]) != "REEL" {
		t.Fatalf("header %q", h)
	}
	seg := get("/streams/clip/high/1")
	if len(seg) != m.Renditions[1].SegBytes[1] || seg[4] != 0 {
		t.Fatalf("segment 1: %d bytes", len(seg))
	}
	// 4 packets of 6006 bytes = 24 KB at 400 kbps: about 0.48 s
	t0 := time.Now()
	get("/streams/clip/high/0?kbps=400")
	if d := time.Since(t0).Seconds(); d < 0.35 || d > 0.9 {
		t.Fatalf("throttled transfer took %.2f s", d)
	}
}

func TestAbrVectors(t *testing.T) {
	var v struct {
		Kbps  []float64 `json:"kbps"`
		Cases []struct {
			State  AbrState       `json:"state"`
			Expect map[string]int `json:"expect"`
		} `json:"cases"`
	}
	b, err := os.ReadFile("testdata/abr_vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	for i, c := range v.Cases {
		for name, want := range c.Expect {
			if got := Algos[name](c.State, v.Kbps); got != want {
				t.Errorf("case %d %s: got %d, want %d", i, name, got, want)
			}
		}
	}
}

func TestSimulation(t *testing.T) {
	kbps := []float64{200, 400, 800, 1600}
	sizes := make([][]int, 4)
	for r, k := range kbps {
		for i := 0; i < 60; i++ {
			sizes[r] = append(sizes[r], int(k*1000/8*2)) // 2 s segments at the nominal rate
		}
	}
	// plenty of bandwidth: no stalls, ends at the top
	r := Simulate(Hybrid, Traces["constant"], kbps, sizes, 2)
	if r.Rebuffer > 0 || r.Choices[len(r.Choices)-1] != 3 {
		t.Fatalf("constant: %+v", r)
	}
	// a 15 s outage at 300 kbps: the hybrid rule must not stall much longer than a lowest-only player
	low := Simulate(func(AbrState, []float64) int { return 0 }, Traces["drop"], kbps, sizes, 2)
	h := Simulate(Hybrid, Traces["drop"], kbps, sizes, 2)
	if h.Rebuffer > low.Rebuffer+3 {
		t.Fatalf("hybrid rebuffers %.1f s, lowest-only %.1f s", h.Rebuffer, low.Rebuffer)
	}
	if h.AvgKbps <= low.AvgKbps {
		t.Fatalf("hybrid gains nothing: %.0f vs %.0f kbps", h.AvgKbps, low.AvgKbps)
	}
}
