// reel-server: adaptive-bitrate streaming of Reel video.
//
//	reel-server serve [-addr :8090] [-media DIR] [-web DIR] [-seg SECONDS]
//	reel-server simulate [-media DIR] [-seg SECONDS]   (ABR algorithms over bandwidth traces)
//
// A stream is a directory media/<name>/ holding one .reel file per
// rendition, all encoded with the same keyframe positions. The server cuts
// them into keyframe-aligned segments and serves a JSON manifest plus the
// segments; `?kbps=N` or `?trace=NAME&sid=ID` on a segment request throttles
// the transfer to emulate a network.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Rendition struct {
	ID       string  `json:"id"`
	Kbps     float64 `json:"kbps"`
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	SegBytes []int   `json:"segment_bytes"`
	stream   *Stream
}

type Manifest struct {
	Name          string      `json:"name"`
	FPS           float64     `json:"fps"`
	SegmentFrames []int       `json:"segment_frames"`
	Renditions    []Rendition `json:"renditions"`
}

// LoadMedia reads every stream directory under dir.
func LoadMedia(dir string, segSeconds float64) (map[string]*Manifest, error) {
	out := map[string]*Manifest{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		m, err := loadStream(filepath.Join(dir, e.Name()), e.Name(), segSeconds)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", e.Name(), err)
		}
		if m != nil {
			out[e.Name()] = m
		}
	}
	return out, nil
}

func loadStream(dir, name string, segSeconds float64) (*Manifest, error) {
	files, _ := filepath.Glob(filepath.Join(dir, "*.reel"))
	if len(files) == 0 {
		return nil, nil
	}
	m := &Manifest{Name: name}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		info, err := parseHeader(b)
		if err != nil {
			return nil, err
		}
		s, err := ParseStream(b, int(math.Round(segSeconds*info.FPS())))
		if err != nil {
			return nil, err
		}
		r := Rendition{ID: strings.TrimSuffix(filepath.Base(f), ".reel"), Width: info.Width, Height: info.Height, stream: s}
		frames, total := 0, 0
		for _, seg := range s.Segments {
			r.SegBytes = append(r.SegBytes, seg.Length)
			frames += seg.Frames
			total += seg.Length
		}
		r.Kbps = float64(total) * 8 / 1000 / (float64(frames) / info.FPS())
		if m.SegmentFrames == nil {
			m.FPS = info.FPS()
			for _, seg := range s.Segments {
				m.SegmentFrames = append(m.SegmentFrames, seg.Frames)
			}
		} else if !sameFrames(m.SegmentFrames, s.Segments) {
			return nil, fmt.Errorf("rendition %s has different keyframe positions", r.ID)
		}
		m.Renditions = append(m.Renditions, r)
	}
	sort.Slice(m.Renditions, func(i, j int) bool { return m.Renditions[i].Kbps < m.Renditions[j].Kbps })
	return m, nil
}

func sameFrames(want []int, segs []Segment) bool {
	if len(want) != len(segs) {
		return false
	}
	for i, s := range segs {
		if s.Frames != want[i] {
			return false
		}
	}
	return true
}

// throttledWrite sends b at about kbps(), in 8 KB slices.
func throttledWrite(w http.ResponseWriter, b []byte, kbps func() float64) error {
	fl, _ := w.(http.Flusher)
	start := time.Now()
	sent := 0.0
	for len(b) > 0 {
		n := min(len(b), 8192)
		// wait until this slice would have arrived at the emulated rate, then send it
		sent += float64(n) * 8 / 1000
		due := time.Duration(sent / math.Max(kbps(), 1) * float64(time.Second))
		if d := due - time.Since(start); d > 0 {
			time.Sleep(d)
		}
		if _, err := w.Write(b[:n]); err != nil {
			return err
		}
		if fl != nil {
			fl.Flush()
		}
		b = b[n:]
	}
	return nil
}

type Server struct {
	media    map[string]*Manifest
	mu       sync.Mutex
	sessions map[string]time.Time
}

func (s *Server) sessionTime(id string) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	t0, ok := s.sessions[id]
	if !ok {
		t0 = time.Now()
		s.sessions[id] = t0
	}
	return time.Since(t0).Seconds()
}

func (s *Server) routes(web string) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/streams", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"streams": keys(s.media), "traces": keys(Traces)})
	})
	mux.HandleFunc("/streams/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/streams/"), "/")
		m := s.media[parts[0]]
		if m == nil {
			http.NotFound(w, r)
			return
		}
		if len(parts) == 2 && parts[1] == "manifest.json" {
			writeJSON(w, m)
			return
		}
		if len(parts) != 3 {
			http.NotFound(w, r)
			return
		}
		var rend *Rendition
		for i := range m.Renditions {
			if m.Renditions[i].ID == parts[1] {
				rend = &m.Renditions[i]
			}
		}
		if rend == nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Cache-Control", "no-store")
		if parts[2] == "header" {
			w.Write(rend.stream.Header)
			return
		}
		i, err := strconv.Atoi(parts[2])
		if err != nil || i < 0 || i >= len(rend.stream.Segments) {
			http.NotFound(w, r)
			return
		}
		seg := rend.stream.Segments[i]
		body := rend.stream.Data[seg.Offset : seg.Offset+seg.Length]
		var rate func() float64
		q := r.URL.Query()
		if k, err := strconv.ParseFloat(q.Get("kbps"), 64); err == nil && k > 0 {
			rate = func() float64 { return k }
		} else if tr, ok := Traces[q.Get("trace")]; ok {
			sid := q.Get("sid")
			rate = func() float64 { return tr(s.sessionTime(sid)) }
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		if rate == nil {
			w.Write(body)
			return
		}
		if err := throttledWrite(w, body, rate); err != nil {
			log.Printf("segment %s/%s/%d: %v", parts[0], parts[1], i, err)
		}
	})
	if web != "" {
		mux.Handle("/", http.FileServer(http.Dir(web)))
	}
	return mux
}

func keys[V any](m map[string]V) []string {
	out := []string{}
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// simulateAll prints every algorithm on every trace for every stream.
func simulateAll(media map[string]*Manifest) {
	for _, name := range keys(media) {
		m := media[name]
		kbps := []float64{}
		sizes := [][]int{}
		for _, r := range m.Renditions {
			kbps = append(kbps, r.Kbps)
			sizes = append(sizes, append([]int(nil), r.SegBytes...))
		}
		// repeat the clip so the traces have time to vary
		for k := range sizes {
			one := sizes[k]
			for len(sizes[k]) < 240 {
				sizes[k] = append(sizes[k], one...)
			}
		}
		dur := float64(m.SegmentFrames[0]) / m.FPS
		fmt.Printf("%s: renditions %v kbps, %d segments of %.2f s\n", name, roundAll(kbps), len(sizes[0]), dur)
		fmt.Printf("  %-9s %-11s %9s %11s %9s %10s\n", "trace", "algorithm", "avg kbps", "rebuffer s", "switches", "startup s")
		for _, tn := range keys(Traces) {
			for _, an := range []string{"throughput", "buffer", "hybrid"} {
				r := Simulate(Algos[an], Traces[tn], kbps, sizes, dur)
				fmt.Printf("  %-9s %-11s %9.0f %11.2f %9d %10.2f\n", tn, an, r.AvgKbps, r.Rebuffer, r.Switches, r.Startup)
			}
		}
	}
}

func roundAll(xs []float64) []int {
	out := make([]int, len(xs))
	for i, x := range xs {
		out[i] = int(math.Round(x))
	}
	return out
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: reel-server serve|simulate [flags]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	addr := fs.String("addr", ":8090", "listen address")
	mediaDir := fs.String("media", "media", "directory of stream directories")
	web := fs.String("web", "../player", "player files to serve (empty: none)")
	seg := fs.Float64("seg", 2, "minimum segment length in seconds")
	fs.Parse(os.Args[2:])
	media, err := LoadMedia(*mediaDir, *seg)
	if err != nil {
		log.Fatal(err)
	}
	switch os.Args[1] {
	case "serve":
		s := &Server{media: media, sessions: map[string]time.Time{}}
		log.Printf("%d streams, listening on %s", len(media), *addr)
		log.Fatal(http.ListenAndServe(*addr, s.routes(*web)))
	case "simulate":
		simulateAll(media)
	default:
		log.Fatalf("unknown command %s", os.Args[1])
	}
}
