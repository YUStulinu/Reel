package main

import "math"

// ABR state seen by an algorithm before choosing the next segment's rendition.
type AbrState struct {
	Buffer     float64   `json:"buffer"`     // seconds of video buffered
	Throughput []float64 `json:"throughput"` // measured kbps of recent segments, oldest first
	Last       int       `json:"last"`       // rendition of the previous segment (-1 at start)
}

type Algo func(st AbrState, kbps []float64) int

// highest rendition whose bitrate fits under a budget (0 if none)
func fit(kbps []float64, budget float64) int {
	best := 0
	for i, k := range kbps {
		if k <= budget {
			best = i
		}
	}
	return best
}

func harmonic(xs []float64, n int) float64 {
	if len(xs) == 0 {
		return 0
	}
	if len(xs) > n {
		xs = xs[len(xs)-n:]
	}
	s := 0.0
	for _, x := range xs {
		s += 1 / math.Max(x, 1)
	}
	return float64(len(xs)) / s
}

// Throughput picks the highest rendition under 80% of the harmonic mean of
// the last five segment throughputs.
func Throughput(st AbrState, kbps []float64) int {
	if len(st.Throughput) == 0 {
		return 0
	}
	return fit(kbps, 0.8*harmonic(st.Throughput, 5))
}

// Buffer is BBA-0: the lowest rendition below a 5 s reservoir, the highest
// above 20 s, linear in between.
func Buffer(st AbrState, kbps []float64) int {
	const reservoir, cushion = 5.0, 20.0
	if st.Buffer <= reservoir {
		return 0
	}
	if st.Buffer >= reservoir+cushion {
		return len(kbps) - 1
	}
	budget := kbps[0] + (kbps[len(kbps)-1]-kbps[0])*(st.Buffer-reservoir)/cushion
	return fit(kbps, budget)
}

// Hybrid (the player's algorithm): the throughput rule with a safety factor
// that depends on the buffer, at most one step up per segment, and a drop to
// the lowest rendition when the buffer is nearly empty.
func Hybrid(st AbrState, kbps []float64) int {
	if len(st.Throughput) == 0 {
		return 0
	}
	if st.Buffer < 2 {
		return 0
	}
	safety := 0.7
	if st.Buffer > 10 {
		safety = 0.9
	}
	choice := fit(kbps, safety*harmonic(st.Throughput, 4))
	if st.Last >= 0 && choice > st.Last+1 {
		choice = st.Last + 1
	}
	return choice
}

var Algos = map[string]Algo{"throughput": Throughput, "buffer": Buffer, "hybrid": Hybrid}

// A Trace gives the available bandwidth (kbps) at time t seconds.
type Trace func(t float64) float64

var Traces = map[string]Trace{
	"constant": func(t float64) float64 { return 3000 },
	"step": func(t float64) float64 {
		if math.Mod(t, 60) < 30 {
			return 4000
		}
		return 800
	},
	"wave": func(t float64) float64 { return 2200 + 1800*math.Sin(t/9) },
	"drop": func(t float64) float64 {
		if t > 20 && t < 35 {
			return 300
		}
		return 2500
	},
}

type SimResult struct {
	AvgKbps  float64 `json:"avg_kbps"`
	Rebuffer float64 `json:"rebuffer_s"`
	Switches int     `json:"switches"`
	Startup  float64 `json:"startup_s"`
	Choices  []int   `json:"choices"`
}

// Simulate plays segSizes[rendition][segment] (bytes) of segDur seconds each
// over a bandwidth trace. Playback starts once one segment is buffered; the
// player stops fetching while 30 s are buffered.
func Simulate(algo Algo, trace Trace, kbps []float64, segSizes [][]int, segDur float64) SimResult {
	const maxBuffer = 30.0
	t, buffer := 0.0, 0.0
	playing := false
	res := SimResult{Startup: -1}
	st := AbrState{Last: -1}
	sum := 0.0
	nseg := len(segSizes[0])
	for i := 0; i < nseg; i++ {
		if buffer > maxBuffer-segDur {
			wait := buffer - (maxBuffer - segDur)
			t += wait
			buffer -= wait
		}
		r := algo(st, kbps)
		if st.Last >= 0 && r != st.Last {
			res.Switches++
		}
		res.Choices = append(res.Choices, r)
		// download, integrating the trace in 10 ms steps
		bits := float64(segSizes[r][i]) * 8
		start := t
		for bits > 0 {
			bw := math.Max(trace(t), 1) * 1000
			dt := 0.01
			if bits <= bw*dt {
				// the last slice: finish exactly (subtracting could leave a float residue forever)
				dt = bits / bw
				bits = 0
			} else {
				bits -= bw * dt
			}
			t += dt
			if playing {
				buffer -= dt
				if buffer < 0 {
					res.Rebuffer += -buffer
					buffer = 0
				}
			}
		}
		elapsed := t - start
		st.Throughput = append(st.Throughput, float64(segSizes[r][i])*8/1000/math.Max(elapsed, 1e-6))
		buffer += segDur
		if !playing {
			playing = true
			res.Startup = t
		}
		st.Buffer = buffer
		st.Last = r
		sum += kbps[r]
	}
	res.AvgKbps = sum / float64(nseg)
	return res
}
