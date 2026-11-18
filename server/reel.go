package main

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// The .reel container (see codec/src/bitstream.rs): a 17-byte header, then
// packets of [u32 length][type][qp][payload]. Type 0 is a keyframe.

const headerLen = 17

type StreamInfo struct {
	Width, Height  int
	FpsNum, FpsDen uint32
}

func (s StreamInfo) FPS() float64 { return float64(s.FpsNum) / float64(s.FpsDen) }

// Segment is a run of packets that starts with a keyframe.
type Segment struct {
	Offset, Length int // byte range of the packets (with their length prefixes)
	Frames         int
}

type Stream struct {
	Info     StreamInfo
	Header   []byte
	Data     []byte
	Segments []Segment
}

func parseHeader(b []byte) (StreamInfo, error) {
	if len(b) < headerLen || string(b[:4]) != "REEL" {
		return StreamInfo{}, errors.New("not a Reel stream")
	}
	if b[4] != 1 {
		return StreamInfo{}, fmt.Errorf("unsupported version %d", b[4])
	}
	s := StreamInfo{
		Width:  int(binary.LittleEndian.Uint16(b[5:])),
		Height: int(binary.LittleEndian.Uint16(b[7:])),
		FpsNum: binary.LittleEndian.Uint32(b[9:]),
		FpsDen: binary.LittleEndian.Uint32(b[13:]),
	}
	if s.Width == 0 || s.Height == 0 || s.FpsDen == 0 {
		return StreamInfo{}, errors.New("bad stream header")
	}
	return s, nil
}

// ParseStream splits a stream into keyframe-aligned segments of at least
// minFrames frames (a segment closes at the first keyframe after that).
func ParseStream(b []byte, minFrames int) (*Stream, error) {
	info, err := parseHeader(b)
	if err != nil {
		return nil, err
	}
	s := &Stream{Info: info, Header: b[:headerLen], Data: b}
	p := headerLen
	cur := Segment{Offset: p}
	for p < len(b) {
		if p+4 > len(b) {
			return nil, errors.New("truncated packet length")
		}
		n := int(binary.LittleEndian.Uint32(b[p:]))
		if n < 2 || p+4+n > len(b) {
			return nil, errors.New("truncated packet")
		}
		key := b[p+4] == 0
		if key && cur.Frames >= minFrames {
			cur.Length = p - cur.Offset
			s.Segments = append(s.Segments, cur)
			cur = Segment{Offset: p}
		}
		if cur.Frames == 0 && !key {
			return nil, fmt.Errorf("segment at byte %d does not start with a keyframe", p)
		}
		cur.Frames++
		p += 4 + n
	}
	if cur.Frames > 0 {
		cur.Length = p - cur.Offset
		s.Segments = append(s.Segments, cur)
	}
	return s, nil
}
