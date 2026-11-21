// Command server runs the pipeline-parallel scheduler web UI.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strconv"
	"strings"

	"pipesched/internal/pipe"
)

type eventJSON struct {
	Dev   int     `json:"dev"`
	Kind  string  `json:"kind"`
	MB    int     `json:"mb"`
	Chunk int     `json:"chunk"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type transferJSON struct {
	From  int     `json:"from"`
	To    int     `json:"to"`
	MB    int     `json:"mb"`
	Chunk int     `json:"chunk"`
	Bwd   bool    `json:"bwd"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type schedJSON struct {
	Name         string         `json:"name"`
	Makespan     float64        `json:"makespan"`
	Bubble       float64        `json:"bubble"`
	BubbleTheory float64        `json:"bubbleTheory"`
	Peak         []int          `json:"peak"`
	PeakTheory   []int          `json:"peakTheory"`
	Events       []eventJSON    `json:"events"`
	Transfers    []transferJSON `json:"transfers"`
}

type runResponse struct {
	P int `json:"p"`
	M int `json:"m"`
	V int `json:"v"`
	S int `json:"s"`

	Schedules []schedJSON `json:"schedules"`
	Ideal     []schedJSON `json:"ideal"` // same schedules on identical cards, no latency
}

func atoi(s string, def int) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return n
	}
	return def
}

func atof(s string, def float64) float64 {
	if f, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil {
		return f
	}
	return def
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func parseSpeeds(s string, p int) []float64 {
	sp := make([]float64, p)
	for i := range sp {
		sp[i] = 1
	}
	for i, part := range strings.Split(s, ",") {
		if i >= p {
			break
		}
		if f, err := strconv.ParseFloat(strings.TrimSpace(part), 64); err == nil && f > 0 {
			sp[i] = f
		}
	}
	return sp
}

func simulateAll(scheds []*pipe.Schedule, cfg pipe.SimConfig) ([]schedJSON, error) {
	out := make([]schedJSON, 0, len(scheds))
	for _, sched := range scheds {
		res, err := pipe.Simulate(sched, cfg)
		if err != nil {
			return nil, err
		}
		bubbleT, peakT := pipe.Theory(sched)
		sj := schedJSON{
			Name: sched.Name, Makespan: res.Makespan, Bubble: res.Bubble,
			BubbleTheory: bubbleT, Peak: res.Peak, PeakTheory: peakT,
		}
		for _, e := range res.Events {
			kind := "F"
			if e.Task.Kind == pipe.Bwd {
				kind = "B"
			}
			sj.Events = append(sj.Events, eventJSON{
				Dev: e.Task.Device, Kind: kind, MB: e.Task.MB, Chunk: e.Task.Chunk,
				Start: e.Start, End: e.End,
			})
		}
		for _, tr := range res.Transfers {
			sj.Transfers = append(sj.Transfers, transferJSON{
				From: tr.FromDev, To: tr.ToDev, MB: tr.MB, Chunk: tr.Chunk,
				Bwd: tr.Bwd, Start: tr.Start, End: tr.End,
			})
		}
		out = append(out, sj)
	}
	return out, nil
}

func runHandler(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	L := clamp(atoi(q.Get("layers"), 12), 1, 64)
	p := clamp(atoi(q.Get("devices"), 4), 1, 12)
	m := clamp(atoi(q.Get("microbatches"), 8), 1, 64)
	v := clamp(atoi(q.Get("vchunks"), 2), 1, 8)
	lat := atof(q.Get("latency"), 0.15)
	fwd := atof(q.Get("fwdcost"), 1)
	bwd := atof(q.Get("bwdcost"), 2)
	speeds := parseSpeeds(q.Get("speeds"), p)

	scheds := []*pipe.Schedule{pipe.GenGPipe(p, m, L), pipe.Gen1F1B(p, m, L), pipe.GenInterleaved(p, m, v, L)}
	cfg := pipe.SimConfig{Speed: speeds, Latency: lat, FwdCost: fwd, BwdCost: bwd}
	ideal := pipe.DefaultSimConfig(p)
	ideal.FwdCost, ideal.BwdCost = fwd, bwd

	resp := runResponse{P: p, M: m, V: v, S: p * v}
	var err error
	if resp.Schedules, err = simulateAll(scheds, cfg); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if resp.Ideal, err = simulateAll(scheds, ideal); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	flag.Parse()
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(indexHTML))
	})
	http.HandleFunc("/api/run", runHandler)
	log.Printf("流水线并行训练调度器: http://%s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
