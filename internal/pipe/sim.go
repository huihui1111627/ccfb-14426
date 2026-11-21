package pipe

import (
	"fmt"
	"math"
	"sort"
)

// SimConfig holds the hardware model: per-device compute speed (a divisor of
// the base cost, so 0.5 = half-speed card) and the transfer latency between
// neighbouring devices.
type SimConfig struct {
	Speed   []float64 // per device, length P
	Latency float64   // per inter-device transfer
	FwdCost float64   // base time units per layer, forward
	BwdCost float64   // base time units per layer, backward
}

// DefaultSimConfig returns an idealised config: identical cards, no latency.
func DefaultSimConfig(p int) SimConfig {
	sp := make([]float64, p)
	for i := range sp {
		sp[i] = 1
	}
	return SimConfig{Speed: sp, Latency: 0, FwdCost: 1, BwdCost: 2}
}

// Event is one executed task with its virtual start/end time.
type Event struct {
	Task       Task
	Start, End float64
}

// Transfer is one inter-device activation/gradient hop.
type Transfer struct {
	MB, Chunk      int // chunk of the *producing* task
	FromDev, ToDev int
	Bwd            bool
	Start, End     float64
}

// SimResult is the outcome of one simulated step.
type SimResult struct {
	Events    []Event
	Transfers []Transfer
	Makespan  float64
	Busy      []float64 // busy time per device
	Bubble    float64   // 1 - sum(busy)/(P*makespan)
	Peak      []int     // peak in-flight chunk-activations per device
}

func (cfg SimConfig) duration(t Task) float64 {
	layers := float64(t.LayerHi - t.LayerLo)
	cost := cfg.FwdCost
	if t.Kind == Bwd {
		cost = cfg.BwdCost
	}
	return layers * cost / cfg.Speed[t.Device]
}

// Simulate runs one training step of the schedule against the hardware model
// and returns the full timeline. Every card is simulated by its own
// goroutine: the orchestrator resolves dependencies on a virtual clock and
// dispatches each ready task to its card's goroutine, which "executes" it
// (computes its duration) and reports completion.
func Simulate(s *Schedule, cfg SimConfig) (*SimResult, error) {
	p := s.P
	if len(cfg.Speed) != p {
		return nil, fmt.Errorf("SimConfig.Speed has %d entries, want %d", len(cfg.Speed), p)
	}

	type request struct {
		t     Task
		start float64
	}
	type completion struct {
		t     Task
		start float64
		end   float64
	}
	reqs := make([]chan request, p)
	dones := make([]chan completion, p)
	for d := 0; d < p; d++ {
		reqs[d] = make(chan request)
		dones[d] = make(chan completion)
		go func(d int) { // one goroutine per card
			for r := range reqs[d] {
				dones[d] <- completion{r.t, r.start, r.start + cfg.duration(r.t)}
			}
		}(d)
	}
	defer func() {
		for d := 0; d < p; d++ {
			close(reqs[d])
		}
		// drain not needed: every request is matched by a received completion
	}()

	finish := make(map[[3]int]float64) // task key -> end time
	devFree := make([]float64, p)
	idx := make([]int, p)
	total := 0
	for d := 0; d < p; d++ {
		total += len(s.Queues[d])
	}

	res := &SimResult{Busy: make([]float64, p), Peak: make([]int, p)}

	// readyTime returns the earliest start allowed by dependencies.
	readyTime := func(t Task) (float64, bool) {
		ready := 0.0
		for _, dep := range s.Deps(t) {
			f, ok := finish[[3]int{int(dep.Kind), dep.MB, dep.Chunk}]
			if !ok {
				return 0, false
			}
			depDev := dep.Chunk % p
			if depDev != t.Device {
				f += cfg.Latency
			}
			if f > ready {
				ready = f
			}
		}
		return ready, true
	}

	for remaining := total; remaining > 0; remaining-- {
		// pick the dispatchable task with the earliest possible start
		best, bestStart := -1, math.Inf(1)
		var bestReady float64
		for d := 0; d < p; d++ {
			if idx[d] >= len(s.Queues[d]) {
				continue
			}
			t := s.Queues[d][idx[d]]
			ready, ok := readyTime(t)
			if !ok {
				continue
			}
			st := math.Max(ready, devFree[d])
			if st < bestStart {
				best, bestStart, bestReady = d, st, ready
			}
		}
		if best < 0 {
			return nil, fmt.Errorf("schedule %s deadlocked with %d tasks left", s.Name, remaining)
		}
		t := s.Queues[best][idx[best]]
		_ = bestReady
		reqs[best] <- request{t, bestStart}
		c := <-dones[best]
		res.Events = append(res.Events, Event{Task: t, Start: c.start, End: c.end})
		finish[t.Key()] = c.end
		devFree[best] = c.end
		res.Busy[best] += c.end - c.start
		// record inter-device transfers triggered by this task's deps
		for _, dep := range s.Deps(t) {
			depDev := dep.Chunk % p
			if depDev == t.Device {
				continue
			}
			depEnd := finish[[3]int{int(dep.Kind), dep.MB, dep.Chunk}]
			res.Transfers = append(res.Transfers, Transfer{
				MB: t.MB, Chunk: dep.Chunk, FromDev: depDev, ToDev: t.Device,
				Bwd: t.Kind == Bwd, Start: depEnd, End: depEnd + cfg.Latency,
			})
		}
		idx[best]++
	}

	for _, e := range res.Events {
		if e.End > res.Makespan {
			res.Makespan = e.End
		}
	}
	busySum := 0.0
	for d := 0; d < p; d++ {
		busySum += res.Busy[d]
	}
	if res.Makespan > 0 {
		res.Bubble = 1 - busySum/(float64(p)*res.Makespan)
	}

	// Peak in-flight activations: a forward stores one chunk-activation on its
	// device when it ends; the matching backward frees it when it ends.
	type mark struct {
		t     float64
		dev   int
		delta int
	}
	marks := make([]mark, 0, 2*len(res.Events))
	for _, e := range res.Events {
		if e.Task.Kind == Fwd {
			marks = append(marks, mark{e.End, e.Task.Device, +1})
		} else {
			marks = append(marks, mark{e.End, e.Task.Device, -1})
		}
	}
	// at equal timestamps count stores before frees (a backward that ends at
	// the same instant as a forward still held its activation until then)
	sort.Slice(marks, func(i, j int) bool {
		if marks[i].t != marks[j].t {
			return marks[i].t < marks[j].t
		}
		return marks[i].delta > marks[j].delta
	})
	inflight := make([]int, p)
	for _, mk := range marks {
		inflight[mk.dev] += mk.delta
		if inflight[mk.dev] > res.Peak[mk.dev] {
			res.Peak[mk.dev] = inflight[mk.dev]
		}
	}
	return res, nil
}
