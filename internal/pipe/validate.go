package pipe

import (
	"fmt"
	"sort"
)

// Validate checks a simulated timeline against every scheduling rule:
//   - every queued task executed exactly once
//   - per-device order matches the schedule and tasks never overlap
//   - every forward/backward dependency is respected, including the
//     transfer latency when the dependency crosses devices
func Validate(s *Schedule, res *SimResult, cfg SimConfig) error {
	const eps = 1e-9

	// one event per task
	byKey := make(map[[3]int]Event, len(res.Events))
	for _, e := range res.Events {
		k := e.Task.Key()
		if _, dup := byKey[k]; dup {
			return fmt.Errorf("task %+v executed twice", e.Task)
		}
		byKey[k] = e
	}
	total := 0
	for d := 0; d < s.P; d++ {
		total += len(s.Queues[d])
	}
	if len(res.Events) != total {
		return fmt.Errorf("got %d events, schedule has %d tasks", len(res.Events), total)
	}

	// per-device order and non-overlap
	perDev := make([][]Event, s.P)
	for _, e := range res.Events {
		perDev[e.Task.Device] = append(perDev[e.Task.Device], e)
	}
	for d := 0; d < s.P; d++ {
		evs := perDev[d]
		sort.Slice(evs, func(i, j int) bool { return evs[i].Start < evs[j].Start })
		if len(evs) != len(s.Queues[d]) {
			return fmt.Errorf("device %d executed %d tasks, schedule has %d", d, len(evs), len(s.Queues[d]))
		}
		prevEnd := 0.0
		for i, e := range evs {
			want := s.Queues[d][i]
			if e.Task.Key() != want.Key() {
				return fmt.Errorf("device %d task %d: executed %+v, schedule says %+v", d, i, e.Task, want)
			}
			if i > 0 && e.Start < prevEnd-eps {
				return fmt.Errorf("device %d: task %+v starts at %v before previous ends at %v", d, e.Task, e.Start, prevEnd)
			}
			prevEnd = e.End
		}
	}

	// every dependency of every task
	for _, e := range res.Events {
		for _, dep := range s.Deps(e.Task) {
			de, ok := byKey[[3]int{int(dep.Kind), dep.MB, dep.Chunk}]
			if !ok {
				return fmt.Errorf("task %+v: dependency %+v never executed", e.Task, dep)
			}
			ready := de.End
			if dep.Chunk%s.P != e.Task.Device {
				ready += cfg.Latency
			}
			if e.Start < ready-eps {
				return fmt.Errorf("task %+v starts at %v but dependency %+v only ready at %v",
					e.Task, e.Start, dep, ready)
			}
		}
		// a backward also needs its own chunk's forward activations
		if e.Task.Kind == Bwd {
			fe, ok := byKey[[3]int{int(Fwd), e.Task.MB, e.Task.Chunk}]
			if !ok {
				return fmt.Errorf("backward %+v has no matching forward", e.Task)
			}
			if e.Start < fe.End-eps {
				return fmt.Errorf("backward %+v starts at %v before its forward ends at %v",
					e.Task, e.Start, fe.End)
			}
		}
	}
	return nil
}
