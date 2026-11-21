package pipe

import "sort"

// TrainConfig describes one training run.
type TrainConfig struct {
	Dims      []int // network dims, len = layers+1
	M         int   // microbatches per step
	Steps     int
	LR        float64
	ModelSeed int64
	DataSeed  int64
}

// genData deterministically generates the same microbatch sequence for every
// trainer: xs[step][mb], ts[step][mb].
func genData(seed int64, steps, m, din, dout int) (xs, ts [][][]float64) {
	r := &rng64{s: uint64(seed)}
	xs = make([][][]float64, steps)
	ts = make([][][]float64, steps)
	for s := 0; s < steps; s++ {
		xs[s] = make([][]float64, m)
		ts[s] = make([][]float64, m)
		for mb := 0; mb < m; mb++ {
			x := make([]float64, din)
			for i := range x {
				x[i] = r.uniform()*2 - 1
			}
			t := make([]float64, dout)
			for i := range t {
				t[i] = r.uniform()*2 - 1
			}
			xs[s][mb] = x
			ts[s][mb] = t
		}
	}
	return
}

// TrainSingleCard is the reference: plain single-card training, gradients
// accumulated microbatch by microbatch in index order.
func TrainSingleCard(cfg TrainConfig) *MLP {
	model := NewMLP(cfg.ModelSeed, cfg.Dims)
	L := len(model.Layers)
	xs, ts := genData(cfg.DataSeed, cfg.Steps, cfg.M, cfg.Dims[0], cfg.Dims[L])
	for step := 0; step < cfg.Steps; step++ {
		acc := NewAccumulator(model)
		for mb := 0; mb < cfg.M; mb++ {
			out, caches := model.forwardRange(xs[step][mb], 0, L)
			_, dY := mseLossGrad(out, ts[step][mb])
			_, grads := model.backwardRange(caches, dY, 0)
			for l := 0; l < L; l++ {
				acc.Add(l, grads[l])
			}
		}
		acc.Update(model, cfg.LR, cfg.M)
	}
	return model
}

// TrainPipeline executes one schedule's timeline step by step. The timing
// comes from the simulator; the numerics are identical to the single-card
// run because every microbatch's forward/backward performs the same float
// ops and gradients are accumulated in fixed microbatch order at step end.
func TrainPipeline(cfg TrainConfig, sched *Schedule, res *SimResult) *MLP {
	model := NewMLP(cfg.ModelSeed, cfg.Dims)
	L := len(model.Layers)
	S := sched.S
	xs, ts := genData(cfg.DataSeed, cfg.Steps, cfg.M, cfg.Dims[0], cfg.Dims[L])

	// Execute events in an order that respects data dependencies even when
	// zero-cost chunks create start-time ties.
	events := make([]Event, len(res.Events))
	copy(events, res.Events)
	sort.Slice(events, func(i, j int) bool {
		a, b := events[i].Task, events[j].Task
		if events[i].Start != events[j].Start {
			return events[i].Start < events[j].Start
		}
		if a.Kind != b.Kind {
			return a.Kind == Fwd
		}
		if a.Kind == Fwd {
			return a.Chunk < b.Chunk
		}
		return a.Chunk > b.Chunk
	})

	type mbChunk struct{ mb, chunk int }
	for step := 0; step < cfg.Steps; step++ {
		acts := make(map[mbChunk][]float64)   // chunk outputs
		caches := make(map[mbChunk][]layerCache)
		down := make(map[mbChunk][]float64)   // grad wrt chunk output
		mbGrads := make([][]LayerGrad, cfg.M) // [mb][layer]
		for mb := range mbGrads {
			mbGrads[mb] = make([]LayerGrad, L)
		}
		for _, e := range events {
			t := e.Task
			key := mbChunk{t.MB, t.Chunk}
			if t.Kind == Fwd {
				var in []float64
				if t.Chunk == 0 {
					in = xs[step][t.MB]
				} else {
					in = acts[mbChunk{t.MB, t.Chunk - 1}]
				}
				out, c := model.forwardRange(in, t.LayerLo, t.LayerHi)
				acts[key] = out
				caches[key] = c
			} else {
				var dY []float64
				if t.Chunk == S-1 {
					_, dY = mseLossGrad(acts[key], ts[step][t.MB])
				} else {
					dY = down[mbChunk{t.MB, t.Chunk + 1}]
				}
				dX, grads := model.backwardRange(caches[key], dY, t.LayerLo)
				for i, g := range grads {
					mbGrads[t.MB][t.LayerLo+i] = g
				}
				if t.Chunk > 0 {
					down[key] = dX
				}
			}
		}
		// Fixed microbatch order: this is what makes the update bit-for-bit
		// identical to single-card training.
		acc := NewAccumulator(model)
		for mb := 0; mb < cfg.M; mb++ {
			for l := 0; l < L; l++ {
				acc.Add(l, mbGrads[mb][l])
			}
		}
		acc.Update(model, cfg.LR, cfg.M)
	}
	return model
}
