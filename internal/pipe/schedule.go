package pipe

// Kind is the task type: forward or backward of one microbatch on one chunk.
type Kind int

const (
	Fwd Kind = iota
	Bwd
)

// Task is one schedulable unit: the forward or backward of microbatch MB on
// virtual pipeline chunk Chunk (covering layers [LayerLo,LayerHi)), executed
// on Device. Activations/gradients only flow between consecutive chunks,
// which sit on adjacent devices (chunk c lives on device c%P).
type Task struct {
	Kind           Kind
	MB, Chunk      int
	Device         int
	LayerLo, LayerHi int
}

// Key uniquely identifies a task.
func (t Task) Key() [3]int { return [3]int{int(t.Kind), t.MB, t.Chunk} }

// Schedule is, per device, the ordered list of tasks that device executes.
type Schedule struct {
	Name string
	Kind string // "gpipe" | "1f1b" | "interleaved"
	P    int    // number of devices (cards)
	M    int // microbatches per step
	V    int // virtual chunks per device (1 unless interleaved)
	S    int // total virtual chunks = P*V
	Layers int // total model layers
	Ranges [][2]int // layer range per chunk, len S
	Queues [][]Task   // per device, in execution order
}

// chunkLayers splits L layers into S chunks as evenly as possible.
func chunkLayers(L, S int) [][2]int {
	ranges := make([][2]int, S)
	base, rem := L/S, L%S
	start := 0
	for c := 0; c < S; c++ {
		n := base
		if c < rem {
			n++
		}
		ranges[c] = [2]int{start, start + n}
		start += n
	}
	return ranges
}

func newSchedule(name, kind string, p, m, v, L int) *Schedule {
	s := &Schedule{Name: name, Kind: kind, P: p, M: m, V: v, S: p * v, Layers: L}
	s.Ranges = chunkLayers(L, s.S)
	s.Queues = make([][]Task, p)
	return s
}

func (s *Schedule) add(kind Kind, mb, chunk int) {
	dev := chunk % s.P
	r := s.Ranges[chunk]
	s.Queues[dev] = append(s.Queues[dev], Task{
		Kind: kind, MB: mb, Chunk: chunk, Device: dev,
		LayerLo: r[0], LayerHi: r[1],
	})
}

// GenGPipe: every device runs all its forwards, then all its backwards.
// Peak activation per device = M, bubble = (P-1)/(M+P-1).
func GenGPipe(p, m, L int) *Schedule {
	s := newSchedule("GPipe(全前向-全反向)", "gpipe", p, m, 1, L)
	for mb := 0; mb < m; mb++ {
		for c := 0; c < s.S; c++ {
			s.add(Fwd, mb, c)
		}
	}
	for mb := 0; mb < m; mb++ {
		for c := s.S - 1; c >= 0; c-- {
			s.add(Bwd, mb, c)
		}
	}
	return s
}

// Gen1F1B: each device warms up with P-1-d forwards, then alternates one
// forward / one backward, then drains. In-flight microbatches on device d
// never exceed P-d (<= P, the stage count). When M < P the alternating
// steady state never pays off and the schedule degenerates to GPipe order,
// which is what the theory formulas assume.
func Gen1F1B(p, m, L int) *Schedule {
	s := newSchedule("1F1B(一前一后)", "1f1b", p, m, 1, L)
	if m < p {
		g := GenGPipe(p, m, L)
		g.Name, g.Kind = s.Name, s.Kind
		return g
	}
	for d := 0; d < p; d++ {
		w := p - 1 - d // warmup forwards
		for k := 0; k < w; k++ {
			s.add(Fwd, k, d)
		}
		j := 0
		for k := w; k < m; k++ {
			s.add(Fwd, k, d)
			s.add(Bwd, j, d)
			j++
		}
		for ; j < m; j++ {
			s.add(Bwd, j, d)
		}
	}
	return s
}

// GenInterleaved: each device owns V non-adjacent chunks (c, c+P, c+2P, ...).
// Per device: forwards chunk-by-chunk (each chunk sweeps all microbatches),
// then backwards in reverse chunk order. The chunk is 1/V the size of a full
// stage, so the pipeline fill/drain shrinks by V: bubble = (P-1)/(M*V+P-1)
// in the device-bound regime. Activations still flow only between
// consecutive virtual chunks.
func GenInterleaved(p, m, v, L int) *Schedule {
	s := newSchedule("交错(interleaved)", "interleaved", p, m, v, L)
	for c := 0; c < s.S; c++ {
		for mb := 0; mb < m; mb++ {
			s.add(Fwd, mb, c)
		}
	}
	for c := s.S - 1; c >= 0; c-- {
		for mb := 0; mb < m; mb++ {
			s.add(Bwd, mb, c)
		}
	}
	return s
}

// TaskRef identifies a task without its device/layer payload.
type TaskRef struct {
	Kind     Kind
	MB, Chunk int
}

// Deps returns the tasks that must finish before t may start:
//   - F(mb,c) needs F(mb,c-1)         (activation flows downstream)
//   - B(mb,c) needs B(mb,c+1)         (gradient flows upstream)
//   - B(mb,S-1) needs F(mb,S-1)       (loss at the last chunk)
func (s *Schedule) Deps(t Task) []TaskRef {
	if t.Kind == Fwd {
		if t.Chunk == 0 {
			return nil
		}
		return []TaskRef{{Fwd, t.MB, t.Chunk - 1}}
	}
	if t.Chunk == s.S-1 {
		return []TaskRef{{Fwd, t.MB, t.Chunk}}
	}
	return []TaskRef{{Bwd, t.MB, t.Chunk + 1}}
}
