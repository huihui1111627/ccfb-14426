// Package pipe implements a pipeline-parallel training scheduler simulator:
// a deterministic MLP, schedule generators (GPipe / 1F1B / interleaved),
// an event-driven timing simulator and exact gradient-accumulation trainers.
package pipe

import "math"

// Layer is one fully-connected layer: y = act(W x + b).
// The last layer of the model is linear, all others use tanh.
type Layer struct {
	In, Out int
	W       []float64 // row-major, Out x In
	B       []float64 // Out
}

// MLP is a plain multi-layer perceptron standing in for the crack/weathering
// classifier. All arithmetic is plain float64 loops in a fixed order so that
// results are bit-for-bit reproducible.
type MLP struct {
	Layers []Layer
}

// rng64 is a splitmix64 generator so model init and data do not depend on
// the global rand state.
type rng64 struct{ s uint64 }

func (r *rng64) next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *rng64) uniform() float64 { return float64(r.next()>>11) * (1.0 / 9007199254740992.0) }

// NewMLP builds a model with deterministic fixed-seed init:
// W ~ U(-1/sqrt(in), 1/sqrt(in)), B = 0.
func NewMLP(seed int64, dims []int) *MLP {
	r := &rng64{s: uint64(seed)}
	layers := make([]Layer, len(dims)-1)
	for i := range layers {
		in, out := dims[i], dims[i+1]
		scale := 1.0 / math.Sqrt(float64(in))
		W := make([]float64, out*in)
		for j := range W {
			W[j] = (r.uniform()*2 - 1) * scale
		}
		layers[i] = Layer{In: in, Out: out, W: W, B: make([]float64, out)}
	}
	return &MLP{Layers: layers}
}

// Equal reports bit-for-bit equality of all parameters.
func (m *MLP) Equal(o *MLP) bool {
	if len(m.Layers) != len(o.Layers) {
		return false
	}
	for i := range m.Layers {
		a, b := m.Layers[i], o.Layers[i]
		if a.In != b.In || a.Out != b.Out || len(a.W) != len(b.W) || len(a.B) != len(b.B) {
			return false
		}
		for j := range a.W {
			if a.W[j] != b.W[j] {
				return false
			}
		}
		for j := range a.B {
			if a.B[j] != b.B[j] {
				return false
			}
		}
	}
	return true
}

// layerCache holds the per-layer values a backward pass needs.
type layerCache struct {
	x []float64 // layer input
	y []float64 // layer output (post-activation)
}

// forwardRange runs layers [a,b) on x and returns the output plus caches.
// With a==b it is the identity (an empty pipeline chunk).
func (m *MLP) forwardRange(x []float64, a, b int) ([]float64, []layerCache) {
	caches := make([]layerCache, b-a)
	cur := x
	last := len(m.Layers) - 1
	for i := a; i < b; i++ {
		l := &m.Layers[i]
		y := make([]float64, l.Out)
		for o := 0; o < l.Out; o++ {
			sum := l.B[o]
			row := l.W[o*l.In : (o+1)*l.In]
			for k := 0; k < l.In; k++ {
				sum += row[k] * cur[k]
			}
			if i != last {
				y[o] = math.Tanh(sum)
			} else {
				y[o] = sum
			}
		}
		xc := make([]float64, len(cur))
		copy(xc, cur)
		caches[i-a] = layerCache{x: xc, y: y}
		cur = y
	}
	return cur, caches
}

// LayerGrad holds dW and dB of one layer for one microbatch.
type LayerGrad struct {
	dW, dB []float64
}

// backwardRange backpropagates dY (grad wrt the output of layer b-1) through
// layers [a,b), where caches came from forwardRange over the same range.
// It returns the grad wrt the range input and one LayerGrad per layer.
func (m *MLP) backwardRange(caches []layerCache, dY []float64, a int) ([]float64, []LayerGrad) {
	grads := make([]LayerGrad, len(caches))
	d := dY
	last := len(m.Layers) - 1
	for i := len(caches) - 1; i >= 0; i-- {
		l := &m.Layers[a+i]
		c := caches[i]
		dz := make([]float64, l.Out)
		if a+i == last {
			copy(dz, d)
		} else {
			for o := range dz {
				dz[o] = d[o] * (1 - c.y[o]*c.y[o])
			}
		}
		dW := make([]float64, l.Out*l.In)
		for o := 0; o < l.Out; o++ {
			row := dW[o*l.In : (o+1)*l.In]
			for k := 0; k < l.In; k++ {
				row[k] = dz[o] * c.x[k]
			}
		}
		dB := make([]float64, l.Out)
		copy(dB, dz)
		dx := make([]float64, l.In)
		for k := 0; k < l.In; k++ {
			s := 0.0
			for o := 0; o < l.Out; o++ {
				s += l.W[o*l.In+k] * dz[o]
			}
			dx[k] = s
		}
		grads[i] = LayerGrad{dW: dW, dB: dB}
		d = dx
	}
	return d, grads
}

// Accumulator sums per-microbatch gradients in a fixed microbatch order,
// which is what makes pipeline training bit-for-bit equal to single-card.
type Accumulator struct {
	dW [][]float64 // per layer
	dB [][]float64
}

func NewAccumulator(m *MLP) *Accumulator {
	acc := &Accumulator{dW: make([][]float64, len(m.Layers)), dB: make([][]float64, len(m.Layers))}
	for i, l := range m.Layers {
		acc.dW[i] = make([]float64, len(l.W))
		acc.dB[i] = make([]float64, len(l.B))
	}
	return acc
}

// Add accumulates one microbatch's gradient of one layer.
func (acc *Accumulator) Add(layer int, g LayerGrad) {
	for i := range g.dW {
		acc.dW[layer][i] += g.dW[i]
	}
	for i := range g.dB {
		acc.dB[layer][i] += g.dB[i]
	}
}

// Update applies SGD with the microbatch-averaged gradient. Every trainer
// calls this same function after accumulating in the same order, so the
// resulting parameters are bit-for-bit identical.
func (acc *Accumulator) Update(m *MLP, lr float64, mbs int) {
	for li := range m.Layers {
		l := &m.Layers[li]
		for i := range l.W {
			g := acc.dW[li][i] / float64(mbs)
			l.W[i] -= lr * g
		}
		for i := range l.B {
			g := acc.dB[li][i] / float64(mbs)
			l.B[i] -= lr * g
		}
	}
}

// mseLossGrad returns the MSE loss and its grad wrt the model output.
func mseLossGrad(y, t []float64) (float64, []float64) {
	n := float64(len(y))
	dY := make([]float64, len(y))
	loss := 0.0
	for i := range y {
		d := y[i] - t[i]
		loss += d * d
		dY[i] = 2 * d / n
	}
	return loss / n, dY
}
