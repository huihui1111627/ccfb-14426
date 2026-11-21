package pipe

import (
	"math"
	"math/rand"
	"testing"
)

func dimsFor(L, din, hidden, dout int) []int {
	dims := make([]int, L+1)
	dims[0] = din
	for i := 1; i < L; i++ {
		dims[i] = hidden
	}
	dims[L] = dout
	return dims
}

func allSchedules(p, m, v, L int) []*Schedule {
	return []*Schedule{GenGPipe(p, m, L), Gen1F1B(p, m, L), GenInterleaved(p, m, v, L)}
}

// TestBitwiseRandomCombos: for dozens of random (layers, cards, microbatches,
// virtual chunks) combos, every schedule must (a) produce a legal timeline
// with zero dependency violations and (b) train to parameters bit-for-bit
// identical to single-card training.
func TestBitwiseRandomCombos(t *testing.T) {
	rng := rand.New(rand.NewSource(20261122))
	for i := 0; i < 48; i++ {
		L := 1 + rng.Intn(24)
		p := 1 + rng.Intn(6)
		m := 1 + rng.Intn(16)
		v := 1 + rng.Intn(4)
		steps := 1 + rng.Intn(3)
		cfg := TrainConfig{
			Dims: dimsFor(L, 3, 5, 2), M: m, Steps: steps,
			LR: 0.05, ModelSeed: 42, DataSeed: 7,
		}
		sim := DefaultSimConfig(p)
		for d := range sim.Speed { // random heterogeneous cards incl. slow ones
			sim.Speed[d] = 0.3 + rng.Float64()*1.2
		}
		sim.Latency = rng.Float64() * 0.5

		want := TrainSingleCard(cfg)
		for _, sched := range allSchedules(p, m, v, L) {
			res, err := Simulate(sched, sim)
			if err != nil {
				t.Fatalf("combo L=%d p=%d m=%d v=%d %s: %v", L, p, m, v, sched.Name, err)
			}
			if err := Validate(sched, res, sim); err != nil {
				t.Fatalf("combo L=%d p=%d m=%d v=%d %s: dependency violation: %v",
					L, p, m, v, sched.Name, err)
			}
			got := TrainPipeline(cfg, sched, res)
			if !want.Equal(got) {
				t.Fatalf("combo L=%d p=%d m=%d v=%d steps=%d %s: params differ from single-card",
					L, p, m, v, steps, sched.Name)
			}
		}
	}
}

// TestTheoryMatch: in the idealised setting (equal cards, no latency, equal
// chunks) simulated bubble ratios and peak activations must equal the
// closed-form theoretical values for every schedule.
func TestTheoryMatch(t *testing.T) {
	for p := 1; p <= 5; p++ {
		for m := 1; m <= 12; m++ {
			for v := 1; v <= 3; v++ {
				L := p * v * 2 // divisible by both p and p*v -> equal chunks
				sim := DefaultSimConfig(p)
				for _, sched := range allSchedules(p, m, v, L) {
					res, err := Simulate(sched, sim)
					if err != nil {
						t.Fatalf("p=%d m=%d v=%d %s: %v", p, m, v, sched.Name, err)
					}
					if err := Validate(sched, res, sim); err != nil {
						t.Fatalf("p=%d m=%d v=%d %s: %v", p, m, v, sched.Name, err)
					}
					wantBubble, wantPeak := Theory(sched)
					if math.Abs(res.Bubble-wantBubble) > 1e-9 {
						t.Errorf("p=%d m=%d v=%d %s: bubble %.6f, theory %.6f",
							p, m, v, sched.Name, res.Bubble, wantBubble)
					}
					for d := 0; d < p; d++ {
						if res.Peak[d] != wantPeak[d] {
							t.Errorf("p=%d m=%d v=%d %s: peak[%d]=%d, theory %d",
								p, m, v, sched.Name, d, res.Peak[d], wantPeak[d])
						}
					}
				}
			}
		}
	}
}

// TestInFlightLimit1F1B: 1F1B must never have more microbatches in flight on
// a device than the stage count.
func TestInFlightLimit1F1B(t *testing.T) {
	for p := 2; p <= 6; p++ {
		for m := 1; m <= 3*p; m++ {
			sched := Gen1F1B(p, m, p*2)
			res, err := Simulate(sched, DefaultSimConfig(p))
			if err != nil {
				t.Fatal(err)
			}
			for d := 0; d < p; d++ {
				if res.Peak[d] > p {
					t.Errorf("p=%d m=%d: device %d has %d microbatches in flight (> %d stages)",
						p, m, d, res.Peak[d], p)
				}
			}
		}
	}
}

// TestSlowCard: with one slow card and transfer latency, all schedules must
// still produce legal timelines; report how each schedule's makespan reacts.
func TestSlowCard(t *testing.T) {
	p, m, v, L := 4, 8, 2, 16
	ideal := DefaultSimConfig(p)
	slow := DefaultSimConfig(p)
	slow.Speed[1] = 0.25 // one card at quarter speed
	slow.Latency = 0.3

	for _, sched := range allSchedules(p, m, v, L) {
		ri, err := Simulate(sched, ideal)
		if err != nil {
			t.Fatal(err)
		}
		rs, err := Simulate(sched, slow)
		if err != nil {
			t.Fatal(err)
		}
		if err := Validate(sched, rs, slow); err != nil {
			t.Fatalf("%s slow-card: %v", sched.Name, err)
		}
		// makespan must cover the busiest card
		maxBusy := 0.0
		for _, b := range rs.Busy {
			if b > maxBusy {
				maxBusy = b
			}
		}
		if rs.Makespan < maxBusy-1e-9 {
			t.Errorf("%s: makespan %.2f below busiest card %.2f", sched.Name, rs.Makespan, maxBusy)
		}
		t.Logf("%-22s ideal makespan %6.2f (bubble %5.1f%%)  slow-card makespan %6.2f (bubble %5.1f%%)",
			sched.Name, ri.Makespan, ri.Bubble*100, rs.Makespan, rs.Bubble*100)
	}
}
