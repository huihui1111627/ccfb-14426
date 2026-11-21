package pipe

// Theoretical values, derived for the idealised hardware model (identical
// cards, zero transfer latency, equal-size chunks). The simulator must
// reproduce these exactly in that setting.

// TheoryBubbleGPipe: makespan (M+P-1)*(tf+tb) vs useful M*(tf+tb) per device.
func TheoryBubbleGPipe(p, m int) float64 {
	return float64(p-1) / float64(m+p-1)
}

// TheoryPeakGPipe: every device stores all M microbatches' activations.
func TheoryPeakGPipe(p, m int) []int {
	peak := make([]int, p)
	for d := range peak {
		peak[d] = m
	}
	return peak
}

// TheoryBubble1F1B: same makespan as GPipe — 1F1B saves memory, not time.
func TheoryBubble1F1B(p, m int) float64 {
	return float64(p-1) / float64(m+p-1)
}

// TheoryPeak1F1B: device d keeps at most P-d microbatches in flight
// (warmup depth P-1-d plus the one whose backward is pending). For M < P the
// schedule degenerates to GPipe order and every device peaks at M.
func TheoryPeak1F1B(p, m int) []int {
	peak := make([]int, p)
	for d := range peak {
		if m < p {
			peak[d] = m
		} else {
			peak[d] = p - d
		}
	}
	return peak
}

// TheoryBubbleInterleaved: with V chunks per device the fill/drain shrinks to
// (P-1)/V of a full stage. Makespan = max(M+P*V-1, M*V+P-1) chunk-slots
// (chain-bound for small M, device-bound otherwise) vs useful M*V slots.
func TheoryBubbleInterleaved(p, m, v int) float64 {
	chain := m + p*v - 1
	device := m*v + p - 1
	mk := chain
	if device > mk {
		mk = device
	}
	return 1 - float64(m*v)/float64(mk)
}

// TheoryPeakInterleaved: in this construction each device runs all its
// forwards before its backwards, so it peaks at M*V chunk-activations
// (= M full-stage equivalents, each chunk being 1/V of a stage).
func TheoryPeakInterleaved(p, m, v int) []int {
	peak := make([]int, p)
	for d := range peak {
		peak[d] = m * v
	}
	return peak
}

// Theory returns the theoretical bubble fraction and per-device peak
// activation for the given schedule.
func Theory(s *Schedule) (bubble float64, peak []int) {
	switch s.Kind {
	case "1f1b":
		return TheoryBubble1F1B(s.P, s.M), TheoryPeak1F1B(s.P, s.M)
	case "interleaved":
		return TheoryBubbleInterleaved(s.P, s.M, s.V), TheoryPeakInterleaved(s.P, s.M, s.V)
	default:
		return TheoryBubbleGPipe(s.P, s.M), TheoryPeakGPipe(s.P, s.M)
	}
}
