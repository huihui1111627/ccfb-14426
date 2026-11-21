// Command demo prints a slow-card scenario comparison of the three schedules.
package main

import (
	"flag"
	"fmt"

	"pipesched/internal/pipe"
)

func main() {
	L := flag.Int("layers", 16, "model layers")
	p := flag.Int("devices", 4, "number of cards")
	m := flag.Int("microbatches", 8, "microbatches per step")
	v := flag.Int("vchunks", 2, "virtual chunks per device (interleaved)")
	slowCard := flag.Int("slowcard", 1, "index of the slowed card (-1 = none)")
	slowFactor := flag.Float64("slowfactor", 0.25, "compute speed of the slowed card")
	latency := flag.Float64("latency", 0.3, "inter-device transfer latency")
	flag.Parse()

	ideal := pipe.DefaultSimConfig(*p)
	slow := pipe.DefaultSimConfig(*p)
	if *slowCard >= 0 && *slowCard < *p {
		slow.Speed[*slowCard] = *slowFactor
	}
	slow.Latency = *latency

	scheds := []*pipe.Schedule{
		pipe.GenGPipe(*p, *m, *L),
		pipe.Gen1F1B(*p, *m, *L),
		pipe.GenInterleaved(*p, *m, *v, *L),
	}

	fmt.Printf("L=%d  p=%d  m=%d  v=%d  慢卡=卡%d(%.2fx)  传输时延=%.2f\n\n",
		*L, *p, *m, *v, *slowCard, *slowFactor, *latency)
	fmt.Printf("%-24s %12s %12s %14s %14s %12s\n", "调度", "理想耗时", "慢卡耗时", "气泡(理想)", "气泡(理论)", "峰值激活")
	for _, s := range scheds {
		ri, err := pipe.Simulate(s, ideal)
		if err != nil {
			panic(err)
		}
		rs, err := pipe.Simulate(s, slow)
		if err != nil {
			panic(err)
		}
		if err := pipe.Validate(s, rs, slow); err != nil {
			panic(err)
		}
		bubbleT, peakT := pipe.Theory(s)
		fmt.Printf("%-24s %12.2f %12.2f %13.1f%% %13.1f%%  %v\n",
			s.Name, ri.Makespan, rs.Makespan, ri.Bubble*100, bubbleT*100, peakT)
	}
}
