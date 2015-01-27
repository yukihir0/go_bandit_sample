package main

import (
	"./bandit"
	"fmt"
	"image/color"
	"math/rand"
	"time"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"code.google.com/p/plotinum/vg"
)

const (
	NUM_SIMS = 5000
	HORIZON  = 250
)

// algorithms
const (
	EPSILON_GREEDY = iota
	EPSILON_GREEDY_ANNEAL
	SOFTMAX
	SOFTMAX_ANNEAL
	UCB1
)

func main() {
	rand.Seed(time.Now().Unix())

	// setup arms
	means := []float64{0.1, 0.1, 0.1, 0.1, 0.9}
	nArms := len(means)
	shuffleFloat64(means)

	arms := []bandit.Arm{}
	for _, mu := range means {
		arms = append(arms, bandit.NewBernoulliArm(mu))
	}
	bestArm := bestArm(means)

	// setup plot
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "bandit"
	p.X.Label.Text = "time"
	p.X.Min = 0.0
	p.X.Max = 250.0
	p.Y.Label.Text = "prob"
	p.Y.Min = 0.0
	p.Y.Max = 1.0
	p.Add(plotter.NewGrid())
	colors := []color.RGBA{
		color.RGBA{R: 244, G: 67, B: 54, A: 255},
		color.RGBA{R: 233, G: 30, B: 99, A: 255},
		color.RGBA{R: 156, G: 39, B: 176, A: 255},
		color.RGBA{R: 103, G: 58, B: 183, A: 255},
		color.RGBA{R: 63, G: 81, B: 181, A: 255},
	}
	pts := make(plotter.XYs, HORIZON)

	// simulation
	epsilons := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	for i, epsilon := range epsilons {
		algorithms := []bandit.Algorithm{
			bandit.NewEpsilonGreedy(epsilon, nArms),
			bandit.NewEpsilonGreedyAnneal(epsilon, nArms),
			bandit.NewSoftmax(epsilon, nArms),
			bandit.NewSoftmaxAnneal(epsilon, nArms),
			bandit.NewUCB1(nArms),
		}

		//return value: simNums, times, chosenArms, rewards, comulativeRewards
		_, _, chosenArms, _, _ := bandit.Simulate(algorithms[EPSILON_GREEDY], arms, NUM_SIMS, HORIZON)

		calculateProbPoint(pts, bestArm, chosenArms)
		plotLine(p, pts, colors[i], fmt.Sprintf("%v", epsilon))
	}

	fileName := p.Title.Text + ".png"
	if err := p.Save(4, 4, fileName); err != nil {
		panic(err)
	}
}

func shuffleFloat64(s []float64) {
	n := len(s)
	for i := 0; i < n; i++ {
		j := rand.Intn(n - 1)
		s[j], s[i] = s[i], s[j]
	}
}

func bestArm(means []float64) (index int) {
	index, _ = bandit.Max(means)
	return
}

func calculateProbPoint(pts plotter.XYs, bestArm int, chosenArms []int) {
	for i := 0; i < HORIZON; i++ {
		count := 0
		for j := 0; j < NUM_SIMS; j++ {
			index := HORIZON*j + i
			if chosenArms[index] == bestArm {
				count = count + 1
			}
		}

		pts[i].X = float64(i)
		pts[i].Y = float64(count) / float64(NUM_SIMS)
	}
}

func plotLine(p *plot.Plot, pts plotter.XYs, color color.RGBA, legend string) {
	l, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Color = color
	p.Add(l)
	p.Legend.Add(legend, l)
}
