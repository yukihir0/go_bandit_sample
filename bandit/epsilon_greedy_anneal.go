package bandit

import (
	"math"
	"math/rand"
)

type EpsilonGreedyAnneal struct {
	Epsilon float64
	Counts  []int
	Values  []float64
}

func NewEpsilonGreedyAnneal(epsilon float64, nArms int) (a *EpsilonGreedyAnneal) {
	return &EpsilonGreedyAnneal{Epsilon: epsilon, Counts: make([]int, nArms), Values: make([]float64, nArms)}
}

func (self *EpsilonGreedyAnneal) Reset(nArms int) {
	self.Counts = make([]int, nArms)
	self.Values = make([]float64, nArms)
}

func (self *EpsilonGreedyAnneal) SelectArm() (index int) {
	t := 0
	for _, v := range self.Counts {
		t = t + v
	}
	self.Epsilon = 1.0 / math.Log(float64(t+1)+0.0000001)

	if rand.Float64() > self.Epsilon {
		maxIndex, _ := Max(self.Values)
		index = maxIndex
	} else {
		index = rand.Intn(len(self.Values))
	}
	return
}

func (self *EpsilonGreedyAnneal) Update(chosenArm int, reward float64) {
	self.Counts[chosenArm] = self.Counts[chosenArm] + 1
	n := self.Counts[chosenArm]

	value := self.Values[chosenArm]
	if n == 0 {
		self.Values[chosenArm] = reward
	} else {
		self.Values[chosenArm] = (float64(n-1)/float64(n))*value + (1.0/float64(n))*reward
	}
}
