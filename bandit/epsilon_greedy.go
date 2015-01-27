package bandit

import (
	"math/rand"
)

type EpsilonGreedy struct {
	Epsilon float64
	Counts  []int
	Values  []float64
}

func NewEpsilonGreedy(epsilon float64, nArms int) (a *EpsilonGreedy) {
	return &EpsilonGreedy{Epsilon: epsilon, Counts: make([]int, nArms), Values: make([]float64, nArms)}
}

func (self *EpsilonGreedy) Reset(nArms int) {
	self.Counts = make([]int, nArms)
	self.Values = make([]float64, nArms)
}

func (self *EpsilonGreedy) SelectArm() (index int) {
	if rand.Float64() > self.Epsilon {
		maxIndex, _ := Max(self.Values)
		index = maxIndex
	} else {
		index = rand.Intn(len(self.Values))
	}
	return
}

func (self *EpsilonGreedy) Update(chosenArm int, reward float64) {
	self.Counts[chosenArm] = self.Counts[chosenArm] + 1
	n := self.Counts[chosenArm]

	value := self.Values[chosenArm]
	if n == 0 {
		self.Values[chosenArm] = reward
	} else {
		self.Values[chosenArm] = (float64(n-1)/float64(n))*value + (1.0/float64(n))*reward
	}
}
