package bandit

import (
	"math"
	"math/rand"
)

type SoftmaxAnneal struct {
	Temperature float64
	Counts      []int
	Values      []float64
}

func NewSoftmaxAnneal(temperature float64, nArms int) (a *SoftmaxAnneal) {
	return &SoftmaxAnneal{Temperature: temperature, Counts: make([]int, nArms), Values: make([]float64, nArms)}
}

func (self *SoftmaxAnneal) Reset(nArms int) {
	self.Counts = make([]int, nArms)
	self.Values = make([]float64, nArms)
}

func (self *SoftmaxAnneal) SelectArm() (index int) {
	t := 0
	for _, v := range self.Counts {
		t = t + v
	}
	self.Temperature = 1.0 / math.Log(float64(t+1)+0.0000001)

	z := 0.0
	for _, v := range self.Values {
		z = z + math.Exp(v/self.Temperature)
	}

	probs := []float64{}
	for _, v := range self.Values {
		probs = append(probs, math.Exp(v/self.Temperature)/z)
	}

	return self.CategoricalDraw(probs)
}

func (self *SoftmaxAnneal) CategoricalDraw(probs []float64) (index int) {
	z := rand.Float64()
	cumProb := 0.0
	for i, prob := range probs {
		cumProb = cumProb + prob
		if cumProb > z {
			return i
		}
	}

	return len(probs) - 1
}

func (self *SoftmaxAnneal) Update(chosenArm int, reward float64) {
	self.Counts[chosenArm] = self.Counts[chosenArm] + 1
	n := self.Counts[chosenArm]

	value := self.Values[chosenArm]
	if n == 0 {
		self.Values[chosenArm] = reward
	} else {
		self.Values[chosenArm] = (float64(n-1)/float64(n))*value + (1.0/float64(n))*reward
	}
}
