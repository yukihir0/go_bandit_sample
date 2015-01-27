package bandit

import (
	"math"
)

type UCB1 struct {
	Counts []int
	Values []float64
}

func NewUCB1(nArms int) (a *UCB1) {
	return &UCB1{Counts: make([]int, nArms), Values: make([]float64, nArms)}
}

func (self *UCB1) Reset(nArms int) {
	self.Counts = make([]int, nArms)
	self.Values = make([]float64, nArms)
}

func (self *UCB1) SelectArm() (index int) {
	for i, v := range self.Counts {
		if v == 0 {
			return i
		}
	}

	totalCounts := 0
	for _, v := range self.Counts {
		totalCounts = totalCounts + v
	}

	ucbValues := []float64{}
	for i, _ := range self.Counts {
		bonus := math.Sqrt(2.0 * math.Log(float64(totalCounts)) / float64(self.Counts[i]))
		ucbValues = append(ucbValues, self.Values[i]+bonus)
	}

	maxIndex, _ := Max(ucbValues)
	return maxIndex
}

func (self *UCB1) Update(chosenArm int, reward float64) {
	self.Counts[chosenArm] = self.Counts[chosenArm] + 1
	n := self.Counts[chosenArm]

	value := self.Values[chosenArm]
	if n == 0 {
		self.Values[chosenArm] = reward
	} else {
		self.Values[chosenArm] = (float64(n-1)/float64(n))*value + (1.0/float64(n))*reward
	}
}
