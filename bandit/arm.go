package bandit

import (
	"math/rand"
)

type Arm interface {
	Draw() float64
}

type BernoulliArm struct {
	Prob float64
}

func NewBernoulliArm(p float64) (b *BernoulliArm) {
	return &BernoulliArm{Prob: p}
}

func (self *BernoulliArm) Draw() (p float64) {
	if rand.Float64() > self.Prob {
		p = 0.0
	} else {
		p = 1.0
	}
	return
}
