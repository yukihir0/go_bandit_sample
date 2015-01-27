package bandit

type Algorithm interface {
	Reset(int)
	SelectArm() int
	Update(int, float64)
}

func Max(d []float64) (index int, value float64) {
	index = 0
	value = d[0]
	for i, v := range d {
		if v > value {
			index = i
			value = v
		}
	}

	return
}
