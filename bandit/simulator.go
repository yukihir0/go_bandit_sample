package bandit

func Simulate(algo Algorithm, arms []Arm, numSims, horizon int) (simNums, times, chosenArms []int, rewards, comulativeRewards []float64) {
	simNums = make([]int, numSims*horizon)
	times = make([]int, numSims*horizon)
	chosenArms = make([]int, numSims*horizon)
	rewards = make([]float64, numSims*horizon)
	comulativeRewards = make([]float64, numSims*horizon)

	for i := 0; i < numSims; i++ {
		algo.Reset(len(arms))
		for j := 0; j < horizon; j++ {
			index := i*horizon + j
			simNums[index] = i
			times[index] = j

			chosenArm := algo.SelectArm()
			chosenArms[index] = chosenArm
			reward := arms[chosenArm].Draw()
			rewards[index] = reward

			if j == 0 {
				comulativeRewards[index] = reward
			} else {
				comulativeRewards[index] = comulativeRewards[index-1] + reward
			}
			algo.Update(chosenArm, reward)
		}
	}

	return
}
