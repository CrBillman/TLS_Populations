import numpy as np
import matplotlib.pyplot as plt
import populations as pop

temp = 1
nSteps = 100000


plt.subplot(121)
energies = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
relaxations = []
for energy in energies:
	relaxations.append(pop.Population(Tau = 10, population = 1.0, E = energy, temp = temp))
steps = np.empty([nSteps, len(energies) + 1])

for i in xrange(0, 100000):
	for j in xrange(0, len(relaxations)):
		steps[i,0] = i
		steps[i, j + 1] = relaxations[j].GetPopulationDifference()
		relaxations[j].TakeStep()
		


for j in xrange(0, len(relaxations)):
	plt.plot(steps[:,0], steps[:,j + 1], label = str(energies[j]) + " energy")

plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Dependence of Relaxation on Energy Splitting")


plt.subplot(122)
temps = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
relaxations = []
for temp in temps:
        relaxations.append(pop.Population(Tau = 10, population = 1.0, E = 1.0, temp = temp))
steps = np.empty([nSteps, len(energies) + 1])

for i in xrange(0, 100000):
        for j in xrange(0, len(relaxations)):
                steps[i,0] = i
                steps[i, j + 1] = relaxations[j].GetPopulationDifference()
                relaxations[j].TakeStep()



for j in xrange(0, len(relaxations)):
        plt.plot(steps[:,0], steps[:,j + 1], label = str(energies[j]) + " temperature")

plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Dependence of Relaxation on Temperature")

plt.show()
