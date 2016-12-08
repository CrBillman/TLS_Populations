import numpy as np
import matplotlib.pyplot as plt
import populations as pop

temp = 1
nSteps = 100000


energies = [1e-2, 5e-2, 1e-1, 1e0]
relaxations = []
field = 0.0
for energy in energies:
	relaxations.append(pop.Population(Tau = 10, population = 1.0, E = energy, temp = temp, Efield = field))

field = 1
for energy in energies:
        relaxations.append(pop.Population(Tau = 10, population = 1.0, E = energy, temp = temp, Efield = field))
steps = np.empty([nSteps, len(relaxations) + 1])

for i in xrange(0, 100000):
	for j in xrange(0, len(relaxations)):
		steps[i,0] = i
		steps[i, j + 1] = relaxations[j].GetPopulationDifference()
		relaxations[j].TakeStep()
		


for j in xrange(0, len(energies)):
	plt.plot(steps[:,0], steps[:,j + 1], label = "Zero field, energy=" + str(energies[j]))

for j in xrange(0, len(energies)):
        plt.plot(steps[:,0], steps[:,j + +len(energies) + 1], label = str(field) + " field, energy=" + str(energies[j]))



plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Dependence of Relaxation on Energy Splitting")
plt.show()
