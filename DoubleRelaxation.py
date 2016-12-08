import numpy as np
import matplotlib.pyplot as plt
import populations as pop

temp = 1
nSteps = 100000

plt.subplot(221)
energies = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 0.0, -1e-2, -5e-2, -1e-1, -5e-1, -1e0, -5e0,]
relaxations = []
for energy in energies:
        relaxations.append(pop.Population(Tau = 10, Tau2 = 10, population = 1.0, E = energy, temp = temp))
steps = np.empty([nSteps, len(energies) + 1])

for i in xrange(0, 100000):
        for j in xrange(0, len(relaxations)):
                steps[i,0] = i
                steps[i, j + 1] = relaxations[j].GetPopulationDifference()
                relaxations[j].TakeStep()



for j in xrange(0, len(relaxations)):
        plt.plot(steps[:,0], steps[:,j + 1], label = str(energies[j]) + " energy", lw = 3)

plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Relaxation Down")


plt.subplot(223)
energies = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 0.0, -1e-2, -5e-2, -1e-1, -5e-1, -1e0, -5e0,]
relaxations = []
for energy in energies:
        relaxations.append(pop.Population(Tau = 10, Tau2 = 10, population = -1.0, E = energy, temp = temp))
steps = np.empty([nSteps, len(energies) + 1])

for i in xrange(0, 100000):
        for j in xrange(0, len(relaxations)):
                steps[i,0] = i
                steps[i, j + 1] = relaxations[j].GetPopulationDifference()
                relaxations[j].TakeStep()



for j in xrange(0, len(relaxations)):
        plt.plot(steps[:,0], steps[:,j + 1], label = str(energies[j]) + " energy", lw = 3)

plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Relaxation Up")


plt.subplot(222)
energies = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 0.0, -1e-2, -5e-2, -1e-1, -5e-1, -1e0, -5e0,]
relaxations = []
for energy in energies:
	relaxations.append(pop.Population(Tau = 10, Tau2 = 100, population = 1.0, E = energy, temp = temp))
steps = np.empty([nSteps, len(energies) + 1])

for i in xrange(0, 100000):
	for j in xrange(0, len(relaxations)):
		steps[i,0] = i
		steps[i, j + 1] = relaxations[j].GetPopulationDifference()
		relaxations[j].TakeStep()
		


for j in xrange(0, len(relaxations)):
	plt.plot(steps[:,0], steps[:,j + 1], label = str(energies[j]) + " energy", lw = 3)

plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Relaxation Down")


plt.subplot(224)
energies = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 0.0, -1e-2, -5e-2, -1e-1, -5e-1, -1e0, -5e0,]
relaxations = []
for energy in energies:
        relaxations.append(pop.Population(Tau = 10, Tau2 = 100, population = -1.0, E = energy, temp = temp))
steps = np.empty([nSteps, len(energies) + 1])

for i in xrange(0, 100000):
        for j in xrange(0, len(relaxations)):
                steps[i,0] = i
                steps[i, j + 1] = relaxations[j].GetPopulationDifference()
                relaxations[j].TakeStep()



for j in xrange(0, len(relaxations)):
        plt.plot(steps[:,0], steps[:,j + 1], label = str(energies[j]) + " energy", lw = 3)

plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.legend()
plt.title("Relaxation Up")

plt.show()
