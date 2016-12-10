import numpy as np
import matplotlib.pyplot as plt
import populations as pop

def RunSimulation(sim, nSteps = 1000):
	steps = np.empty([nSteps, 2])
	steps[:, 0] = np.linspace(0, stop = nSteps, num = nSteps, endpoint = False)
	for j in xrange(0, nSteps):
		steps[j, 1] = np.real(sim.GetPopulationDifference())
		sim.TakeStep()
	return steps

def RunSimulationOscillatingField(sim, nSteps = 1000, omega = 10, e0 = 1.0):
        steps = np.empty([nSteps, 2])
        steps[:, 0] = np.linspace(0, stop = nSteps, num = nSteps, endpoint = False)
        for j in xrange(0, nSteps):
		sim.ApplyField(e0 * np.exp( 1j * omega * j))
                steps[j, 1] = np.real(sim.GetPopulationDifference())
                sim.TakeStep()
        return steps


def RunSimulations(sims, nSteps = 1000):
	steps = np.empty([nSteps, len(sims) + 1])
	steps[:, 0] = np.linspace(1, stop = nSteps, num = nSteps)
	print steps[0,0], steps[0, -1]
	for j in xrange(0, len(sims)):
		steps[:, j + 1] = sims[j].RunSimulation(sims[j], nSteps)[:, 1]
	return steps

def RunSimulationsOscillatingField(sims, nSteps = 1000, omega = 10, e0 = 1.0):
        steps = np.empty([nSteps, len(sims) + 1])
        steps[:, 0] = np.linspace(1, stop = nSteps, num = nSteps)
        print steps[0,0], steps[0, -1]
        for j in xrange(0, len(sims)):
                steps[:, j + 1] = sims[j].RunSimulationOscillatingField(sims[j], nSteps, omega, e0)[:, 1]
        return steps

temp = 1
nSteps = 100000
omega = 5
delta = 0.1
delta0 = 1e-4
dipole = 0.01

fields = [1e2, 1e1, 1, 1e-1, 1e-2]

pops = []
for field in fields:
	pops.append(pop.Population(Tau = 100, population = 0.5, temp = temp, delta = delta, delta0 = delta0, dipole = dipole))

AllSteps = []
for i in xrange(0,len(pops)):
	AllSteps.append(RunSimulationOscillatingField(pops[i], nSteps = nSteps, omega = omega, e0 = fields[i]))

for i in xrange(0,len(pops)):
	steps = AllSteps[i]
        plt.plot(steps[:,0], steps[:,1], label = "E-Field = " + str(fields[i]))

equil = dipole * np.tanh( np.sqrt(delta ** 2 + delta0 ** 2) / (2 * temp))
plt.axhline(equil, color = 'black', lw = 2, ls = '-')


plt.xlabel("Number of Steps")
plt.ylabel("Population Difference")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("Dependence of Relaxation on Energy Splitting")
plt.show()
