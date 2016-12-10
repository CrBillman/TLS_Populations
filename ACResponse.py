import numpy as np
import matplotlib.pyplot as plt
import populations as pop

def RunSimulation(sim, nSteps = 1000, stepSize = 0.1):
	steps = np.empty([nSteps, 2])
	steps[:, 0] = np.linspace(0, stop = nSteps, num = nSteps, endpoint = False) * stepSize
	for j in xrange(0, nSteps):
		steps[j, 1] = np.real(sim.GetPopulationDifference())
		sim.TakeStep(stepSize)
	return steps

def RunSimulationLite(sim, fn, nSteps = 1000, stepSize = 0.1):
        for j in xrange(0, nSteps):
                popDiff = sim.GetPopulationDifference()
		with open(fn, 'a') as f:
			f.write(", ".join([str(j * stepSize), str(np.real(popDiff)), str(np.imag(popDiff))]) + '\n')
		sim.TakeStep(stepSize)
		if j % (0.01 * nSteps) == 0:
			print str(j / (0.01 * nSteps)) + "% done."
        return steps


def RunSimulationOscillatingField(sim, nSteps = 1000, omega = 10, e0 = 1.0, stepSize = 0.1):
        steps = np.empty([nSteps, 2])
        steps[:, 0] = np.linspace(0, stop = nSteps, num = nSteps, endpoint = False) * stepSize
        for j in xrange(0, nSteps):
		steps[j, 1] = np.real(sim.GetPopulationDifference())
		sim.ApplyField(e0 * np.exp( 1j * omega * j))
                sim.TakeStep(stepSize)
        return steps

def RunSimulationOscillatingFieldLite(sim, fn, nSteps = 1000, omega = 10, e0 = 1.0, stepSize = 0.1, clear = None):
	if clear:
		with open(fn, 'w') as f:
			f.write('')
	field = 1.0
	dumpList = []
        for j in xrange(0, nSteps):
                popDiff = sim.GetPopulationDifference()
		dumpList.append(", ".join([str(j * stepSize), str(np.real(popDiff)), str(np.imag(popDiff)), str(field), str(np.real(popDiff / field)), str(np.imag(popDiff / field))]) + '\n')
		field = e0 * np.exp( 1j * omega * j)
		sim.ApplyField(field)
                sim.TakeStep(stepSize)
		if j % (10000) == 0:
			with open(fn, 'a') as f:
				f.write("".join(dumpList))
			dumpList = []
                if j % (0.01 * nSteps) == 0:
                        print str(j / (0.01 * nSteps)) + "% done."
	if len(dumpList) > 0:
		with open(fn, 'a') as f:
			f.write("".join(dumpList))

        return


def RunSimulations(sims, nSteps = 1000, stepSize = 0.1):
	steps = np.empty([nSteps, len(sims) + 1]) * stepSize
	steps[:, 0] = np.linspace(1, stop = nSteps, num = nSteps)
	for j in xrange(0, len(sims)):
		steps[:, j + 1] = sims[j].RunSimulation(sims[j], nSteps, stepSize)[:, 1]
	return steps

def RunSimulationsOscillatingField(sims, nSteps = 1000, omega = 10, e0 = 1.0, stepSize = 0.1):
        steps = np.empty([nSteps, len(sims) + 1]) * stepSize
        steps[:, 0] = np.linspace(1, stop = nSteps, num = nSteps)
        for j in xrange(0, len(sims)):
                steps[:, j + 1] = sims[j].RunSimulationOscillatingField(sims[j], nSteps = nSteps, omega = omega, e0 = e0, stepSize = stepSize)[:, 1]
        return steps

def PlotSimulations(sims, fn = None):
	labels = [1e4, 1e3, 1e2]
	labels = ["E-Field = " + label for label in labels]
	for i in xrange(0,len(pops)):
		steps = sims[i]
		plt.plot(steps[:-1:2,0], steps[:-1:2,1], label = "E-Field = " + str(fields[i]))

	plt.xlabel("Time")
	plt.ylabel("Population Difference")

	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.title("Dependence of Relaxation on Energy Splitting")
	if fn:
		plt.savefig(fn)
	else:
		plt.show()

filenames = "ACR_"
temp = 1
nSteps = 1e7
nSteps = int(nSteps)
omega = 10
delta = 0.1
delta0 = 1e-4
dipole = 0.01
stepSize = 1e-5
initPop = 5e-1
fields = [1e5, 1e4, 1e3]
#fields = [1e5, 1e4, 1e3, 1e2]

pops = []
for field in fields:
	pops.append(pop.Population(Tau = 0.1, Tau2 = 0.1, population = initPop, temp = temp, delta = delta, delta0 = delta0, dipole = dipole))

AllSteps = []
#for i in xrange(0,len(pops)):
#	AllSteps.append(RunSimulationOscillatingField(pops[i], nSteps = nSteps, omega = omega, e0 = fields[i], stepSize = stepSize))
#	np.savetxt(filenames + str(i), AllSteps[i])

for i in xrange(0, len(pops)):
	RunSimulationOscillatingFieldLite(pops[i], filenames + str(i), nSteps = nSteps, omega = omega, e0 = fields[i], stepSize = stepSize, clear = True)
