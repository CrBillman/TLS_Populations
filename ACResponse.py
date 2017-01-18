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
        popDiff = sim.GetPopulationDifference() - sim.GetEquilibrium()
        field = e0 * np.exp( -1j * omega * j * stepSize)
        sim.ApplyField(field)
        dumpList.append(", ".join([str(j * stepSize), str(np.real(popDiff)), str(np.imag(popDiff)), str(field), str(np.real(popDiff / field)), str(np.imag(popDiff / field))]) + '\n')
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

def MeasureLoss(sim, omega = 10, e0 = 1.0, stepSize = 0.1, prevSteps = 0):
    period = 2 * np.pi / omega
    nSteps = int(period / stepSize + 1)
    field = e0 * np.exp( -1j * omega * prevSteps * stepSize)
    Qsum = 0.0
    info = []
    data = np.zeros([nSteps, 2])
    for j in xrange(prevSteps, prevSteps + nSteps):
        popDiff = sim.GetPopulationDifference() - sim.GetEquilibrium()
        Qsum = Qsum + np.imag(popDiff / field)
        field = e0 * np.exp( -1j * omega * j * stepSize)
        sim.ApplyField(field)
        sim.TakeStep(stepSize)
        info.append(", ".join([str(j * stepSize), str(np.real(popDiff)), str(np.imag(popDiff)), str(field), str(np.real(popDiff / field)), str(np.imag(popDiff / field))]) + '\n')
        data[j - prevSteps, 0] = j * stepSize
        data[j - prevSteps, 1] = np.imag(popDiff / field)
    with open("Measure.dat", 'a') as f:
        f.write("".join(info))
        return np.trapz(data[:,1], x = data[:,0]) / (period)

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

def CalcLoss(dipole, temp, delta, delta0, omega, tau1, tau2 = None):
    if tau2:
        tau = 0.5 * (tau1 + tau2)
    else:
        tau = tau1
    E = np.sqrt(delta0**2 + delta**2)
    pref = dipole ** 2 / (3 * temp) * (delta / E) **2
    res = omega * tau / (1 + omega **2 * tau ** 2)
    sech = np.cosh( E / (2 * temp)) ** (-2.0)
    return pref * res * sech

def CalcAverageLoss(dipole, temp, delta, delta0, omega, tau1, tau2):
    l1 = CalcLoss(dipole, temp, delta, delta0, omega, tau1)
    l2 = CalcLoss(dipole, temp, delta, delta0, omega, tau2)
    return 0.5 * (l1 + l2)

filenames = "ACR_"
temp = 1.0
nSteps = 1e7
nSteps = int(nSteps)
omega = 10
delta = 0.1
delta0 = 1e-3
dipole = 0.01
stepSize = 1e-4
initPop = 5e-4
fields = [1e5, 1e4, 1e3]
field = 1e1
tau1 = 1.0
#tau2s = [0.05, 0.1, 0.125, 0.2, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 20.0]
tau2s = np.logspace(-2,2, num = 100)
#tau2s = np.logspace(0,2, num = 15)
#print tau2s
#tau2s = [1000.0]
#tau2 = 2

ps = []
output = []
for tau2 in tau2s:
    print( "\tOn " + str(tau2))
    p = pop.Population(Tau = tau1, Tau2 = tau2, population = initPop, temp = temp, delta = delta, delta0 = delta0, dipole = dipole)
    p.SetToEquilibrium()
    stepSize = min(1e-4 * tau2, 1e-4)
    nSteps = int(max(round(2 * tau2 / stepSize), round(100 / stepSize)))
    print str(nSteps) + " of size " + str(stepSize) + " for " + str(stepSize * nSteps) + " units of total time."
    RunSimulationOscillatingFieldLite(p, filenames + str(field), nSteps = nSteps, omega = omega, e0 = field, stepSize = stepSize, clear = True)
    Qm = MeasureLoss(p, omega = omega, e0 = field, stepSize = stepSize, prevSteps = nSteps)
    Qp = CalcLoss(dipole, temp, delta, delta0, omega, tau1, tau2)
    #Qav = CalcAverageLoss(dipole, temp, delta, delta0, omega, tau1, tau2)
    #output.append("For Tau=" + str(tau2) + ", Qm = " + str(Qm) + ", Qc = " + str(Qp) + ", Qav = " + str(Qav) + ".  Diff is " + str(abs(Qm-Qp)/Qm * 100) + "%")
    with open("err.dat", 'a') as f:
        f.write(str(tau2) + " " + str(abs(Qm-Qp)/Qm * 100) + '\n')
    #output.append(str(tau2) + ", Qm = " + str(Qm) + ", Qc = " + str(Qp) + ", Qav = " + str(Qav) + ".  Diff is " + str(abs(Qm-Qp)/Qm * 100))
#print "\n".join(output)
