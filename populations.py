import numpy as np
import matplotlib.pyplot as plt

class Population:
        def __init__(self, Tau = None, population = None, temp = None, delta = None, delta0 = None, Tau2 = None, Efield = None, dipole = None):
                if Tau:
                        self.Tau1 = Tau
                else:
                        self.Tau1 = 100
		if Tau2:
			self.Tau2 = Tau2
		else:
			self.Tau2 = self.Tau1

                if population:
                        self.population = population
                else:
                        self.population= 0.0
                if temp:
                        self.temp = temp
                else:
                        self.temp = 0
		if Efield:
			self.field = Efield
		else:
			self.field = 0.0
		if delta:
			self.delta = delta
		else:
			self.delta = 0.0
		if delta0:
			self.delta0 = delta0
		else:
			self.delta0 = 1e-4
		if dipole:
			self.dipole = dipole
		else:
			self.dipole = 0.01
		self.alpha = 0.0
		self.UpdatePolarizability()
		self.UpdateEquilibrium()
                return

        def ApplyField(self, E):
                self.field = E
		self.UpdateEquilibrium()
                return

        def UpdateEquilibrium(self):
		E = np.sqrt(self.delta ** 2 + self.delta0 ** 2)
		try:
			#self.equilibrium = self.alpha * self.field + self.dipole * np.tanh( E / ( 2 * self.temp ))
			self.equilibrium = self.dipole * np.tanh( E / ( 2 * self.temp ))
		except ZeroDivisionError:
			#self.equilibrium = self.alpha * self.field - self.dipole
			self.equilibrium = - self.dipole
		#print("New equilibrium value is " + str(self.equilibrium))
		return

        def TakeStep(self, deltaT = 0.1):
		diff = self.population - self.equilibrium
		if diff >= 0.0:
			partial = (self.alpha * self.field - diff) / self.Tau1
		else:
			partial = (self.alpha * self.field - diff) / self.Tau2
                self.population = self.population + partial * deltaT

        def UpdatePolarizability(self):
		E = np.sqrt(self.delta ** 2 + self.delta0 ** 2)
		self.alpha = self.dipole ** 2 / (3 * self.temp) * (self.delta / E) ** 2 * np.cosh( E / (2 * self.temp)) ** (-2.0)
                return

        def GetPopulationDifference(self):
                return self.population
