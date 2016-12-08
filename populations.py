import numpy as np
import matplotlib.pyplot as plt

class Population:
        def __init__(self, Tau = None, population = None, temp = None, E = None, Tau2 = None):
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
                        population= 0.0
                if temp:
                        self.temp = temp
                else:
                        self.temp = 0
                if E:
                        self.E = E
                else:
                        self.E = 0.0
		self.UpdateEquilibrium()
                self.field = 0.0
                return

        def ApplyField(self, E):
                self.field = E
                return

        def UpdateEquilibrium(self):
		try:
			self.equilibrium = np.tanh( self.E / ( 2 * self.temp ))
		except ZeroDivisionError:
			self.equilibrium = -1.0
		print("New equilibrium value is " + str(self.equilibrium))
		return

        def TakeStep(self, deltaT = 0.1):
		diff = self.population - self.equilibrium
		if diff >= 0.0:
			partial = - self.GetPolarizability() * self.field - diff / self.Tau1
		else:
			partial = - self.GetPolarizability() * self.field - diff / self.Tau2
                self.population = self.population + partial * deltaT

        def GetPolarizability(self):
                return 0.0

        def GetPopulationDifference(self):
                return self.population
