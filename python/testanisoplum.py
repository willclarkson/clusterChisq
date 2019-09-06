import numpy as N
import pylab as P
from scipy import interpolate

class Aniso(object):

    def __init__(self):

        self.R2D = N.linspace(0.0,4.0, 100.0,endpoint=True)
        self.ModelQ      = 0.5
        self.ModelR0     = 1.0
        self.ModelVDisp0 = 5.0

    def EvaluateGM(self):

        """Evaluate GM for anisotropic plummer model"""

        r = self.R2D

        R0 = self.ModelR0
        u = 1.0 + (r/R0)**2

        s0 = self.ModelVDisp0
        VRad = s0**2 * u**(-0.5)
        
        part1 = 6.0 * s0**2 * r**3 * R0**(-2.0) * u**(-1.5) 

        part2 = 2.0 * r * VRad

        AnisoBracket = (1.0 - 0.5*self.ModelQ * (r**2/(R0**2 + r**2)))
        part3 = 1.0 - AnisoBracket

        self.EvalGM = 231.6 * (part1 - part2*part3)

    def EvaluateRho(self):

        """Evaluate the derivative of GM numerically and calculate the
        density"""

#        rfine = N.linspace(0,N.max(self.r), N.size(self.r)*10, endpoint=True)

        tck = interpolate.splrep(self.R2D, self.EvalGM, s=0)
        tder = interpolate.splev(self.R2D, tck, der = 1)

        rho = tder / (4.0 * N.pi * self.R2D)

        self.EvalRho = N.copy(rho)

    def ShowGM(self, Clobber=False):

        """Plot GM(r) curve for anisotropic plummer model"""

        if Clobber:
            self.FreeFigure()

        P.plot(self.R2D, self.EvalGM,'k')

    def ShowRho(self, Clobber=False):

        """Plot rho(r) curve for anisotropic plummer"""

        if Clobber:
            self.FreeFigure()
            
        P.plot(self.R2D, self.EvalRho, 'g')

    def FreeFigure(self):

        try:
            P.close()
        except:
            dum = 1

             
    
def TestEvalGM(q = 0.5, Clobber=False):

    """Evaluate GM(r) for anisotropic plummer model"""

    A = Aniso()
    A.ModelQ = q
    A.EvaluateGM()
    A.ShowGM(Clobber = Clobber)
    
    
def TestEvalRho(q = 0.5, Clobber=False):

    """Evaluate GM(r) for anisotropic plummer model"""

    A = Aniso()
    A.ModelQ = q
    A.EvaluateGM()
    A.EvaluateRho()
    A.ShowRho(Clobber = Clobber)
    



