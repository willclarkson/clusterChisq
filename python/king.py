#
# King.py - testbed for integrating the isotropic King profile
# 


import numpy as N
import pylab as P
import os, sys, time, string
from scipy.integrate import quad, dblquad, Inf # integration
from scipy import interpolate
from scipy.optimize import leastsq
import pyfits
from matplotlib import rc

# ability to plot cmyk
import mpl_toolkits.ps_cmyk


class King(object):

    def __init__(self):

        self.r3d = N.linspace(0.01, 10.0, 100, endpoint=True)
        self.R2D = N.copy(self.r3d)

        # dummy model parameters
        self.ModelGamma = 0.25
        self.ModelA     = 2.8
        self.ModelR0    = 1.0
        self.ModelNorm  = 1.0
        self.ModelR0Plummer = 2.0

        self.ModelEnclosedMass2D = N.array([])

        self.ModelKingRc = 1.0
        self.ModelKingRt = 10.0
        self.ModelKingS0 = 10.0

        # anisotropy parameters
        self.ModelAnisoNr = 1.0
        self.ModelAnisoNt = 0.5

       # evaluateds
        self.KingSurfDens = N.array([])
        self.KingNumDens = N.array([])
        self.KingSurfDensEval = N.array([])
        self.KingMassIsoEval =  N.array([])

        # King profile mass polynomial
        self.KingPolyCoeffs = N.array([])
        self.KingPolyDegree = 13

        self.GravConstant = 6.67e-11

        # we are going to be evaluating complicated integrands on a
        # fine grid of radii and representing them with
        # fast-integrable splines. set up the radius vector to do so.
        self.rfine = N.linspace(0.01, self.ModelKingRt, 500)

        # equation 22 of LM89 can be written n^{-1} \int( self.tck_f)
        self.tck_f = N.array([])
        self.tev_f = N.array([])

        # we will want to evaluate sigma_iso^2 = n^{-1} y(r) where
        # y(r) is the spline representation of the integral in
        # equation 22 of LM89. Set up those variables
        self.tck_y = N.array([])
        self.tev_y = N.array([])

        # variables for calculating GM(<r)
        self.GMSquareBracket = N.array([])

    def ArrayIntegrateBP(self):

        """Given a set of projected radii, evaluate the surface
        density"""

        r = self.R2D
        SurfDens = N.zeros(N.size(r))
        
        print N.size(r)

        for iRadii in range(N.size(r)):
            ThisInteg = self.PointIntegrateBP(r[iRadii])
            SurfDens[iRadii] = ThisInteg

        self.ModelEnclosedMass2D = N.copy(SurfDens)

    def PointIntegrateBP(self, inR = None):

        """For strip of radii, evaluate the surface density"""

        try:
            RThis = 1.0 * inR
        except:
            RThis = self.R2D[0]

#        P = N.array([self.ModelNorm, self.ModelR0, \
#                         self.ModelGamma, self.ModelA])

#        Integ = quad( lambda r: self.BPVolumeDensity(P,r), \
#                          RThis, Inf)

        Integ = quad( lambda r: self.BPVolumeDens(r) \
                          * r / N.sqrt(r**2 - RThis**2 ), RThis, Inf)

        return Integ[0]

    def EvalKingSurfDens(self, inR = None):

        """Evaluate the analytic form of the king profile surface density"""

        Rc = self.ModelKingRc
        Rt = self.ModelKingRt
        S0 = self.ModelKingS0

        try:
            R = 1.0 * InR
        except:
            R = self.R2D

        part1  = 1.0 / N.sqrt(1.0 + (R/Rc)**2)
        part2  = 1.0 / N.sqrt(1.0 + (Rt/Rc)**2)
        yval = S0 * (part1 - part2)**2
        self.KingSurfDensEval = N.copy(yval)

    def ArrayKingSurfDens(self):

        """Evaluate the King integral for arary """

        EnclosedMass = N.zeros(N.size(self.R2D))
        for iThis in range(N.size(self.R2D)):
            RThis = self.R2D[iThis]
            dum = self.PointKingSurfDens(RThis)
            EnclosedMass[iThis] = N.copy(dum)

        self.KingSurfDens = N.copy(EnclosedMass)

    def PointKingSurfDens(self, inR = None):

        """Check on the numerical integration"""

        try:
            R = 1.0 * inR
        except:
            return

        if R >= self.ModelKingRt:
            return 0.0

        # cannot integrate all the way out to infinity. Instead
        # integrate out to tidal radius where density falls to zero
        Integ = quad (lambda r: self.KingNumberDens(r) \
                          * r / N.sqrt(r**2 - R**2), \
                          R, self.ModelKingRt)

        SurfDens = 2.0 * Integ[0]
        return SurfDens

    def FitKingMIsotropic(self):

        """Given an isotropic king model for masses, fit with a
        polynomial to turn the integral into an evaluation. Also
        report the residuals"""

        if not N.any(self.KingMassIsoEval):
            return

        # find the fit coefficients
        self.KingPolyCoeffs = N.polyfit(self.R2D, \
                                            self.KingMassIsoEval, \
                                            self.KingPolyDegree)

        # Evaluate the polynomial
        self.KingMassIsoPoly = N.polyval(self.KingPolyCoeffs, self.R2D)
        
        # and the residuals
        self.KingMassIsoResids = (self.KingMassIsoPoly - self.KingMassIsoEval) \
            / self.KingMassIsoEval


    def ArrayKingMIsotropic(self):

        """Evaluate M(<r) for isotropic King profile"""

        # we are doing this to fit spline function to integrands, so
        # must use fine grid

        NRows = N.size(self.rfine)
        MassIso = N.zeros(NRows)
        for iRow in range(NRows):
            dum = self.PointKingMIsotropic(self.rfine[iRow])
            MassIso[iRow] = N.copy(dum)
            
        self.KingMassIsoEval = N.copy(MassIso)

    def PointKingMIsotropic(self, inr):

        """Evaluate the isotropic version of the enclosed-mass for
        King profile"""

        try:
            r = 1.0 * inr
        except:
            return

        Integ = quad(lambda x: x**2 * self.KingNumberDens(x), 0.0, r)
        return 4.0 * N.pi * Integ[0]

    def PointKingNumMassIsotropic(self, inr = None):

        """Evaluate the integral in equation 22 of LM89"""

        try:
            r = inr * 1.0
        except:
            return 0.0

        if r > self.ModelKingRt:
            return 0.0

        # "RInner" and "ROuter" refer to variables in the order in
        # which they appear in the double-integral
        #Integral = dblquad( lambda XIn, XOut: \
        #                        (self.KingNumberDens(XOut) / XOut**2) * \
        #                        (4.0 * N.pi * XIn**2 * \
        #                             self.KingNumberDens(XIn)), \
        #                        r, self.ModelKingRt, \
        #                        lambda y:0, lambda y:r)

        Integral = quad( lambda r: self.KingNumberDens(r)/r**2 * \
                             N.polyval(self.KingPolyCoeffs, r), \
                             r, self.ModelKingRt)
        

        return Integral

    def ArrayGMSquareBracket(self, inradii = None, DoPlot = False):

        """Expression for GM(<r) has a term [(1-Nr). A^{-1} . dA/dr +
        Nr * f /y]. Evaluate this term."""

        if not self.RadiusIsSet(inradii):
            return None

        # compute the various parts...
        
        # (1-Nr). A^{-1} . dA/dr 
        DensDeriv = self.IsoNumDens_DerivByFunc(inradii, DoPlot = True)

        # integrand of LM89 Eq22 and its integral, evaluated as splines
        y = interpolate.splev(inradii, self.tck_y)
        f = interpolate.splev(inradii, self.tck_f)

        # square bracket term
        self.GMSquareBracket = DensDeriv + self.ModelAnisoNr * f / y
        
        if not DoPlot:
            return
        
        self.FreeFigure()
        P.figure()
        P.subplot(221)
        P.plot(inradii, DensDeriv, 'k.')
        P.subplot(222)
        P.plot(inradii, y, 'k.')

        P.subplot(223)
        P.plot(inradii, f, 'k')

        # have to fill in the bits...

        # NB this will have to be gridded up anyway to evaluate the
        # volume density. Do it both ways; gridded first, then the
        # fancy way with the algebra

    def MaxFinite(self,x):

        """Utility: return the maximum value of x where x is finite"""

        Goods = N.where(N.isfinite(x))[0]
        if N.size(Goods) < 1:
            return None

        return N.max(x[Goods])

    def ArrayNR2(self, DoPlot = False):


        """GM(<r) contains the expression d/dr (n s^2_r). Evaluate
        this expression for a range of points"""

        # velocity dispersions
        self.ArraySigIsoFromSpline()
        self.ArraySigRadFromSpline()
        self.ArraySigTanFromSpline()

        IsoNumDens = self.IsoNumDens_Func(self.rfine) 

        # renormalize both to 1 
        NormIso = self.MaxFinite(IsoNumDens)
        NormSig = self.MaxFinite(self.VDispFromSplineRad)

        IsoNumDens = IsoNumDens / NormIso
        self.VDispFromSplineRad = self.VDispFromSplineRad / NormSig

        # here, dispersion is already squared (from LM89 Eq22)
        NS2 = IsoNumDens * self.VDispFromSplineRad

        Goods = N.where(N.isfinite(NS2) > 0)[0]
        
        # evaluate as spline and deriv
        tck_NS2 = interpolate.splrep(self.rfine[Goods], NS2[Goods], s=0)
        der_NS2 = interpolate.splev(self.rfine, tck_NS2, der = 1)
        eva_NS2 = interpolate.splev(self.rfine, tck_NS2, der = 0)

        print N.min(der_NS2), N.max(der_NS2)

        if not DoPlot:
            return

#        self.FreeFigure()
#        P.figure()
        P.subplots_adjust(hspace=0.3, wspace=0.4)
        P.subplot(211)
        P.rc('text', usetex=True)
        fsz=14
#        P.plot(self.rfine[Goods], NS2[Goods], self.LineSymbol)
        ytoplot = N.hstack((N.sqrt(self.VDispFromSplineRad[Goods]), 0.0))
        xtoplot = N.hstack((self.rfine[Goods], 10.0))
#        P.plot(self.rfine[Goods], N.sqrt(self.VDispFromSplineRad[Goods]), self.LineSymbol)
#        P.semilogy(self.rfine, eva_NS2, 'g.')
        P.plot(xtoplot,ytoplot, self.LineSymbol)
        P.ylabel(r"$\sigma_r$", fontsize=fsz)
        P.title(r"King Profile Velocity dispersion for various $N_r$", fontsize=fsz+2)
        P.xlabel(r"$R/R_c~(R_t = 10 R_c)$", fontsize=fsz)
        P.axis([-0.1, 10.5, -0.1, 1.1])

#        P.subplot(222)
#        P.semilogy(self.rfine, N.abs(der_NS2), 'k')
#        P.semilogy(self.rfine, self.rfine**2 / IsoNumDens, 'b--')

        Jeans2Plot = 0 - (der_NS2[Goods] \
                              * self.rfine[Goods]**2) / IsoNumDens[Goods] \
                              - 2.0*self.rfine[Goods] \
                              * self.VDispFromSplineRad[Goods]
        JeansLabel = r"$- \frac{r^2}{n} \frac{d}{dr}\left\{n \sigma_r^2\right\} - 2 r \sigma^2_r$"

        Jeans2Plot = 0 - (der_NS2[Goods] \
                              * self.rfine[Goods]**2) / IsoNumDens[Goods] \
                              * 1.0/11.15
        JeansLabel = r"$- \frac{r^2}{n} \frac{d}{dr}\left\{n \sigma_r^2\right\}$"

        P.subplot(223)
#        P.plot(self.rfine, 0 - (der_NS2 * self.rfine**2) / IsoNumDens - 2.0*self.rfine * self.VDispFromSplineRad, \
#                   self.LineSymbol)
        P.plot(self.rfine[Goods], Jeans2Plot, self.LineSymbol)
#        P.ylabel(r"$ - \frac{r^2}{n} \frac{d}{dr}\left\{n \sigma_r^2\right\}$" \
#                     , fontsize=fsz)
        P.ylabel(JeansLabel, fontsize=fsz)
        P.xlabel(r"$R/R_c~(R_t = 10 R_c)$", fontsize=fsz)
#        P.title('First term in Jeans')
        P.title(JeansLabel, fontsize=fsz+2)
        P.axis([0.0, 10.0, -0.1, 2.0])

        P.subplot(224)
#        P.semilogy(self.rfine, N.abs(der_NS2 * self.rfine**2) / IsoNumDens, \
#                       self.LineSymbol)
        P.semilogy(self.rfine[Goods], Jeans2Plot, self.LineSymbol)
       
#        P.ylabel(r"$ abs(\frac{r^2}{n} \frac{d}{dr}\left\{n \sigma_r^2\right\})$", fontsize=fsz)
        
        P.ylabel(JeansLabel, fontsize=fsz)
        P.xlabel(r"$R/R_c~(R_t = 10 R_c)$", fontsize=fsz)
        
#        P.ylabel("r^2 / n * d/dr(n sigma_r^2)")
#        P.xlabel("R/Rc (Rt = 10 Rc)")
#        P.title('First term in Jeans')
        P.title(JeansLabel, fontsize=fsz+2)
        P.axis([0.0, 10.0, 1e-3, 1e3])

#        P.subplot(222)
#        P.plot(self.rfine, self.VDispFromSplineRad, 'g')
#        P.title('VDisp_r')

#        P.subplot(223)
#        P.plot(self.rfine, self.VDispFromSplineRad * IsoNumDens, 'r')

#        P.subplot(224)
#        P.plot(self.rfine, - 2.0 * self.rfine * self.VDispFromSplineRad - der_NS2 * self.rfine**2 / IsoNumDens)
        return

        P.subplot(223)
#        P.plot(self.rfine, NS2, 'r')
        P.semilogy(self.rfine, N.abs(der_NS2), 'g.')
        P.title('d/dr (ns_r^2)')

#        P.subplot(223)
#        P.plot(self.rfine, self.VDispFromSplineRad / IsoNumDens , 'b.')
        
        # evaluate y**alpha
        self.tev_y = interpolate.splev(self.rfine, self.tck_y, der = 0)

        # plot da/dr / a numerically...
        A = self.IsoNumDens_Func(self.rfine)
        Goods = N.where(N.isfinite(A) > 0)[0]
        tck = interpolate.splrep(self.rfine[Goods], A[Goods])
        eva = interpolate.splev(self.rfine, tck, der = 0)
        der = interpolate.splev(self.rfine, tck, der = 1)

        P.subplot(223)
        P.cla()
#        P.plot(self.rfine, eva, 'b.')
        P.plot(self.rfine, self.IsoNumDens_Deriv(self.rfine),'r')
        P.plot(self.rfine, der, 'g')

#        P.plot(self.rfine, der / self.IsoNumDens_Deriv(self.rfine))

        P.subplot(224)
        P.semilogy(self.rfine, eva, 'r')
        P.semilogy(self.rfine[Goods],N.abs(der[Goods]), 'g')
        P.plot(self.rfine, N.abs(self.IsoNumDens_Deriv(self.rfine)), 'k')
        P.semilogy(self.rfine, 0.0 - der / eva, 'b')

        return
        P.subplot(224)
        Nr = self.ModelAnisoNr

        # compute the deriv of density wrt r
        IsoDensDeriv = self.IsoNumDens_Deriv(self.rfine)

#        P.plot(self.rfine, self.tev_y**Nr * IsoNumDens**(-1.0 - Nr))
        P.loglog(self.rfine, IsoDensDeriv)

        return
        P.semilogy(self.rfine, 1.0 / IsoNumDens)
        P.semilogy(self.rfine, N.abs(der_NS2))

    def ArraySigIsoFromSpline(self):

        """Given a spline representation for the integral in LM89
        Eq22, evaluate the radial velocity dispersion at a range of
        points."""

        IsoNumDens   = self.IsoNumDens_Func(self.rfine)
        SigRIntegral = interpolate.splev(self.rfine, self.tck_y, der = 0)

        self.VDispFromSplineIso = SigRIntegral / IsoNumDens
        
    def ArraySigRadFromSpline(self):

        """Given a profile of isotropic velocity dispersion, calculate
        the radial dispersion"""

        self.VDispFromSplineRad = self.VDispFromSplineIso**self.ModelAnisoNr

    def ArraySigTanFromSpline(self):

        """Given a profile of isotropic velocity dispersion with
        radius, calculate the radial dispersion"""

        self.VDispFromSplineTan = self.VDispFromSplineIso**self.ModelAnisoNt

    def Eq22RepAsSpline(self, inr = None, DoPlot = False):

        """Represent the integral in Eq22 of LM89 as a spline"""
        
        # set up the radius vector
        try:
            r = 1.0 * inr
        except:
            try:
                r = 1.0 * self.rfine
            except:
                return

        # evaluate the integral at rfine points
        Eq22Integral = self.Eq22EvaluateIntegArray(r)

        # represent this with spline
        self.tck_y = interpolate.splrep(r, Eq22Integral, s=0)

        if DoPlot:
            self.FreeFigure()
            P.figure()
            P.plot(r, Eq22Integral, 'ko')
            P.plot(r, interpolate.splev(r, self.tck_y, der = 0), 'r-')

    def IndicesRadiiWithinTidal(self, inr = None):

        """For a vector of radii, find the indices of the objects
        within the King tidal radius Rt.

        Returns a vector of indices. Return value will be zero-length
        if none of the input values (vector or scalar) are within the
        tidal radius."""

        # also return whether the input was vector or not
        InputIsVector = False
    
        try:
            r = 1.0 * inr
        except:
            return r, False, InputIsVector

        GoodObjs = N.array([])

        # ensure r is vectorized
        try:
            dum = r[0]
            InputIsVector = True
        except:
            InputIsVector = False
            r = N.array([r])

        # initialise return vector
        RetVec = N.zeros(N.size(r))

        # find good objects
        GoodObjs = N.asarray(N.where(r < self.ModelKingRt)[0], 'int')
        
        # return the indices
        return r, GoodObjs, InputIsVector

    def RadiusIsSet(self, inr):
        
        """Utility: is radius vector set to something tht can be
        operated on?"""

        try:
            dum = 1.0 * inr
            return True
        except:
            return False

    def IsoNumDens_DerivByFunc(self, inr = None, DoPlot = True):

        """In the expression for GM(<r), the term (1-Nr) * dA/dr * 1/A appears, where

        n(r) = n_0.A(r). 

        Evaluate the term dA/dr * 1/A"""

        try:
            r = 1.0 * inr
        except:
            return None

        # if Nr = 1 then this term will be zero. Return it.
        if N.abs(self.ModelAnisoNr - 1.0) < 1.0e-4:
            return r * 0.0

        # calculate x
        x = self.KingComputeX(r)

#        NumerPart1 = x*(1.0 - x**2)

        # now evaluate the term
#        NumerPart1 = x*(x**2 - 1.0) / N.sqrt(1.0 - x**2)
#        NumerPart2 = 0.0 - 3.0*N.arccos(x)
#        NumerPart3 = 2.0 * x * N.sqrt(1.0 - x**2)
#        Numer = NumerPart1 + NumerPart2 + NumerPart3

#        Denom = N.arccos(x) - x*N.sqrt(1.0 - x**2)
        
#        Frac = Numer / Denom

#        Numer = N.sqrt(1.0 - x**2) - 3.0 * N.arccos(x) / x
 #       Denom = N.arccos(x) - x * N.sqrt(1.0 - x**2)
 #       Frac = Numer / Denom

        # now evaluate dx / dr
#        dxdr = self.KingComputeXDeriv(r)

        Rc = self.ModelKingRc
        Rt = self.ModelKingRt
#        K = 1.0 / (Rc**2 + Rt**2)

#        NumerPart1 = x*(1.0 - x**2)
#        NumerPart2 = -3.0 * N.sqrt(1.0 - x**2) * N.arccos(x)
#        NumerPart3 = N.sqrt(x**2 * K - Rc**2)

#        Denom = N.sqrt(1.0 - x**2)*N.arccos(x) - x*(1.0 - x**2)

#        Numer = (NumerPart1 + NumerPart2) * NumerPart3


        Numer  = x*N.sqrt(1.0 - x**2) - 3.0 * N.arccos(x)
        Denom  = x*N.sqrt(1.0 - x**2) - 1.0 * N.arccos(x)

        Frac1 = 0.0 - Numer / Denom

        Frac2 = 1.0 / (Rc * N.sqrt(x**2 - 1.0 + (Rt/Rc)**2))

        if DoPlot:
            self.FreeFigure()
            P.figure()
            P.subplot(311)
            P.plot(x, Frac1)
            P.subplot(312)
            P.plot(x, Frac2)
            P.subplot(313)
            P.plot(x, Frac1 * Frac2)

        return Frac

    def IsoNumDens_Deriv(self, inr = None):

        """Given Rc and Rt for King profile, evaluate the derivative dA(r) / dr, where:

        n(r) = n_0 A(r) 

        is the number density for the isotropic King profile (King 1962)"""

        try:
            r = 1.0 * inr
        except:
            return 0.0

        # get inr as a vector, find indices within King tidal radius
        rVec, GoodInds, InputIsVector = self.IndicesRadiiWithinTidal(inr)

        # initialise return vector
        RetVec = rVec * 0.0

        if N.size(GoodInds) < 1:
            return RetVec

        # OK now do the calculation
        x = self.KingComputeX(rVec[GoodInds])
        
        # function has two parts: dA / dX and dX / dr
#        part1 = (1 + x**2) / N.sqrt(1.0 - x**2) \
#            + 2.0*N.sqrt(1.0 - x**2) \
#            + 3.0 * N.arccos(x) / x
#        part1 = part1 / x**3

        Numer = N.sqrt(1.0 - x**2) - 3.0 * N.arccos(x) / x
        Denom = x**3

        part2 = self.KingComputeXDeriv(x)

        RetVec[GoodInds] = Numer * part2 / Denom

        # if the callign routine passed a scalar for r, return one back.
        if not InputIsVector:
            RetVec = RetVec[0]

        return RetVec

    def IsoNumDens_Func(self, inr = None):

        """Given Rc and Rt for King profile, evaluate the function A(r), where:

        n(r) = n_0 A(r) 

        is the number density for the isotropic King profile (King 1962)"""

        try:
            r = 1.0 * inr
        except:
            return 0.0

        # find indices of radii within king tidal radii
        rvec, GoodIndices, IsVector = self.IndicesRadiiWithinTidal(inr)

        if N.size(GoodIndices) < 1:
            return inr * 0.0  # will work for vectors and scalars

        # now do the calculation. First compute x for good objects only
        x      = self.KingComputeX(rvec[GoodIndices])
        part1  = 1.0 / x**2
        part2  = N.arccos(x) / x - N.sqrt(1.0 - x**2)

        RetVec = rvec * 0.0
        RetVec[GoodIndices] = part1 * part2

        print N.size(GoodIndices)

        # if scalar was passed, return a scalar.
        if not IsVector:
            RetVec = RetVec[0]

        return RetVec
        
            
    def Eq22EvaluateArray(self, inr=None):

        """Evaluate eq22 of LM89 for all points in a vector of measurements"""

        # estimate the integrand for all points requested. Note that
        # inr here can be much less finely sampled than the
        # spline-fitting grid if desired.
        Integral = self.Eq22EvaluateIntegArray(inr)

        # now that's done, evaluate n(r) and divide by it
        NumDens = self.KingNumberDens(inr)

        self.KingDispIso = Integral / NumDens

    def Eq22EvaluateIntegArray(self, inr = None):
        
        """Evaluate the integral in Eq22 of LM89"""

        if not self.RadiusIsSet(inr):
            return None
        r = N.copy(inr)

        nRows = N.size(r)
        Integrand = N.zeros(nRows)
        for iRow in range(nRows):
            ThisInteg = self.Eq22EvaluatePoint(r[iRow])
            Integrand[iRow] = N.copy(ThisInteg)

        return Integrand
        
    def Eq22EvaluatePoint(self, inr=None):

        """Evaluate integral_{r}^{+inf} n(r) M( <r ) / r^2 at point r"""

        try:
            r = 1.0 * inr
        except:
            return 0.0

        # ensure the knots are present
        if len(self.tck_f) < 1:
            self.Eq22FitSpline()

        # integrate them
        ThisInteg = interpolate.splint(r, self.ModelKingRt, self.tck_f)

        return ThisInteg

    def Eq22FitSpline(self):

        """Evaluate equation 22 of LM89"""

        # use the fine-grid of samples to get this right

        Integrand = self.EvalEq22Integrand()
        self.tck_f = interpolate.splrep(self.rfine, Integrand, s=0)

    def EvalEq22Integrand(self):

        """Evaluate the integrand in eq 22 of LM89"""

        # replaced to use the fine grid
        Integrand = self.KingNumberDens(self.rfine) * \
            self.KingMassIsoEval / self.rfine**2

        return Integrand

    def ArrayKingSigmaIso(self):

        """Evaluate isotropic sigma for range of radii"""

        NumRows = N.size(self.r3d)
        EvalSigsqIso = N.zeros(NumRows)
        for iRow in range(NumRows):
            ThisDisp = self.PointKingSigmaIso(self.r3d[iRow])
            EvalSigsqIso[iRow] = N.copy(ThisDisp)

        self.EvalKingSigmaIso = N.copy(EvalSigsqIso)

    def PointKingSigmaIso(self, inr = None):

        """Return sigma_iso**2 for king model"""

        try:
            r = inr * 1.0
        except:
            return 0.0

        if r > self.ModelKingRt:
            return 0.0

        # evaluate the 1/r**2 n(r) M(<r) integral
        Integ = self.PointKingNumMassIsotropic(r)
        Disp = Integ[0] 
        numdens = self.KingNumberDens(r)
        
        return Disp / numdens

#    def BPVolumeDensity(self, P, r):

#        """Broken power-law function from Schoedel et al. (2009)"""
#        u = r / P[1]

#       return P[0] * u**(0.0 - P[2]) * (1.0 + u)**(P[2] - P[3])

    def BPVolumeDens(self, r):

        """Broken power-law density profile"""

        u = r / self.ModelR0
        part1  = self.ModelNorm * u**(0.0 - self.ModelGamma)
        part2  = (1.0 + u)**(self.ModelGamma - self.ModelA)

        return part1 * part2

    def PlummerSurfDens(self):

        """Evaluate surface density from plummer law"""
        
        u = 1.0 + (self.R2D / self.ModelR0Plummer)**2
        y = self.ModelNorm * u**(0.0-2)

        return y

    def KingNumberDens(self, inR = None):

        """Function giving surface density at radius r"""

        try:
            r = 1.0 * inR
        except:
            return

        Rc = self.ModelKingRc
        Rt = self.ModelKingRt
        S0 = self.ModelKingS0

#        x = ( (1.0 + (r/Rc)**2) / (1.0 + (Rc/Rt)**2) )**0.5
        x = self.KingComputeX(r)            

        Constant = S0 / (N.pi * Rc * (1.0 + (Rt/Rc)**2)**1.5)
        part1 = 1.0 / x**2
        part2 = (N.arccos(x) / x - N.sqrt(1.0-x**2))

        return Constant * part1 * part2
    
    def KingComputeX(self, r):

        """King (1962) defines new variable x(r, Rc, Rt). Evaluate it"""

        Rc = self.ModelKingRc
        Rt = self.ModelKingRt

        x = ( (1.0 + (r/Rc)**2) / (1.0 + (Rt/Rc)**2) )**0.5

        return x

    def KingComputeXDeriv(self, r):

        """King(1962) defines new variable x(r, Rc, Rt). Evaluate its
        derivative wrt r"""

        Rc = self.ModelKingRc
        Rt = self.ModelKingRt
        
        dxdr = r / ( N.sqrt(Rc**2 + r**2) * N.sqrt(Rc**2 + Rt**2) )
        return dxdr

    def ShowKingProfile(self, WantDispProfile = False):

        self.EvalKingSurfDens()
        self.ArrayKingSurfDens()
        self.ArrayKingMIsotropic()
        self.Eq22EvaluateArray(self.R2D)
        self.Eq22RepAsSpline(DoPlot = False)
        if not WantDispProfile:
            self.ArrayNR2(DoPlot = True)
            return
        
 #       return
 #       self.ArrayGMSquareBracket(self.rfine, DoPlot = False)
        
 #       return

#        self.FitKingMIsotropic()

        # now that the isotropic mass and the surface density are
        # computed, evaluate the integrand in LM89 Eq22
        Integrand = self.EvalEq22Integrand()

#        self.ArrayKingSigmaIso()

        # try fitting the enclosed mass with a polynomial
#        coeffs = N.polyfit(self.R2D, self.KingMassIsoEval, degree)

#        self.KingNumDens = self.KingNumberDens(self.R2D)

        xval = self.KingComputeX(self.R2D)

        self.FreeFigure()
        P.figure()

        P.subplot(221)
        P.loglog(self.R2D, self.KingSurfDensEval, 'k.')
        P.loglog(self.R2D, self.KingSurfDens, 'g-.')

        P.subplot(222)
        P.loglog(self.rfine, self.KingMassIsoEval, 'k.')

        P.subplot(223)
        P.plot(self.rfine, Integrand, 'g')
        P.plot(self.rfine, Integrand, 'g.')

        # try a spline representation
        tck = interpolate.splrep(self.rfine, Integrand, s=0)
        tev = interpolate.splev(self.rfine, tck, der = 0)

        xfine = N.linspace(0, 10.0, 1000)
        efine = interpolate.splev(xfine, tck, der = 0)

#        P.plot(self.rfine, tev, 'r.')
        P.plot(xfine, efine, 'r')
        P.subplot(224)
#        P.semilogx(self.rfine, N.abs((Integrand - tev)/tev), 'r-')

        P.subplot(224)

        self.FreeFigure()
        P.figure(figsize=(8,6))
        P.subplot(111)
        Goods = N.where(N.isfinite(self.KingDispIso) > 0)[0]
        MaxVal = N.max(self.KingDispIso[Goods])
        self.KingDispIso = self.KingDispIso / MaxVal
        P.plot(self.R2D[Goods], N.sqrt(self.KingDispIso[Goods]), 'k')
#        P.plot(self.R2D, N.sqrt(self.KingDispIso), 'k.')
        for ThisPow in [8.0, 4.0, 2.0, 0.5, 0.25, 0.125, 1.0]:
            isoplot = N.sqrt(self.KingDispIso[Goods])**ThisPow
            rplot = N.hstack((self.R2D[Goods], 10.0))
            isoplot = N.hstack((isoplot, 0.0))
            print isoplot[-2], rplot[-2]
            pform = 'r-'
            thick = 1
            if N.abs(ThisPow - 1.0) < 0.1:
                pform='k-'
                thick = 2
            P.plot(rplot, isoplot, pform,linewidth = thick)
#            P.plot(self.R2D, N.sqrt(self.KingDispIso)**(ThisPow), 'r-')
        
        plotax = N.copy(P.axis())
        plotax[0] = -0.1
        plotax[1] = self.ModelKingRt + 0.2
        plotax[2] = -0.2
        plotax[3] = 1.2
        P.axis(plotax)

        P.xlabel("R/Rc (Rt = 10 Rc)")
        P.ylabel("Sigma_r, arbitrary units")
        P.title("Dispersion profiles of LM89")

        P.savefig('LM89_KingProfiles.png')

        return

#        P.loglog(self.R2D, N.polyval(coeffs, self.R2D), 'r-')
#        P.loglog(self.R2D, self.KingMassIsoPoly, 'r-')
       
#        P.subplot(224)
#        P.plot(self.R2D, self.KingMassIsoEval, 'k.')                
#        P.plot(self.R2D, self.KingMassIsoPoly, 'r-')
#        P.plot(self.R2D, N.polyval(coeffs, self.R2D), 'r-')

#        plotax = N.copy(P.axis())
#        plotax[0] = 0.0
#        plotax[1] = 1.0
#        plotax[2] = 0.0
#        plotax[3] = 2.0
#        P.axis(plotax)

        P.subplot(224)

        P.loglog(self.R2D, N.abs(self.KingMassIsoResids), 'r.')
        return

        P.subplot(223)

        GoodVals = N.where( (N.isfinite(self.EvalKingSigmaIso) > 0) \
                                & (self.R2D > 0.1))[0]
        maxval = N.max(self.EvalKingSigmaIso[GoodVals])

        self.EvalKingSigmaIso = self.EvalKingSigmaIso / maxval

        P.plot(self.R2D, self.EvalKingSigmaIso, 'k')
        for ThisPow in [8.0, 4.0, 2.0, 0.5, 0.25, 0.125]:
            P.plot(self.R2D, self.EvalKingSigmaIso**(ThisPow), 'g')

        P.axis([N.min(self.R2D), N.max(self.R2D), 0.0, 1.01])

    def ShowSurfDens(self):

        """Wrapper - compute and show surface density for broken power
        law"""

        self.ArrayIntegrateBP()
        self.FreeFigure()
        P.figure()
        P.subplot(211)
        P.loglog(self.R2D, self.BPVolumeDens(self.R2D), 'k.')

        P.subplot(212)
        P.loglog(self.R2D, self.ModelEnclosedMass2D, 'k.')
#        P.loglog(self.R2D, self.PlummerSurfDens(), 'g.')

    def FreeFigure(self):
        
        try:
            P.close()
        except:
            dum = 1
    

def go(R0 = 1.0):

    A = King()
    A.ModelR0Plummer = R0
    A.ShowSurfDens()

def TestKingProfile(degree = 13, Nr = 1.0, psym = 'k-', Clobber=False):

    A = King()
    A.KingPolyDegree = degree
    A.ModelAnisoNr = Nr
    A.LineSymbol = psym
    if Clobber:
        A.FreeFigure()
        P.figure(figsize=(6,9))
    A.ShowKingProfile()

def DrawJeansEquation(DoEPS = True):

    Nrs  = [1.0, 1.1, 1.5, 1.9, 0.9, 0.5, 0.1]
    Syms = ['k-', 'b--', 'b--', 'b--', 'r-.', 'r-.', 'r-.']

    Nrs   = [1.0, 8.0, 4.0, 2.0, 0.5, 0.25, 0.125, 0.0]
    Syms  = ['k-', 'b--', 'b--', 'b--', 'r-.', 'r-.', 'r-.', 'k-.']

    TestKingProfile(1, Nrs[0], Syms[0], Clobber=True)
    for i in range(1,len(Nrs)):
        TestKingProfile(1, Nrs[i], Syms[i], Clobber=False)

    if not DoEPS:
        P.savefig('JeansEqKing.png')
    else:
        P.savefig('JeansEqKing.eps', format="eps_cmyk")
