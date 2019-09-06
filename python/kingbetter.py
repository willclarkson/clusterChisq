#
# kingbetter.py
# 
# Produce projected-motion curves, enclosed mass curves for King
# profiles.

import numpy as N
import pylab as P
from matplotlib import rc
import os
from scipy import interpolate
from scipy.integrate import quad, Inf
import time
import pyfits
import mpl_toolkits.ps_cmyk
try:
    import atpy
except:
    print "King_better WARN - cannot import atpy"
import glob

class King(object):

    """Contains king profiles"""

    def __init__(self):

        # points at which to evaluate the projections
        self.R2D = N.linspace(0.0, 10.0, 20)

        # model parameters
        self.ModelRc   = 0.10
        self.ModelRt   = 5.0 
        self.ModelMass = 3.1e4   # solar mass

        self.rMin  = 0.002
        self.rMax  = self.ModelRt
        self.rNum  = 2500 # was 2500
#        self.rNum = 200
        self.rfine = N.linspace(self.rMin, self.rMax, self.rNum, endpoint=True)
        
        # normalization constant for density
        self.Rho0 = 1.0

        # central density in physical units
        self.Rho0Phys = 0.0

        # evaluated curves
        #
        # EvalVolumeDensityFunc = volume density without rho_0
        self.EvalKingX = N.array([])
        self.EvalVolumeDensityFunc = N.array([])  
        self.EvalVolumeDensity = N.array([])

        # spline representation of r^2 . rho(r) - knots, integral
        self.R2RhoTCK = N.array([])
        self.R2RhoInt = N.array([])

        # velocity dispersion - integrand and integral
        self.VDispTCK = N.array([])
        self.VDispInt = N.array([])

        # spline-representation of velocity dispersion
        self.VDispShape = N.array([])
        self.VDispShapeR = N.array([])
        self.VDispShapeT = N.array([])

        self.EvalEnclosedMass = N.array([])

        self.EvalVDispFunc = N.array([])
        self.EvalVDisp = N.array([])
        self.EvalVSigma = N.array([])

        # holding arrays for transverse and radial surface densities
        self.EvalVDispR = N.array([])
        self.EvalVDispT = N.array([])

        # surface density
        self.EvalSurfDensNum = N.array([])
        self.EvalSurfDensAnalyticFunc = N.array([])
        self.EvalSurfDensAnalytic = N.array([])

        # grid for velocity dispersion integral
#        self.Num4Disp = 10000 # was 1000 10,000 keeps bias < 1%
        self.Num4Disp = 5000
        self.rgrid4disp = None
        self.rgrid4dispMaster = None
        self.VDispIntegrandShapeR = N.array([])
        self.VDispIntegrandShapeT = N.array([])

        # numerators in the dispersion projection integrals
        self.VProjNumeratorR = N.array([])
        self.VProjNumeratorT = N.array([])

        # evaluated projected dispersions
        self.Radii4Proj = N.array([])
        self.VDispProjectedR = N.array([])
        self.VDispProjectedT = N.array([])

        # evaluated surface density and scale factors
        self.SurfDensOptScale = 1.0
        self.SurfDensScaled = N.array([])

        # chisquare for fits
        self.ChisqSurfDens = None
        self.ChisqVDispRad = None
        self.ChisqVDispTan = None
        self.DChisq = {}

        # data
        self.SurfDensRProj = N.array([])
        self.SurfDensNum = N.array([])
        self.SurfDensErr = N.array([])

        self.R2D = N.array([])
        self.r3d = N.array([])
        self.DataVDispRad = N.array([])
        self.DataVDispTan = N.array([])
        self.DataEDispRad = N.array([])
        self.DataEDispTan = N.array([])

        # grids of parameters tried, chisq, mass evaluated
        self.NumRt   = 40
        self.NumRc   = 40
        self.NumMass = 50  # this last one should be vectorized
                           # out... come back if there is more time
                           # since this is not easy to accomplish.

        self.MinRc   = 0.05
        self.MinRt   = 1.0
        self.MinMass = 5.0e3

        self.MaxRc   = 0.8
        self.MaxRt   = 15.0
        self.MaxMass = 6.0e4

        # trial grids
        self.TrialVects = {}
        self.TrialGrids = {}
        self.TrialGridKeys = ['ChiR', 'ChiT', 'ChiS', \
                                  'Rho0Phys', 'MProj', 'M3d', \
                                  'Rc', 'Rt', 'Mass', \
                                  'SigScale', 'Rho0']
        self.TrialGridShape = ()

        # record array of trials - not used when simulating but
        # convenient when reading in
        self.TrialRecArray = N.array([])

        # delta chi levels for calculating parameter regions
        self.DeltaChiAssess  = [3.5, 7.82, 13.93]
        self.DeltaChiKeys = ["MProj", "M3d", "Rho0Phys", \
                                 "Rc", "Rt", "Mass", "SigScale"]

        # some strings for latex output table
        self.DeltaProbLevels = ["68\%", "95\%", "99.7\%" ]
        self.DeltaSigLevels  = ['"$1\sigma$"', '"$2\sigma$"', '"$3\sigma$"']
        self.KeysForOutTable = ["MProj", "M3d", "Rho0Phys", \
                                    "Rc", "Rt", "Mass", "SigScale"]

        # keys to report
        self.ReportKeys = ['MProj', 'M3d', 'Rho0Phys', 'Mass']
        self.ReportSig  = 2

        # derived parameter-ranges corresponding to ranges of chisq
        self.DChisqRanges = {}
        self.DChisqTables = {}

        # quantities of interest
        self.DInterestKeys = ['Rho0', 'MProj', 'M3d', 'Rho0Phys']
        self.DInterest = {}
        self.RFidMProj = 0.4
        self.RFidM3d  = 1.0

        # output filename
        self.TrialFits2D = 'King_iso_trials2d.fits'

        # variables to do with evaluating profiles given parameters
        # and overplotting them
        self.DEvalProfiles = {}
        self.DEvalKeys = ['Radii', 'MProj', 'M3d', 'SurfDens',\
                              'SigR', 'SigT', 'ChiTot', 'Rho0Phys']
        self.ShowEvalIndices = []
        self.ShowEvalJMin  = None
        self.ShowEvalJMax  = None
        self.ShowEvalAlpha = 0.3
        self.ShowEvalAsSpline = False
        self.ShowYLabels = {}
        self.ShowXLabel = None
        self.ShowTitles = []

        # validation file
        self.ValidationFile = 'tuan_dispersion_curve.txt'
        self.ValidationImg =  'dum.png'

        # some conversion factors
        self.NewtonG = 6.67e-11
        self.Parsec  = 3.0857e16
        self.MSol    = 2.0e30
        self.kms2    = 1.0e6

    def InitModel(self):

        """Initialise intermediate products"""
        self.Rho0 = 1.0
        self.Rho0Phys = 0.0
        self.EvalKingX = N.array([])
        self.EvalVolumeDensityFunc = N.array([])  
        self.EvalVolumeDensity = N.array([])

        self.R2RhoTCK = N.array([])
        self.R2RhoInt = N.array([])
        self.VDispTCK = N.array([])
        self.VDispInt = N.array([])
        self.VDispShape = N.array([])
        self.VDispShapeR = N.array([])
        self.VDispShapeT = N.array([])

        self.EvalEnclosedMass = N.array([])

        self.EvalVDispFunc = N.array([])
        self.EvalVDisp = N.array([])
        self.EvalVSigma = N.array([])

        self.EvalVDispR = N.array([])
        self.EvalVDispT = N.array([])

        self.EvalSurfDensNum = N.array([])
        self.EvalSurfDensAnalyticFunc = N.array([])
        self.EvalSurfDensAnalytic = N.array([])
        
        self.VDispIntegrandShapeR = N.array([])
        self.VDispIntegrandShapeT = N.array([])

#        self.Radii4Proj = N.array([])
        self.VDispProjectedR = N.array([])
        self.VDispProjectedT = N.array([])

        # quantities of interest, chisq
        self.SurfDensOptScale = 1.0
        self.SurfDensScaled = N.array([])

        # chisquare for fits
        self.ChisqSurfDens = None
        self.ChisqVDispRad = None
        self.ChisqVDispTan = None
        self.DChisq = {}

        # parameters of interest
        self.DInterest = {}

    def GetLogSpacedRadii(self):

        """Get log-spaced radii for evaluation"""

        logMin = N.log10(self.rMin)
        logMax = N.log10(self.ModelRt)
        Logspaces = N.linspace(logMin, logMax, self.rNum, endpoint=True)

        self.rfine = 10.0**Logspaces

    def GetLinSpacedRadii(self):

        """Set up linear-spaced radii for evaluating integrals"""

        self.rfine = N.linspace(self.rMin, self.ModelRt, self.rNum, endpoint=True)

    def GetRhoM3dSigma2(self):

        """Evaluate Rho(r), M(<r), sigma^2(r) for Rc, Rt
        combination"""

        # start time
        starttime = time.time()

        # initialise the model
        self.InitModel()

        # set up radii
        self.rfine = N.linspace(self.rMin, self.rMax, self.rNum, endpoint=True)

        # evaluate the form of rho, rho.M(<r) / r^2
        self.KingNumberDens(self.rfine)
        self.KingCalcRho0()

        # evaluate volume density in MSol / pc^3
        self.EvalVolumeDensity = self.EvalVolumeDensityFunc * self.Rho0


        # evaluate the form of M(<r) in order to integrate it
        self.ArrayR2RhoInt(self.rfine)

        # evaluate M(<r)
        self.EvalEnclosedMass = self.R2RhoInt * self.Rho0

        # evaluate 1/n \int (n M / r^2) dr
        self.VDispForm(self.rfine)

        # convert vdisp to vdisp in km/s
        self.VDispFromVDispFunc()

#        print "DBG -- ", N.size(self.Radii4Proj)

        # evaluate the surface density analytically
        self.SurfDensEvalFunction()

        # evaluate the enclosed mass analytically
        self.EvalEnclosed2DAnalytic = self.EnclosedMassAnalyticIso()

        # evaluate the surface density numerically
        #
        # this step is currently quite slow - takes 1.8 seconds for
        # 200 samples
        # 
        # self.SurfDensEvalNum()

        # set up fine grid of radii to evaluate the dispersion integrands
        if not N.any(self.rgrid4dispMaster):
            self.VDispGridForIntegrands()

        # now try projecting the dispersions onto the sky
#        print "Projecting dispersions onto the sky"
        self.VelProjEvaluate()

#        print "TIme elapsed: %.2f" % (time.time() - starttime) 

        # compute quantities of interest (for grids)
        self.ComputeInterestings()

#        print self.DInterest

    def ComputeInterestings(self):

        """Given model parameters, compute quantities of interest. The
        central density, projected mass within 0.4pc and the kinematic
        mass within 1 pc are computed"""
        
        self.DInterest = {}
        
        # first rho0, msol / pc^3
#        brac = (1.0 + (self.ModelRt / self.ModelRc)**2)
#        Constant2 = N.pi * self.ModelRc * brac**1.5

        self.DInterest['Rho0'] = N.copy(self.Rho0) #/ Constant2

        self.DInterest['Rho0Phys'] = N.copy(self.Rho0Phys)

        # if we are isotropic, the enclosed mass is easy to calculate:
        self.DInterest['MProj'] = self.EnclosedMassAnalyticIso(self.RFidMProj)

        # input radius for enclosed mass
        r4m3d = N.copy(self.RFidM3d)
        if r4m3d > self.ModelRt:
            r4m3d = self.ModelRt
            # we could just return the input mass here. However I
            # anticipate using sigma_0 in future.

        # evaluate enclosed mass at fiducial radius
        self.DInterest['M3d'] = self.Rho0 * \
            interpolate.splint(0.0, r4m3d, self.R2RhoTCK)

    def VDispFromVDispFunc(self):

        """Given Velocity dispersion functional form, convert it to
        velocity dispersion in km/s"""

        UnitConvert    = self.NewtonG * self.MSol / self.Parsec
        self.EvalVDisp = self.EvalVDispFunc * self.Rho0 * \
            UnitConvert / self.kms2
        self.EvalVSigma = N.sqrt(self.EvalVDisp)


    def KingCalcRho0(self):

        """Compute the constant rho_0 for king profiles"""

        # note that this is not quite the central density but instead
        # the cluster of constants up-front multiplied by
        # 4\pi. Calculate that quantity as well.

        # set rho0 to 1
        self.Rho0 = 1.0

        # set up the spline representation
        if len(self.R2RhoTCK) < 2:
            self.SetupR2RhoSpline(self.rfine)

        # evaluate it from r = 0 to tidal radius
        MEval = interpolate.splint(0.0, \
                                       self.ModelRt, self.R2RhoTCK)
        
#        print "RHO0 DEBUG:", self.ModelMass, MEval

        # now use this to find the normalizing constant
        self.Rho0 = self.ModelMass / MEval

        # now also compute the central density in physical units
        self.Rho0Phys = self.Rho0 * self.ArrayVolumeDensity(0.0)

    def EnsureR(self, inr = None):

        """Ensure input radii are set. Turn 4 lines into 1"""

        try:
            r = 1.0 * inr
        except:
            r = N.copy(self.rfine)
        
        if N.size(r) < 2:
            return r

        # return only those radii that are inside the tidal radius
        Goods = N.where(r <= self.ModelRt)[0]
        if N.size(Goods) < 1:
            return 0.0
        
        return r[Goods]

    def SetupR2RhoSpline(self, inr = None):

        """Set up the spline representation for the r^2 rho(r)
        integrand"""

        r = self.EnsureR(inr)

        # calculate r from x using grid of fine points
        self.KingComputeX(r)

        # calculate evaluated density without Rho0 upfront
        self.KingNumberDens(r)

        # compute the integrand
        #
        Integrand = 4.0 * N.pi * r**2 * self.EvalVolumeDensityFunc

#        print "R2RhoSpline DBG:", Integrand[0]

        Goods = N.where(N.isfinite(Integrand) > 0)[0]

#        print N.size(Goods), self.ModelRc, self.ModelRt, self.ModelMass
        if N.size(Goods) < 10:
            return

        # represent it with a spline
        self.R2RhoTCK = interpolate.splrep(r[Goods], \
                                               Integrand[Goods], s=0)
        
    def ArrayR2RhoInt(self, inr = None):

        """Calculate the quantity 4pi \int^{r}_{0} r^2 rho(r) dr"""

        r = self.EnsureR(inr)

        self.SetupR2RhoSpline(r)

        nrows = N.size(r)
        self.R2RhoInt = N.zeros(nrows)
        for irad in range(nrows):
            
            # don't do anything if the point is outside the tidal
            # radius
            if r[irad] >= self.ModelRt:
                continue

            self.R2RhoInt[irad] = \
                interpolate.splint(0.0, r[irad], self.R2RhoTCK)

    def VDispIntegrate(self, inr = None):

        """Carry out the integration of EQ22 of LM89, without the
        up-front constants"""

        r = self.EnsureR(inr)

        # spline-representation must actually be present for this to
        # work...
        if not N.any(self.VDispTCK):
            self.DispIntegrandAsSpline(r)

        nradii = N.size(r)
        self.VDispInt = N.zeros(nradii)
        for irad in range(nradii):
            self.VDispInt[irad] = \
                interpolate.splint(r[irad], self.ModelRt, self.VDispTCK)

    def VDispForm(self, inr = None):

        """Calculate functional form of sigma^2 for the isotropic case
        without the upfront constants"""

        r = self.EnsureR(inr)

        # ensure the velocity dispersion has been populated
        if not N.any(self.VDispInt):
            self.VDispIntegrate()

        # compute integrand / n
        self.EvalVDispFunc = self.VDispInt / self.EvalVolumeDensityFunc

    def FreeFigure(self):

        """Free up figure"""

        try:
            P.close()
        except:
            dum = 1

    def ShowVDisps(self):

        """Show velocity dispersion curves"""

        self.FreeFigure()
        P.figure(figsize=(9,9))
        P.subplots_adjust(hspace=0.4)
        P.subplot(211)
        P.semilogx(self.Radii4Proj, self.VDispProjectedR**0.5, 'k+')
#        P.semilogx(self.Radii4Proj, self.VDispProjectedT**0.5, 'bx')
        xt, yt = N.loadtxt(self.ValidationFile, unpack=True)
        P.plot(xt, yt, 'g')

        rc('text', usetex=True)
        P.xlabel(r"Radius (pc)")
        P.ylabel(r"$\sigma$, km s$^{-1}$")
        P.title(r"Comparison of dispersion predictions")

        RcString = r"$R_c$ = %.2f" % (self.ModelRc)
        RtString = r"$R_t$ = %.2f" % (self.ModelRt)
        MaString = r"$M_{cl}$ = %.2f $\times 10^4 M_{\odot}$" % (self.ModelMass / 1.0e4)

        YAnno = 0.4
        fsz = 14
        P.annotate(RcString, (0.05, YAnno - 0.0), xycoords='axes fraction', \
                       fontsize=fsz)
        P.annotate(RtString, (0.05, YAnno - 0.1), xycoords='axes fraction', \
                       fontsize=fsz)
        P.annotate(MaString, (0.05, YAnno - 0.2), xycoords='axes fraction', \
                       fontsize=fsz)


        if N.size(xt) != N.size(self.Radii4Proj):
            return
        
        P.subplot(212)
        P.semilogx(xt,100.0 * (self.VDispProjectedR**0.5 - yt)/yt, 'k+')
        P.xlabel(r"Radius (pc)")
        P.ylabel(r"$(\sigma_1 - \sigma_2)/\sigma_2$, percent")
        P.title(r"Residuals, percent")

        P.savefig(self.ValidationImg)

        rc('text', usetex=False)

    def ShowRhoM3dSigma2(self):

        """Show the curves for rho, M3d and sigma^2"""

        self.FreeFigure()

        P.figure(figsize=(6,7))
        P.subplot(221)

#        print N.size(self.EvalVolumeDensity), N.size(self.EvalVolumeDensityFunc)
        P.semilogx(self.rfine, self.EvalVolumeDensity, 'k.')

        P.subplot(222)
#        P.semilogx(self.rfine, interpolate.splev(self.rfine, self.R2RhoTCK), 'r')
        P.semilogx(self.rfine, self.EvalEnclosedMass, 'g.')
        
        P.subplot(223)
        P.semilogx(self.rfine, self.EvalVSigma, 'b')
        P.semilogx(self.rfine, interpolate.splev(self.rfine, self.VDispShapeR)**0.5, 'g.')

        # load tuan's plot to show comparison
        xt, yt = N.loadtxt(self.ValidationFile, unpack=True)

        P.subplot(224)
#        self.SurfDensEvalFunction()
#        try:
#            P.plot(self.rfine, self.EvalSurfDensNum * self.Rho0 * 2.0, 'r')
#        except:
#            dum = 1
#        P.plot(self.rfine, self.EvalSurfDensAnalytic, 'go')
        P.plot(self.rfine, self.EvalEnclosed2DAnalytic, 'g')

#        print N.shape(self.VDispProjectedR), N.shape(self.VDispProjectedT), \
#            N.shape(self.Radii4Proj)


        P.subplot(224)
        P.cla()


        
#        P.plot(self.rfine, self.EvalVDispR, 'k')
#        P.plot(self.rfine, interpolate.splev(self.rfine, self.VDispShapeR), 'g.')
#        print(self.VProjNumeratorR)
        P.semilogx(self.Radii4Proj, self.VDispProjectedR**0.5, 'k+')
        P.semilogx(self.Radii4Proj, self.VDispProjectedT**0.5, 'bx')
        P.plot(xt, yt, 'g')
#        P.semilogx(self.Radii4Proj, self.VProjNumeratorR)
#        P.semilogx(self.Radii4Proj, self.VProjNumeratorR, 'k')
#        P.semilogx(self.Radii4Proj, self.EvalSurfDensAnalytic/ self.Rho0)

#        P.plot(self.Radii4Proj,self.EvalSurfDensAnalytic, 'r')

        P.grid()

        return
        plotax = N.copy(P.axis())
        plotax[2] = 0.0
        plotax[1] = 10.0
        P.axis(plotax)

    def DispIntegrandAsSpline(self, inr=None):

        """Given enclosed-mass grid of points, represent it as a
        spline. Keep absolute units."""

        # parse input radius
        r = self.EnsureR(inr)

        # ensure the integrand is populated for a start
        if not N.any(self.R2RhoInt):
            self.ArrayR2RhoInt(r)

        # construct the dispersion integrand
        DispIntegrand = self.EvalVolumeDensityFunc * self.R2RhoInt / r**2
        
        # represent it with spline
        Finites = N.where(N.isfinite(DispIntegrand) > 0)[0]
        self.VDispTCK = interpolate.splrep(r[Finites], \
                                               DispIntegrand[Finites], s = 0)
        

    def KingNumberDens(self, inR = None, FindingRho0 = False):

        """Function giving volume density at radius r"""

        try:
            r = 1.0 * inR
        except:
            r = N.copy(self.rfine)

        Rc = self.ModelRc
        Rt = self.ModelRt
        Rho0 = self.Rho0

        if FindingRho0:
            Rho0 = 1.0

        if not N.any(self.EvalKingX):
            self.KingComputeX(r)            

#        # The constant will now be absorbed into rho_0    
#        Constant = Rho0 / (N.pi * Rc * (1.0 + (Rt/Rc)**2)**1.5)

        x = self.EvalKingX

#        print "DBG:", N.min(x), N.max(x), self.ModelRc, self.ModelRt

        Constant = 1.0
#        Constant = 1.0 / (N.pi * Rc * (1.0 + (Rt/Rc)**2)**1.5)
        part1 = 1.0 / x**2
        part2 = (N.arccos(x) / x - N.sqrt(1.0-x**2))

        ThisVolumeDensity = Constant * part1 * part2
        self.EvalVolumeDensityFunc = ThisVolumeDensity

#        print "Density DBG:", ThisVolumeDensity[0]
        # only pass this to volume density if Rho0 nonzero
        if self.Rho0 < 2:
            return

#        self.EvalVolumeDensity = self.EvalVolumeDensityFunc * self.Rho0

    def ArrayVolumeDensity(self, inR):

        """Function to return volume density for input r"""

        try:
            r = 1.0 * inR
        except:
            r = N.copy(self.rfine)
        
        Rc = self.ModelRc
        Rt = self.ModelRt
        Rho0 = self.Rho0
    
        x = self.KingComputeX(r, DoReturn = True)

        Constant = 1.0
        part1 = 1.0 / x**2
        part2 = (N.arccos(x) / x - N.sqrt(1.0-x**2))

        ThisVolumeDensity = Constant * part1 * part2
        return ThisVolumeDensity

    def SurfDensIntegrandNum(self, RProj = 0.0):

        """Compute the surface density integrand numerically. Don't
        include the central density (will cancel from projection)"""

        SDIntegrand = self.rfine * self.EvalVolumeDensityFunc \
            / N.sqrt(self.rfine**2 - RProj**2)

        return SDIntegrand

    def SurfDensPointEval(self, inr3d, inr2d):

        try:
            r3 = 1.0 * inr3d
            r2 = 1.0 * inr2d
        except:
            return 0.0

        # evaluate rho at this point
        RhoThis = self.ArrayVolumeDensity(r3)

        retval = r3 * RhoThis / N.sqrt(r3**2 - r2**2)
        return retval

    def SurfDensPointIntegrate(self, inR = None):

        """Conduct the surface density integral at a given point"""

        Integral = quad (lambda r: self.SurfDensPointEval(r, inR), \
                             inR, self.ModelRt)

        return Integral[0]

    def SurfDensEvalNum(self, inR = None):

        """Evaluate the surface density in absolute units"""

        # use either input R grid or fine-R grid
        R = self.EnsureR(inR)

        nR = N.size(R)
        self.EvalSurfDensNum = N.zeros([nR])

        for iR in range(nR):
            ThisIntegral = self.SurfDensPointIntegrate(R[iR])
            self.EvalSurfDensNum[iR] = N.copy(ThisIntegral)

    def SurfDensEvalFunction(self, inr = None):

        """Evaluate the surface density analytically in absolute units"""

        self.EvalSurfDensAnalyticFunc = self.EvalSurfDensFunc(inr)
        self.EvalSurfDensAnalytic = self.EvalSurfDensAnalyticFunc \
            * self.Rho0

    def EnclosedMassAnalyticIso(self, inr = None):

        """Evaluate the enclosed projected mass analytically"""

        r = self.EnsureR(inr)

        # king integrand variables
        x  = ( r / self.ModelRc )**2
        xt = ( self.ModelRt / self.ModelRc )**2

        part1 = N.log(1.0 + x)
        part2 = -4.0 * (N.sqrt(1.0 + x) - 1.0)/N.sqrt(1.0 + xt)
        part3 = x / (1.0 + xt)

        Constant1 = N.pi * self.ModelRc**2 * self.Rho0

        # don't forget that rho0 contains a number of
        # constants. Evaluate them out (slightly awkward: rho0 = k /
        # () and we want k)
        brac = (1.0 + (self.ModelRt / self.ModelRc)**2)
        Constant2 = N.pi * self.ModelRc * brac**1.5

        return Constant1*Constant2 * (part1 + part2 + part3)

    def EvalSurfDensFunc(self, inr):

        """Evaluate the analytic form of king profile surface
        density"""

        r = self.EnsureR(inr)

        brac = (1.0 + (self.ModelRt / self.ModelRc)**2)
        
        Constant = N.pi * self.ModelRc * brac**1.5

        Func = 1.0 / N.sqrt(1.0 + (r / self.ModelRc)**2) \
            - 1.0 / N.sqrt(1.0 + (self.ModelRt / self.ModelRc)**2)

        return Constant * (Func)**2
        
    def KingComputeX(self, inr, DoReturn = False):

        """King (1962) defines new variable x(r, Rc, Rt). Evaluate it"""

        try:
            r = 1.0 * inr
        except:
            r = N.copy(self.rfine)

        Rc = self.ModelRc
        Rt = self.ModelRt

        x = ( (1.0 + (r/Rc)**2) / (1.0 + (Rt/Rc)**2) )**0.5

        if DoReturn:
            return x

        self.EvalKingX = N.copy(x)

    def PopulateVDisps(self):

        """Populate dispersions in the r and t directions. For now,
        assume isotropic and pass them up"""

        self.EvalVDispR = N.copy(self.EvalVDisp)
        self.EvalVDispT = N.copy(self.EvalVDisp)

    def VDispsAsSplines(self):

        """Represent velocity dispersion components as splines"""
        
        g = N.where(N.isfinite(self.EvalVDispR) > 0)[0]
        self.VDispShapeR = interpolate.splrep(self.rfine[g], \
                                                  self.EvalVDispR[g], s=0)
        g = N.where(N.isfinite(self.EvalVDispT) > 0)[0]
        self.VDispShapeT = interpolate.splrep(self.rfine[g], \
                                                  self.EvalVDispT[g], s=0)

        self.VDispCheckSpline()

    def VDispCheckSpline(self):

        """In isotropic case, disp should not increase with distance"""

        g = N.where(N.isfinite(self.EvalVDispR) > 0)[0]
        DispDer = interpolate.splev(self.rfine[g], self.VDispShapeR, der = 1)
        lo = N.where(DispDer > 0)[0]
        if N.size(lo) < 1:
            return

        NextGood = lo[-1]+1
        self.EvalVDispR[lo] = self.EvalVDispR[NextGood]
        self.VDispShapeR = interpolate.splrep(self.rfine[g], \
                                                  self.EvalVDispR[g], s=0)

    def VDispGridForIntegrands(self):

        """Set up standard grid for evaluating the integrand"""

        self.rgrid4dispMaster = N.logspace(-3.0, \
                                               N.log10(self.ModelRt), \
                                               self.Num4Disp, endpoint=True)

#        self.rgrid4dispMaster = N.linspace(0.001, self.ModelRt, self.Num4Disp, #\
#                                               endpoint=True)

    def VDispSetupIntegrands(self, RProj = None):

        """Produce representations of both dispersion
        projection-integrands"""

        # set up grid of points over which to evaluate the integrand
        # try standardizing the grid for dispersions

#        self.rgrid4disp = N.linspace(RProj, self.ModelRt, \
#                                         self.Num4Disp, endpoint=True)

        self.rgrid4disp = N.logspace(N.log10(RProj), N.log10(self.ModelRt), \
                                         self.Num4Disp, endpoint=False)

        # don't want to divide by zero
        self.rgrid4disp = self.rgrid4disp[1::]

        # for teh dispersion grid, bring in only those points for
        # which rgrid > rproj
        Good4Proj = N.where(self.rgrid4dispMaster > RProj)[0]
        if N.size(Good4Proj) < 1:
            return

#        self.rgrid4disp = N.copy(self.rgrid4dispMaster[Good4Proj])
#        self.rgrid4disp = self.rgrid4disp[1::]

        # evaluate the upfront bracket part
        Part1 = self.VDispIntegrandFront(self.rgrid4disp, RProj)

        # return if there are no finite parts to this integral
        if N.size(N.where(N.isfinite(Part1) > 0)[0]) < 2:
            return
        
        DispEvalRad = interpolate.splev(self.rgrid4disp, self.VDispShapeR)
        DispEvalTan = interpolate.splev(self.rgrid4disp, self.VDispShapeT)
        
        Frac = (RProj / self.rgrid4disp)
        Part2 = Frac**2 * DispEvalRad + (1.0 - Frac**2)*DispEvalTan

        # evaluate the integrand along the grid
        IntegrandR = Part1 * Part2
        IntegrandT = Part1 * DispEvalTan

#        print "DBG:", Part2[50], DispEvalTan[50], Frac[50]

#        diffs = IntegrandR - IntegrandT
#        print "DBGDiffs:", N.min(diffs), N.max(diffs), IntegrandR[10] - IntegrandT[10]

        # represent each with splines
        g = N.where(N.isfinite(IntegrandR) > 0)[0]
        self.VProjTCKR = interpolate.splrep(self.rgrid4disp[g], \
                                                IntegrandR[g], s=0)
        h = N.where(N.isfinite(IntegrandT) > 0)[0]
        self.VProjTCKT = interpolate.splrep(self.rgrid4disp[h], \
                                                IntegrandT[h], s=0)
        
    def VelProjSplineInitialize(self):

        """Initialise the quantities required to project velocities"""

        self.rgrid4disp = N.array([])
        self.VProjTCKR  = N.array([])
        self.VProjTCKT  = N.array([])

    def VelProjEvaluate(self):

        """Project the velocity dispersions onto the sky"""

        # first ensure populated
        if not N.any(self.EvalVDispR):
            self.PopulateVDisps()

        # spline representation exists?
        if not N.any(self.VDispShapeR):
            self.VDispsAsSplines()

        # set up grid of points at which to eval everything
#        self.Radii4Proj = N.linspace(0.01, self.ModelRt, 50, endpoint=False)

        if not N.any(self.Radii4Proj):
            self.Radii4Proj = N.logspace(N.log10(0.01), N.log10(self.ModelRt), \
                                             50, endpoint=False)

        # evaluate the numerators
#        print "VelProjEvaluate INFO - evaluating the top parts..."
        self.VelDispersionsNumerators(self.Radii4Proj)
#        print "VelProjEvaluate INFO - ... done."

        # now evaluate surface densities
        self.SurfDensEvalFunction(self.Radii4Proj)

        # now find the ratio
        #
        # self.Rho0 is not the physical surface density but the
        # cluster of constants up-front in the volume density
        # integral.
        denom = self.EvalSurfDensAnalytic / self.Rho0
        self.VDispProjectedR = self.VProjNumeratorR / denom
        self.VDispProjectedT = self.VProjNumeratorT / denom

    def VelDispersionsNumerators(self, inR = None):

        """Evaluate velocity dispersions projected on-sky"""

        # well, just the numerator...

        R = self.EnsureR(inR)
        nrows = N.size(R)
        
        self.VProjNumeratorR = N.zeros(nrows)
        self.VProjNumeratorT = N.zeros(nrows)

        for iR in range(nrows):

            # initialize
            self.VelProjSplineInitialize()

            # set up the integrands
            self.VDispSetupIntegrands(R[iR])

            if len(self.VProjTCKR) < 3:
                continue

            # evaluate the integrals
            ThisIntegR = interpolate.splint(R[iR], self.ModelRt, \
                                                self.VProjTCKR)
            ThisIntegT = interpolate.splint(R[iR], self.ModelRt, \
                                                self.VProjTCKT)

            diffs = ThisIntegR - ThisIntegT
#            print "DBG INTEG:", N.min(diffs), N.max(diffs)

            # pass up to the results vector - factor 2 comes here
            self.VProjNumeratorR[iR] = 2.0 * N.copy(ThisIntegR)
            self.VProjNumeratorT[iR] = 2.0 * N.copy(ThisIntegT)

            # when evaluating the denom numerically, would come here.

    def VDispIntegrandFront(self, r3d = None, R2D = None):

        """Set up the front part of the v_r integrand for given
        projected radius. Make this a function that can be called"""

        # include self.Rho0 (this cancelation seems a bit odd)
        ThisRho = self.ArrayVolumeDensity(r3d) / self.Rho0
        Bracket = r3d / N.sqrt(r3d**2 - R2D**2)

#        print "VDispIntegrandFront DBG:", N.min(r3d), R2D

        return ThisRho * Bracket * self.Rho0

    def ShowModelCompData(self, Clobber = True):

        """Show the data compared to model for surfdens and vdisp"""

        self.FreeFigure()
        P.figure(figsize=(7,7))
        P.subplots_adjust(hspace=0.3, wspace=0.3)
        P.subplot(221)
        P.errorbar(self.R2D, self.DataVDispRad, \
                       yerr = self.DataEDispRad, \
                       fmt = 'k.')
        P.plot(self.R2D, self.VDispProjectedR**0.5, 'g')

        P.subplot(222)
        P.errorbar(self.R2D, self.DataVDispTan, \
                       yerr = self.DataEDispTan, \
                       fmt = 'k.')
        P.plot(self.R2D, self.VDispProjectedT**0.5, 'g')

        P.subplot(212)
        self.ShowSurfDensScaled(Clobber = False)

    def ShowSurfDensScaled(self, Clobber = True):

        """Having scaled surface density profile to data, plot it over
        the surfdens dataset"""

        if Clobber:
            self.FreeFigure()
        P.loglog(self.SurfDensRProj, self.SurfDensNum, 'k.')
        P.errorbar(self.SurfDensRProj, self.SurfDensNum, \
                       yerr=self.SurfDensErr, fmt='k.')

        # evaluate a finer grid
        r4plot = N.logspace(N.log10(N.min(self.SurfDensRProj))-0.05, \
                                N.log10(N.max(self.SurfDensRProj))+0.05 )
        y4plot = self.EvalSurfDensFunc(r4plot) \
            * self.Rho0 * self.SurfDensOptScale

#        P.plot(self.SurfDensRProj, self.SurfDensScaled, 'g')
        P.plot(r4plot, y4plot,'g')
        P.xlabel('R (pc)')
        P.ylabel('Surface Density, N/pc2')
        P.title('Arches - surface density')

    # compare surface density to surface density data
    def ScaleSurfDensToData(self):

        """Given a model for the surface density of the number of
        tracer stars, evaluate it and scale it to the surface density
        data"""
        
        # Aug19 2011 We actually want to use the physical central
        # density here.
        SurfDensityPattern = self.EvalSurfDensFunc(self.SurfDensRProj) \
            * self.Rho0Phys

        ScaleFactor = self.OptScale(SurfDensityPattern, \
                                        self.SurfDensNum, \
                                        self.SurfDensErr)

        # evaluate
        self.SurfDensOptScale = N.copy(ScaleFactor)
        self.SurfDensScaled = SurfDensityPattern * ScaleFactor


    def OptScale(self,P,y,e):

        """Scale pattern P to fit data Y that has errors E"""

        wgts = P**2 / e**2
        pats = P*y / e**2

        return N.sum(pats) / N.sum(wgts)
    
    def GetChisqModel(self):

        """Evaluate chi-squared for model paramset"""
            
        # evaluate chisq and get surf dens
#        print "DBG:", N.size(self.SurfDensScaled)
        self.ChisqSurfDens = self.ComputeChisq(self.SurfDensScaled, \
                                                   self.SurfDensNum, \
                                                   self.SurfDensErr)

        # chisq for radial vdisp...
        self.ChisqVDispRad = self.ComputeChisq(N.sqrt(self.VDispProjectedR), \
                                                   self.DataVDispRad, \
                                                   self.DataEDispRad)

        # ... and tangential vdisp
        self.ChisqVDispTan = self.ComputeChisq(N.sqrt(self.VDispProjectedT), \
                                                   self.DataVDispTan, \
                                                   self.DataEDispTan)
        
        # roll into dictionary for easy access later
        self.DChisq = {}
        self.DChisq['ChiR'] = self.ChisqVDispRad
        self.DChisq['ChiT'] = self.ChisqVDispTan
        self.DChisq['ChiS'] = self.ChisqSurfDens

    def ComputeChisq(self, yeval, ydata, edata):

        """Compute chi-squared"""

        resids = ydata - yeval
        return N.sum((resids / edata)**2)

    # read in surface density data
    def LoadSurfaceDensity(self, SurfFile = 'Espinoza_Reread.txt', \
                               DoPlot = False):

        """Load espinoza et al. surface density data"""

        # old surffile was Espinoza_10-30Msol.txt

#        SurfFile = 'Espinoza_Reread.txt'

#        print "DBG: %s" % (SurfFile)

        if not os.access(SurfFile, os.R_OK):
            print "LOADTXT FATAL - cannot read surffile %s" % (SurfFile)
            return

        # format is R, log10(counts), lower lim
        RProj, LogNum, LogMinus, LogPlus = N.loadtxt(SurfFile, unpack = True)
        
        # lognum was offset by +0.5 dex for clarity on the total-range dataset
        if SurfFile.find('Reread') > -1:
            LogNum = LogNum - 0.5

        # convert to raw numbers
        Num = 10.0**LogNum
        NumLow  = 10.0**( LogNum - LogMinus )
        NumHigh = 10.0**( LogNum + LogMinus )

        # errorbar in linear-N
        DiffLo = N.abs(Num - NumLow  )
        DiffHi = N.abs(Num - NumHigh )
        NumErr = 0.5 * (DiffLo + DiffHi)

        self.SurfDensRProj = RProj
        self.SurfDensNum = Num
        self.SurfDensErr = NumErr


    # read in motions
    def DataReadCaptured(self):
        
        """Standalone routine to read in captured motion-files"""
        
        CapturedRad = 'MotionsRadial.txt'
        CapturedTan = 'MotionsTangential.txt'

        if not os.access(CapturedRad,os.R_OK) or \
                not os.access(CapturedTan, os.R_OK):
            return

        rr, vr, mr, pr = N.loadtxt(CapturedRad, skiprows = 1, unpack=True)
        rt, vt, mt, pt = N.loadtxt(CapturedTan, skiprows = 1, unpack=True)

        er = N.average(N.vstack(( N.abs(mr), N.abs(pr) )), 0)
        et = N.average(N.vstack(( N.abs(mt), N.abs(pt) )), 0)
        r  = N.average(N.vstack(( rr, rt )), 0)

        # pass these up to the class
        self.R2D = N.copy(r)
        self.r3d = N.copy(r)
        self.DataVDispRad = N.copy(vr)
        self.DataVDispTan = N.copy(vt)
        self.DataEDispRad = N.copy(er)
        self.DataEDispTan = N.copy(et)

    def SetupTrialGrids(self):

        """Grid of parameters for runs"""

        self.TrialVects= {}
        self.TrialVects['Rc'] = N.linspace(self.MinRc, self.MaxRc, \
                                              self.NumRc, endpoint=True)

        self.TrialVects['Rt'] = N.linspace(self.MinRt, self.MaxRt, \
                                              self.NumRt, endpoint=True)

        self.TrialVects['Mass'] = N.linspace(self.MinMass, self.MaxMass, \
                                                self.NumMass, endpoint=True)

        # sizes for param-grids
        self.TrialGridShape = (self.NumRc, self.NumRt, self.NumMass)
        ZeroGrid = N.zeros(self.TrialGridShape)

        self.TrialGrids = {}
        for ThisKey in self.TrialGridKeys:
            self.TrialGrids[ThisKey] = N.copy(ZeroGrid)

        self.PopulateTrialGrids()

    def PopulateTrialGrids(self):

        """Populate parameter grids from parameter vectors"""

        for iRc in range(self.TrialGridShape[0]):
            self.TrialGrids['Rc'][iRc, : :] \
                = N.copy(self.TrialVects['Rc'][iRc])

        for iRt in range(self.TrialGridShape[1]):
            self.TrialGrids['Rt'][:, iRt, :] \
                = N.copy(self.TrialVects['Rt'][iRt])

        for iMass in range(self.TrialGridShape[2]):
            self.TrialGrids['Mass'][:, :, iMass] \
                = N.copy(self.TrialVects['Mass'][iMass])

#        self.TrialGrids['ChiS'] = 1.0e3
#        self.TrialGrids['ChiR'] = 1.0e3
#        self.TrialGrids['ChiT'] = 1.0e3

    def EvalMotions(self):

        """Evaluate motions at given radii"""

        # ensure radii are set for kinematic comparison
        self.Radii4Proj = N.copy(self.R2D)
        self.GetRhoM3dSigma2()

    def LoopEvalMotions(self, Debug = False):

        """Go through param-grids of interest"""

        # set up grids
        self.SetupTrialGrids()
        
        iRcMax = self.NumRc
        iRtMax = self.NumRt
        iMaMax = self.NumMass
        if Debug:
            iRcMax = 6
            iRtMax = 6
            iMaMax = 6
            
        # do the loops
        timestarted = time.time()
        NumToDo = iRcMax * iRtMax * iMaMax
        idone = 0

        # write out every five thousand trials
        NumShow = 5000

        for iRc in range(iRcMax):
            self.iRc = iRc

            for iRt in range(iRtMax):
                self.iRt = iRt

                if self.TrialVects['Rc'][iRc] >= self.TrialVects['Rt'][iRt]:
                    continue

                for iMass in range(iMaMax):
                    self.iMass = iMass

                    try:
                        # initialise model
                        self.InitModel()                    
                        self.PassModelParams()

                        # evaluate these parameters, getchisq
                        self.EvalMotions()
                        self.ScaleSurfDensToData()
                        self.GetChisqModel()

                        # pass results to grid
                        self.PassModelResults()

                    except(KeyboardInterrupt):
                        print "LoopEvalMotions INFO - keyboard interrupt detected after %i of %i iterations. Returning." % (idone, NumToDo)
                        return
                    except:
                        dum = 1 # fit was bad for some reason
                        continue

                    idone = idone + 1
                    if idone % NumShow == 1 and idone > NumShow:
                        print "LoopEvalMotions INFO - trial %i of %i, time elapsed %.2f s" % (idone, NumToDo, time.time()-timestarted)
        #                        stdout.flush()
                        self.WriteGridToFits()
        print "\n"
        timel = time.time() - timestarted
        print "time elapsed: %.2f, %i, %.2f, %.2f" % (timel, NumToDo, N.float(NumToDo) / timel, N.float(NumToDo) / timel * 3600.0)

    def PassModelResults(self):

        """Pass model results to trial grids"""
        ThisIndex = (self.iRc, self.iRt, self.iMass)

        # chisq
        for ThisKey in self.DChisq.keys():
            self.TrialGrids[ThisKey][ThisIndex] = N.copy(self.DChisq[ThisKey])

        # parameters of interest
        for Key in self.DInterest.keys():

            try:
                self.TrialGrids[Key][ThisIndex] = N.copy(self.DInterest[Key])
            except: 
                print "PassModelResults WARN - problem at %s" % (Key)

        


        # scale factor for sigma_0
        self.TrialGrids['SigScale'][ThisIndex] = N.copy(self.SurfDensOptScale)

    def PassModelParams(self):

        """Set up model parameters"""

        ThisIndex = (self.iRc, self.iRt, self.iMass)
        self.ModelRc = self.TrialGrids['Rc'][ThisIndex]
        self.ModelRt = self.TrialGrids['Rt'][ThisIndex]
        self.ModelMass = self.TrialGrids['Mass'][ThisIndex]

    def ComputeMoreChisq(self):

        """Given chisq-surf and chisq-r, t, compute kinem and total"""

        self.TrialGrids['ChiKinem'] = self.TrialGrids['ChiR'] \
            + self.TrialGrids['ChiT']

        self.TrialGrids['ChiTot'] = self.TrialGrids['ChiKinem'] \
            + self.TrialGrids['ChiS']

    def WriteGridToFits(self, OutFits = None):

        """Write trial-grid to fits file"""

        if not OutFits:
            OutFits = self.TrialFits2D

        LArrays = []
        LUnits  = []
        for Key in self.TrialGrids.keys():
            LArrays.append(N.ravel(self.TrialGrids[Key]))
            LUnits.append((Key, N.float))

#            if Key.find("Rho0") > -1:
#                print "WriteGridToFits DBG: %s %e %e" % (Key, N.min(self.TrialGrids[Key]), N.max(self.TrialGrids[Key]))

        # now write to fits
        pyfits.writeto(OutFits, \
                           N.rec.fromarrays(LArrays, LUnits), \
                           clobber=True)

    def ReadTrials2D(self, InFits = None):

        """Populate trial-files from fitsfile"""

        FitsFile = self.TrialFits2D
        if InFits:
            FitsFile = InFits

        if not os.access(FitsFile, os.R_OK):
            return

        self.TrialRecArray = pyfits.getdata(FitsFile, 1)

    # now plot a sample of profiles from the chisq bubble
    def CurveSubsetInBubble(self, ChiLev = 7.82, InFits = None, \
                                NRetrieve = 10, RadMin = 0.01, \
                                RadMax = 20.0, NRad = 20, \
                                OutFile = 'ProfilesEval.fits', \
                                UseKinem = False):

        """Read in trials, pick random set of parameters from within
        chisq region, compute and plot resulting curves"""

        if N.size(self.TrialRecArray) < 2:
            self.ReadTrials2D(InFits)
        
        # convenient variables to use
        Chisq    = self.TrialRecArray['ChiTot']
        Rc       = self.TrialRecArray['Rc']
        Rt       = self.TrialRecArray['Rt']
        Mass     = self.TrialRecArray['Mass']
        SigScale = self.TrialRecArray['SigScale']

        if UseKinem:
            Chisq = self.TrialRecArray['ChiKinem']
            OutFile = 'ProfilesEvalKinem.fits'

        ChiMin = N.min(Chisq)
        Goods = N.where(Chisq - ChiMin <= ChiLev)[0]
        if N.size(Goods) < NRetrieve:
            return

        if NRetrieve < 1:
            NRetrieve = N.size(Goods)

        IndsToEval = N.random.random_integers(low=0, high=N.size(Goods)-1, \
                                                  size=NRetrieve)
        IndsToEval = Goods[IndsToEval]

        # set up radius vector to evaluate
        lmin = N.log10(RadMin)
        lmax = N.log10(N.max(Rt[Goods]))
        lnum = NRad
        RProjMaster = N.logspace(lmin, lmax, lnum, endpoint=True)

        self.Radii4Proj = N.copy(RProjMaster)

        # set up results array
        ZeroArray = N.zeros(( NRetrieve, N.size(self.Radii4Proj) ))
        DExampleProfiles = {'Radii':self.Radii4Proj, \
                               'MProj':N.copy(ZeroArray), \
                               'M3d':N.copy(ZeroArray), \
                               'SurfDens':N.copy(ZeroArray), \
                               'SigR':N.copy(ZeroArray), \
                               'SigT':N.copy(ZeroArray), \
                               'ChiTot':Chisq[IndsToEval]}

        # now populate it
        idone = 0
        NWrite = NRetrieve / 10
        timestart = time.time()
        for iTrial in range(NRetrieve):

            jPar = IndsToEval[iTrial]

            ## initialize the model
            self.InitModel()
            self.ModelRc   = Rc[jPar]
            self.ModelRt   = Rt[jPar]
            self.ModelMass = Mass[jPar]
            self.SurfDensOptScale = SigScale[jPar]

            # can only evaluate this at points >= Rc and < Rt
            RGoods = N.where(RProjMaster < self.ModelRt)[0]

            if N.size(RGoods) < 2:
                continue

            self.Radii4Proj = RProjMaster[RGoods]

            try:

                # evaluate the model
                self.GetRhoM3dSigma2()

                # evaluate interior mass at desired radii
                EvalMass2D = self.EnclosedMassAnalyticIso(self.Radii4Proj)

                # evaluate surface density plot
                self.SurfDensOptScale = SigScale[jPar]
                # Aug 19 2011 - self.RhoPhysical was self.Rho0
                SurfDensPlot = self.EvalSurfDensFunc(self.Radii4Proj) \
                    * self.SurfDensOptScale * self.Rho0Phys

                print "DBG:", self.Rho0, self.SurfDensOptScale, SurfDensPlot[0]

                # last thing - evaluate enclosed mass in 3D
                self.ArrayR2RhoInt(self.Radii4Proj)
                EvalMass3D = self.R2RhoInt * self.Rho0

                # pass to master array
                DExampleProfiles['SigR'][iTrial][RGoods] \
                    = N.sqrt(self.VDispProjectedR)

                DExampleProfiles['SigT'][iTrial][RGoods] \
                    = N.sqrt(self.VDispProjectedT)

                DExampleProfiles['M3d'][iTrial][RGoods]   \
                    = N.copy(EvalMass3D)
                
                DExampleProfiles['MProj'][iTrial][RGoods] \
                    = N.copy(EvalMass2D)

                DExampleProfiles['SurfDens'][iTrial][RGoods] \
                    = N.copy(SurfDensPlot)

#                print N.size(RGoods), N.size(EvalMass3D), EvalMass3D[-1]

            except(KeyboardInterrupt):
                print "CurveSubsetInBubble WARN - keyboard interrupt detected."
                self.WriteEvalTrials(DExampleProfiles, OutFile)
                return

            except:
                print "CurveSubsetInBubble WARN - bad fit"
            

            idone = idone + 1
            if idone % NWrite == 1 and idone > NWrite:
                print "CurveSubsetInbubble INFO - at %i of %i - time elapsed %.2f" % (idone, NRetrieve, time.time()-timestart)
                self.WriteEvalTrials(DExampleProfiles, OutFile)

        # final writing out of completed table
        print "Total time elapsed: %.2f" % (time.time()-timestart) 
        self.WriteEvalTrials(DExampleProfiles, OutFile)

    def WriteEvalTrials(self, DProfiles = {}, \
                            OutFits = 'TrialsEval.fits'):

        """Given a set of evaluated trials from King model, output
        them to fits file. One extension per quantity."""

        if len(DProfiles.keys()) < 1:
            return

        # initialise the hdulist
        PriHDU = pyfits.PrimaryHDU()
        hdulist = pyfits.HDUList([PriHDU])
        for ThisKey in DProfiles.keys():
            
            data_hdu = pyfits.ImageHDU(N.copy(DProfiles[ThisKey]))
            data_hdu.name = ThisKey
            hdulist.append(data_hdu)


        hdulist.writeto(OutFits, clobber=True)
        hdulist.close()

    def LoadEvalTrials(self, InFits = 'ProfilesEval.fits'):

        """Load fitsfile with trials to record array"""

        try:
            hdulist = pyfits.open(InFits)
        except:
            return

        # set up dictionary of plots
        self.DEvalProfiles = {}
        for ThisKey in self.DEvalKeys:
            try:
                self.DEvalProfiles[ThisKey] = N.copy(hdulist[ThisKey].data)
            except:
                missed = 1

        hdulist.close()

    def PlotEvalTrials(self, ChiLev = 999.9, NumToPlot = -1, UseKinem=False):

        """Plot eval trials. Also can select by chilev"""
        
        try:
            # key is ChiTot no matter what input
            chisq = self.DEvalProfiles['ChiTot']
        except:
            return


        Goods = N.where(chisq - N.min(chisq) < ChiLev)[0]

        print "here 1"
        if N.size(Goods) < 1:
            return
        
        print "here 2"

        # ensure upper limit is sensible
        if NumToPlot < 0 or NumToPlot > N.size(chisq):
            NumToPlot = N.size(chisq)

        self.ShowEvalAlpha = 10.0/NumToPlot # was 5 / NumToplot AUG 20
        self.ShowEvalAlpha=1.0

        # indices to show - were picked randomly anyway so can get
        # sequentially
        self.ShowEvalIndices = N.arange(NumToPlot)

        # begin figure
        rc('text', usetex = True)

        self.ShowXLabel = r"$R$ $({\rm pc})$"

        # set ylabels here
        self.ShowYLabels = {'MProj':r"$M(<R) / 10^4 M_{\odot}$", \
                                'M3d':r"$M(<r) / 10^4 M_{\odot}$", \
                                'SigR':r"$\sigma_R$ (${\rm km}$ $s^{-1}$)", \
                                'SigT':r"$\sigma_T$ (${\rm km}$ $s^{-1}$)", \
                                'SurfDens':r"$\Sigma_N(R)$, ${\rm Stars}$ ${\rm pc}^{-2}$"}

        self.ShowTitles = {'MProj':'Enclosed Mass - 2D', \
                               'M3d':'Enclosed Mass - 3D', \
                               'SigR':'Radial Dispersion', \
                               'SigT':'Tangential Dispersion', \
                               'SurfDens':'Surface Density'}

        self.FreeFigure()
        P.figure(figsize=(12,8))
        P.subplots_adjust(hspace=0.4,wspace=0.4)

        P.subplot(234)
        self.OverplotSetOfProfiles('MProj', DoLogX = True)

        P.subplot(235)
        self.OverplotSetOfProfiles('MProj', DoLogX = False, \
                                       XRange=[0.1,0.6], YRange =[0.0,1.5])

        P.annotate('Zoom', (0.05,0.90), xycoords='axes fraction', \
                       horizontalalignment='left')

        P.plot([0.4, 0.4], [0.0, 1.5], 'g--')

        P.subplot(236)
        self.OverplotSetOfProfiles('M3d', DoLogX = True)
        P.xlabel(r"$r$ (${\rm pc}$)")

        P.subplot(231)
        self.OverplotSetOfProfiles('SigR', DoLogX = True, DoExtend = False, \
                                       YRange=[2.0,8.0])
    
        # do we have data?
        if N.any(self.DataVDispRad):
            P.errorbar(self.r3d, self.DataVDispRad, yerr=self.DataEDispRad, \
                           fmt='wo', ecolor='b',zorder=2)

        P.subplot(232)

        # is the layering reversed when using eps?
        self.OverplotSetOfProfiles('SigT', DoLogX = True, DoExtend = False, \
                                       YRange=[2.0,8.0])
        if N.any(self.DataVDispTan):
            P.errorbar(self.r3d, self.DataVDispTan, yerr=self.DataEDispTan, \
                           fmt='wo', ecolor='b', zorder=2)

#        self.OverplotSetOfProfiles('SigT', DoLogX = True, DoExtend = False, \
#                                       YRange=[2.0,8.0])
            
        if not UseKinem:
            P.subplot(233)
            self.OverplotSetOfProfiles('SurfDens', \
                                           DoLogX = True, DoLogY = True, \
                                           XRange=[0.01,2.0], \
                                           YRange=[50.0, 1e4], \
                                           UseAlpha = 0.006)


#        P.annotate('Espinoza et al. (2009)', (0.1,0.9), \
#                       xycoords='axes fraction', \
#                       horizontalalignment='left')

        # do we have data?
            if N.any(self.SurfDensRProj):
                P.errorbar(self.SurfDensRProj, self.SurfDensNum, \
                               yerr = self.SurfDensErr, \
                               fmt='wo', ecolor='r', \
                               markersize=3, zorder=2)

        # overall figure title        
#        SuperTitle = r"Isotropic King profiles from $\Delta \chi^2$ confidence region"
        SuperTitle = r"Isotropic King profiles from 95$\%$ confidence region"

        if UseKinem:
            SuperTitle = SuperTitle+' - Kinematics only'

        P.annotate(SuperTitle, (0.5, 0.95), xycoords='figure fraction', \
                       horizontalalignment='center', \
                       verticalalignment='middle', fontsize=16)

        # save the figure
        FigName = 'RadProfiles_King_test.eps'
        if UseKinem:
            FigName = 'RadProfiles_King_Kinem.eps'
            
        if FigName.find('eps') > -1:
            P.savefig(FigName, format="eps_cmyk")
        else:
            P.savefig(FigName)


        rc('text', usetex = False)
   
    def OverplotSetOfProfiles(self, KeyName = None, \
                                  DoLogX = False, DoLogY = False, \
                                  XRange = None, YRange=None, \
                                  DoExtend = True, UseAlpha = None, \
                                  LayerOrder = 1):

        """Overplot a given set of profiles"""

        if not KeyName in self.DEvalProfiles.keys():
            return

        ScaleFactor = 1.0
        if KeyName.find('M') > -1:
            ScaleFactor = 1.0e-4

        R  = self.DEvalProfiles['Radii']
        YA = self.DEvalProfiles[KeyName] * ScaleFactor

        # have we been given minmax indices to highlight??
        JMin = 1e9
        JMax = 1e9
        try:
            JMin = 1 * self.ShowEvalJMin
        except:
            dum = 1

        try:
            JMax = 1 * self.ShowEvalJMax
        except:
            dum = 1

        for iRow in self.ShowEvalIndices:
            
            YThis = YA[iRow]
            Goods = N.where((N.isfinite(YThis) > 0) & \
                                (YThis > 1.0e-3))[0]

            if N.size(Goods) < 2:
                continue

            # plot fanciness
            PlotColor = 'k'
            # try forcing the transparency since .eps doesn't support it
            PlotColor = '0.75'

            # better way:
            PlotColor = '%.2f' % (N.random.uniform(0.3,0.99))

            PlotLinestyle = '-'
            try:
                PlotAlpha = UseAlpha * 1.0
            except:
                PlotAlpha = self.ShowEvalAlpha

            if N.abs(iRow - JMin) < 0.4:
                PlotColor = 'b'
                PlotLinestyle = '-.'
                PlotAlpha = 0.8

            if N.abs(iRow - JMax) < 0.4:
                PlotColor = 'r'
                PlotLinestyle='--'
                PlotAlpha = 0.8

            # yforplot - extend outer limits where Router = Rt
            RPlot = R[Goods]
            YPlot = YThis[Goods]

            Saturated = N.where( YPlot < 1.0e-4)[0]
            if N.size(Saturated) > 0:
                SatHi = N.where(RPlot[Saturated] > RPlot[Goods[-2]])[0]
                if N.size(SatHi) > 0:
                    SatHi = Saturated[SatHi]
                    MaxVal = YPlot[Goods[-2]]
                    YPlot[SatHi] = MaxVal

                    RPlot = N.hstack(( RPlot, N.max(R) ))
                    YPlot = N.hstack(( YPlot, MaxVal   ))

            if RPlot[-1] < N.max(R) and DoExtend:
                RPlot = N.hstack(( RPlot, N.max(R) ))
                YPlot = N.hstack(( YPlot, YPlot[-1] ))

            if self.ShowEvalAsSpline:
                tck = interpolate.splrep(RPlot, YPlot, s=0)

                rfine = N.logspace(N.log10(N.min(R[Goods])), \
                                       N.log10(N.max(R[Goods])), \
                                       N.size(Goods)*10.0, \
                                       endpoint=True)

                yfine = interpolate.splev(rfine, tck, der=0)
                
                if rfine[-1] > N.max(R) and DoExtend:
                    rfine = N.hstack(( rfine, N.max(R)  ))
                    yfine = N.hstack(( yfine, yfine[-1] ))

                RPlot = N.copy(rfine)
                YPlot = N.copy(yfine)

            if DoLogX:
#                print "QQQ"
                P.semilogx(RPlot, YPlot, \
#                           alpha = PlotAlpha, \
                               color=PlotColor, ls = PlotLinestyle, \
                               lw = 0.5,zorder=LayerOrder)
    

                if DoLogY:
                    P.loglog(RPlot, YPlot, alpha = PlotAlpha, 
                             color=PlotColor, ls = PlotLinestyle, \
                                 zorder = LayerOrder)
            else:
                if DoLogY:
                    P.semilogy(RPlot, YPlot, alpha = PlotAlpha, 
                               color=PlotColor, ls = PlotLinestyle, \
                                   zorder = LayerOrder)

                P.plot(RPlot, YPlot, alpha = PlotAlpha, 
                       color=PlotColor, ls = PlotLinestyle, \
                           zorder = LayerOrder)


        # set axis
        plotax = N.copy(P.axis())
        if XRange:
            if N.size(XRange) > 1:
                plotax[0] = XRange[0]
                plotax[1] = XRange[1]
        if YRange:
            if N.size(YRange) > 1:
                plotax[2] = YRange[0]
                plotax[3] = YRange[1]

        P.axis(plotax)

        # set label
        if KeyName in self.ShowYLabels.keys():
            P.ylabel(self.ShowYLabels[KeyName], fontsize=14)

        try:
            P.xlabel(self.ShowXLabel, fontsize=14)
        except:
            dum = 1

        if KeyName in self.ShowTitles:
            P.title(self.ShowTitles[KeyName], fontsize=14)

    def AssessChisqBubbles(self, InFits = None):

        """Assess chi-square bubbles for results of trials"""

        # ensure trial vector is populated
        if N.size(self.TrialRecArray) < 2:
            self.ReadTrials2D(InFits)

        self.DChisqRanges = {}
        ChiKeys = ['ChiTot', 'ChiKinem']
        for Chi in ChiKeys:
            self.DChisqRanges[Chi] = self.FindChisqRanges(Chi)

    def FindChisqRanges(self, ChiKey = 'ChiTot'):

        """Called by EvalChisqRanges - find param ranges corresponding
        to a particular chisq value from the minimum"""

        # also finds the value at the minimum chisquared

        # initialise
        DRet = {}

        try: 
            Chi = self.TrialRecArray[ChiKey]
        except:
            return DRet

         # find the global minimum, populate delta-chisq
        iMinChi = N.argmin(Chi)
        ChiMin = Chi[iMinChi]
        DeltaChi = Chi - ChiMin

        # pass values at minimum to return-dictionary
#        DRet['MinVals'] = N.copy(self.TrialRecArray[iMinChi])

#        print "DBG - min vals:" , DRet['MinVals']
#        print "DBG - TrialRecArray", N.shape(self.TrialRecArray), N.shape(Chi)

        # initialise the minvals array, this syntax changed since
        # previous version (and the variable DRet['MinVals'] not
        # currently used anywhere else
        DRet['MinVals'] = {}
        
        # now find parameter-ranges
        for iChi in range(len(self.DeltaChiAssess)):
            DeltaChiThis = self.DeltaChiAssess[iChi]

            # find parameters in the bubble
            g = N.where(DeltaChi <= DeltaChiThis)[0]
            if N.size(g) < 1:
                continue

            # populate return dictionary. Each par gets its own minmax
            # list.
            DRet[DeltaChiThis] = {}
            for Par in self.DeltaChiKeys:

                # only continue if par is in the record-array!
                if not Par in self.TrialRecArray._names:
                    continue

                # compute minmax in the region and pass up to
                # dictionary
                Vec = self.TrialRecArray[Par]
                DRet[DeltaChiThis][Par] = [N.min(Vec[g]), N.max(Vec[g])]

                # find the minimum value in the grid
                if iChi < 1:
                    DRet['MinVals'][Par] = self.TrialRecArray[Par][iMinChi]

#                print Par, DRet['MinVals'][Par], DRet[DeltaChiThis][Par]

        self.MassValues4Report(DRet, "ReportValues_%s.txt" % (ChiKey))
                
        # return the dictionary
        return DRet

    def MassValues4Report(self, DThis={}, ThisFil='Values4Report.txt'):

        """Find mass values to report for paper"""

        if len(DThis.keys()) < 1:
            return

        # assemble list of numerical keys
        NKeys = []
        for ChiVal in DThis.keys():

            try:
                bob = N.float(ChiVal)
                NKeys.append(ChiVal)
            except:
                dum = 1
            
        # open output file
        fObj = open(ThisFil, 'w')
                
        # now convert the min +/- into ranges
        ChiKey = NKeys[self.ReportSig]
        fObj.write("#ChiMin %.2f \n" % (ChiKey))
        for ParReport in self.ReportKeys:
            ThisMinVal = DThis['MinVals'][ParReport]
            ThisRange = DThis[ChiKey][ParReport]
            ThisLimLo = ThisRange[0]
            ThisLimHi = ThisRange[1]

            ThisRangeLo = ThisMinVal - ThisLimLo
            ThisRangeHi = ThisLimHi - ThisMinVal

            #print ParReport, ThisLimLo, ThisMinVal, ThisLimHi
            #print ParReport, ThisMinVal, ThisRangeLo, ThisRangeHi

            # write to output file
            fObj.write("%s %.2f %.2f %.2f %.2f %.2f \n" % (ParReport, ThisMinVal, ThisRangeLo, ThisRangeHi, ThisLimLo, ThisLimHi))

        fObj.close()

    def AssembleChiRegionsTables(self):

        """Assemble tables of chi-square ranges"""

        ChiKeys = ['ChiTot', 'ChiKinem']
        self.DChisqTables = {}
        for Chi in ChiKeys:
            self.ChiRegionsTable(Chi)

    def WriteChiRegionsTables(self,OutFits='ChisqTable.fits'):

        """Having assembled chisq region tables, write them to disk"""

        Keys = self.DChisqTables.keys()
        try:
            TableSet = atpy.TableSet()
        except:
            print "WriteChiRegionsTables FATAL - atpy not imported"
            return

        for Key in Keys:
            TableSet.append(self.DChisqTables[Key])

        TableSet.write(OutFits, type='fits', overwrite=True)

    def WriteChi2LaTeX(self, FullDoc = True):

        """Write chisq regions to latex table"""

        self.ChiRegionsLaTeX("ChiTot", 'TableChiTot.tex', \
                                 FullChi = True, FullDoc = FullDoc)
        self.ChiRegionsLaTeX("ChiKinem", 'TableChiKin.tex', \
                                 FullChi = False, FullDoc = FullDoc)

    def ChiRegionsLaTeX(self, ChiKey = 'ChiTot', OutFil='test.tex', \
                            FullChi = True, FullDoc = False):

        """Generate LaTeX table for paper / report"""

        try:
            DThis = self.DChisqRanges[ChiKey]
        except:
            return
        
        
        # one column per confidence interval
        # one row per quantity
        
        TableLines = []

        if FullDoc:
            TableLines = ["\\documentclass{article} \n",\
                              "\\begin{document} \n"]

        # frontmatter
        TableLines.append('\\begin{table}\n')
        TableLines.append('\\begin{tabular}{r|c|c|c} \n')

        # what are we calling chisq here?
        CallChi = "\chi^2_{full}"
        ChiLabel = "both the Arches kinematic dataset and the surface density dataset of Espinoza et al. (2009)."
        if not FullChi:
            CallChi = "\chi^2_{kinem}"
            ChiLabel = "the Arches kinematic dataset only."


        # caption string
        CapString = "Significance regions for isotropic King modeling of the Arches cluster. Ranges of each parameter corresponding to the stated significance level are given, when $R_c, R_t, M_{cluster}$~are all allowed to vary. The quantity $%s$~denotes the badness-of-fit when comparing model predictions to %s" % (CallChi, ChiLabel)

        # endmatter
        TailLines = ['\\end{tabular} \n',\
                         '\\caption{%s} \n' % (CapString),\
                         '\\end{table} \n']

        HeaderLine = '$\\Delta %s$' % (CallChi)
        ChiVals = DThis.keys()[0:-1]
        for ChiVal in ChiVals:
#            HeaderLine = "%s & \\multicolumn{2}{|c|}{%.2f} " % (HeaderLine, ChiVal)
            HeaderLine = "%s & %.2f " % (HeaderLine, ChiVal)
            
        HeaderLine = "%s \\\\ \n" % (HeaderLine) 

        TableLines.append(HeaderLine)

        ProbLine = ""
        if len(self.DeltaProbLevels) == len(ChiVals):
            ProbLine = "Confidence"
            for Conf in self.DeltaProbLevels:
#                ProbLine = "%s & \\multicolumn{2}{|c|}{%s}" % (ProbLine, Conf)
                ProbLine = "%s & %s" % (ProbLine, Conf)
            ProbLine = "%s \\\\ \n" % (ProbLine)
            TableLines.append(ProbLine)
            
        # significance levels
        if len(self.DeltaSigLevels) == len(ChiVals):
            SigLine = ""
            for Sig in self.DeltaSigLevels:
                SigLine = "%s & %s" % (SigLine, Sig)
            SigLine = "%s \\\\ \n" % (SigLine)
            TableLines.append(SigLine)

        TableLines.append("\\hline \n")

        # now assemble the data lines
        # print "Minimum Values keys", DThis['MinVals'].keys()

        for ThisQuant in self.KeysForOutTable:

            # don't include f_massive if using kinematic information only
            if not FullChi and ThisQuant.find("SigScale") > -1:
                continue

            # produce the line
            LThis = self.ChiMakeLaTeXLine(DThis, ThisQuant)

            print DThis['MinVals'][ThisQuant]

            if len(LThis) < 1:
                continue

            for dline in LThis:
                TableLines.append(dline)

        # append tail lines onto the table list
        for eline in TailLines:
            TableLines.append(eline)

        if FullDoc:
            TableLines.append("\\end{document} \n")

        fObj = open(OutFil, 'w')
        for line in TableLines:
            fObj.write(line)
        fObj.close()

    def ChiMakeLaTeXLine(self, DataDict = {}, Quant=None):

        """From chisq ranges dictionary, produce table line that
        corresponds to the quantity of interest. Return as string."""

        try:
            Keys = DataDict.keys()
            ChiLevs = Keys[0:-1]
            QuantVec = DataDict[ChiLevs[0]][Quant]
        except:
            return []

        # set the row title, unit and scale factor
        RowTitle, RowScale, RowUnit = self.LaTeXRowPars(Quant)

        # two rows: data and units
        RowData = RowTitle
        RowUnit = RowUnit

        # loop thru the delta-chi values
        for ThisChi in ChiLevs:

            # minval, maxval, scale
            ThisVec = DataDict[ThisChi][Quant]
            MinRep = "%.2f" % (ThisVec[0] * RowScale)
            MaxRep = "%.2f" % (ThisVec[1] * RowScale)

            # wrap into row
            RowEntry = "%s - %s" % (MinRep, MaxRep)

            # append onto row
            RowData = "%s & %s" % (RowData,RowEntry)
            RowUnit = "%s &  " % (RowUnit)

        # add the carriage return to the end of the lines
        RowData = "%s \\\\ \n" % (RowData)
        RowUnit = "%s \\\\ \n" % (RowUnit)
        
        # produce list of return lines
        RowLines = [RowData,RowUnit,'\\hline \n']

        return RowLines

    def LaTeXRowPars(self, Quant = None):

        """Set latex-ready formatting for output table"""

        print "DBG:", Quant

        if not Quant:
            return '', 1.0, ''

        RowScale = 1.0
        RowTitle = Quant
        RowUnit  = ''

        if RowTitle.find("MProj") > -1:
            RowTitle = "$M(R< %.2f~{\\rm pc})$" % (self.RFidMProj)
            RowScale = 1.0e-4
            RowUnit  = "$10^4~M_{\\odot}$"

        if RowTitle.find("M3d") > -1:
            RowTitle = "$M(r < %.1f~{\\rm pc})$" % (self.RFidM3d)
            RowScale = 1.0e-4
            RowUnit  = "$10^4~M_{\\odot}$"

        if RowTitle.find("Rho0Phys") > -1:
            RowTitle = "$\\rho_0$"
            RowScale = 1.0e-5
            RowUnit = "$10^5 M_{\\odot}~{\\rm pc^{-3}}$"

        if RowTitle.find("Rc") > -1:
            RowTitle = "$R_c$"
            RowUnit  = 'pc'

        if RowTitle.find("Rt") > -1:
            RowTitle = "$R_t$"
            RowUnit  = 'pc' 

        if RowTitle.find("Mass") > -1:
            RowTitle = "$M_{cluster}$"
            RowScale = 1.0e-4
            RowUnit  = "$10^4~M_{\\odot}$"

        if RowTitle.find("SigScale") > -1:
            RowTitle = "$1000 \\times \\Sigma_{N,0} / \\rho_0$"
            RowScale = 1000.0
            RowUnit  = 'stars pc$^{-2} / M_{\\odot}~{\\rm pc}^{-3}$'

        # add the parentheses if rowunit > 0
        if len(RowUnit) > 0:
            RowUnit = "(%s)" % (RowUnit)

        return RowTitle, RowScale, RowUnit

    def ChiRegionsTable(self, ChiKey = 'ChiTot'):

        """Given dictionary of parameter-regions corresponding to
        delta-chisq, arrange them into a table"""

        try:
            DThis = self.DChisqRanges[ChiKey]
        except:
            return

        # use atpy for convenience
        #
        # construct blank table from keys
        KeysForTable = ['MProj', 'M3d', 'Rc', 'Rt', 'Mass']

        # dictionary is organized first by chisq values then by
        # quantity keyname. Set up chi-values vector and use this to
        # initialise the table.

        # currently "minval" is the final key entry, which behaves
        # differently from the chilevels.
        ChiVals = DThis.keys()[0:-1]

        try:
            Table = atpy.Table()
        except:
            print "ChiRegionsTable FATAL - atpy not imported"
            return

        Table.table_name = ChiKey

        Table.add_column(ChiKey, ChiVals)

        # add the other couple of columns
        if len(ChiVals) == len(self.DeltaProbLevels):
            Table.add_column("C", self.DeltaProbLevels)

        for Quant in KeysForTable:
            
            # key must be present in dictionary
            if not Quant in DThis[ChiVals[0]].keys():
                continue

            # initialise vectors and headers
            MinVals = N.array([])
            MaxVals = N.array([])
            MinName = "$%s_{min}$" % (Quant)
            MaxName = "$%s_{max}$" % (Quant)

            MinReps = []
            MaxReps = []

            # now go thru chilevs and populate the vectors
            for ChiLev in ChiVals:

                try:
                    dum = 1.0 * ChiLev
                except:
                    continue

                if not Quant in DThis[ChiLev].keys():
                    continue
                
                # minmax vector
                ThisVec = N.asarray(DThis[ChiLev][Quant])

                # we want masses in msol
                if Quant.find('M') > -1:
                    ThisVec = ThisVec / 1.0e4

                # append values
                MinVals = N.hstack((MinVals, ThisVec[0]))
                MaxVals = N.hstack((MaxVals, ThisVec[1]))

                MinReps.append("%.2f" % (ThisVec[0]))
                MaxReps.append("%.2f" % (ThisVec[1]))

            # now the vector is populated, add it to the table.
            Table.add_column(MinName, MinReps)
            Table.add_column(MaxName, MaxReps)

        self.DChisqTables[ChiKey] = Table

    def AssembleMasterTables(self, TopDir=None):

        """Read in a set of latex tables and produce a master table
        giving the spread over all the center-choices."""

        # assemble the dictionary
        DLatex = self.ReadLaTeXTables(TopDir)

        # canned list of row-headers that correspond to quantities we
        # care about
        HeadingsData = ['$M(R< 0.40~{\\rm pc})$', \
                            '$M(r < 1.0~{\\rm pc})$', \
                            '$\\rho_0$', \
                            '$R_c$', '$R_t$', '$M_{cluster}$', \
                            '$100 \\times \\Sigma_0 / \\rho_0$', \
                            ]

        for Table in DLatex.keys():

            # convenience variable
            ThisSet = DLatex[Table]

            # output file name
            NameBits = Table.split('.')
            TexName = NameBits[0]+'_master.'+NameBits[-1]

            # initialise the table
            LTable = []

            # use the first in the list as the template for the output
            # table
            Key0 = ThisSet.keys()[0]
            L0 = ThisSet[Key0]
            
            NumSigs = 3

            iLine = -1
            for line0 in L0:
                iLine = iLine + 1
                vline = line0.split('&')

                # Count the number of significance values
                IsSig = vline[0].strip() in "Significance"
                if IsSig:
                    NumSigs = N.size(vline)-1

                # if not data, pass right through to the master
                # line-list
                IsData = vline[0].strip() in HeadingsData
                if not IsData:
                    LTable.append(line0)
                    continue

                # gather the ranges of this quantity from all the
                # tables
                DRange = {}
                for Key in ThisSet.keys():
                    DRange[Key] = ThisSet[Key][iLine]

                LineThis = self.RangesFromLaTeX(DRange)

                # append the line
                LTable.append(LineThis)


            # write the table to disk
            try:
                fObj = open(TexName,'w')
                for line in LTable:
                    fObj.write("%s \n" % (line))
                fObj.close()

            except:
                nevermind = 1


    def RangesFromLaTeX(self, DPars = {}):

        """Given dictionary of variable ranges, return the overall
        range of values"""

        NTables = len(DPars.keys())
        if NTables < 1:
            return

        # minmax array
        AMins = N.array([])
        AMaxs = N.array([])

        # master string-list with entries
        LMins = []
        LMaxs = []

        LineHeader = ''

        # number of entries
        for Key in DPars.keys():
            VLos = N.array([])
            VHis = N.array([])

            DPars[Key] = DPars[Key].split('\\\\')[0]
            Vec = DPars[Key].split('&')

            LineHeader = Vec[0].strip()

            ListMins = []
            ListMaxs = []
            for iCol in range(1, len(Vec)):
                nums = Vec[iCol].split('-')

                ListMins.append(nums[0])
                ListMaxs.append(nums[1])
                
                VLos = N.hstack(( VLos, N.float(nums[0]) ))
                VHis = N.hstack(( VHis, N.float(nums[1]) ))

            LMins.append(ListMins)
            LMaxs.append(ListMaxs)

            if N.size(AMins) > 2:
                AMins = N.vstack(( AMins, VLos ))
                AMaxs = N.vstack(( AMaxs, VHis ))
            else:
                AMins = N.copy(VLos)
                AMaxs = N.copy(VHis)

        # find minmax values...
        VecMin = N.min(AMins, axis=0)
        VecMax = N.max(AMaxs, axis=0)

        aVecMin = N.argmin(AMins, axis=0)
        aVecMax = N.argmax(AMaxs, axis=0)

        if LineHeader.find('1.0') > -1:
            print "RangesFromLaTeX DBG:"
            print AMins
            print aVecMin

            print AMaxs
            print aVecMax

        #... and construct return string 
        LineThis = LineHeader 
        for iCol in range(N.size(aVecMin)):
            RangeThis = '%s - %s' % (LMins[aVecMin[iCol]][iCol].strip(), \
                                         LMaxs[aVecMax[iCol]][iCol].strip())
            LineThis = LineThis + ' & ' + RangeThis

        LineThis = LineThis + '\\\\'

        if LineHeader.find('1.0') > -1:
            print LineThis

        return LineThis

    def ReadLaTeXTables(self, TopDir=None, SubDirs = None):

        """Given a set of tables with paramranges from subset of runs
        with different cluster centers, gather them into a master
        table giving the ranges including center-variations"""

        if not TopDir:
            TopDir = '/Users/clarkson/Data/scratch'

        if not SubDirs:
            SubDirs = ['', '/m05_m25', '/m05_m15', '/p15_m25']

        # first read in the tables
        TableNames = ['TableChiTot.tex', 'TableChiKin.tex']

        DLatex = {}
        for Table in TableNames:
            DLatex[Table] = {}
            for iDir in range(len(SubDirs)):
                ThisFile = TopDir+SubDirs[iDir]+'/'+Table
                DLatex[Table][iDir] = self.ReadLatexTable(ThisFile)
                
        return DLatex
    
    def ReadLatexTable(self, TableFile = None):

        """ReadLatexTable into list"""

        LTable = []
        if not TableFile:
            return

        try:
            fObj = open(TableFile, 'r')
            for line in fObj:
                LTable.append(line.strip())
            fObj.close()    
        except:
            dum = 1

        return LTable
            

def TestSetupForms():

    """Test routine to set up rho(r), rho0, M(<r), sigma^2(r)"""

    A = King()
    A.GetRhoM3dSigma2()
    A.ShowRhoM3dSigma2()

def TestValidates():

    """Validate the two methods of getting the projected
    dispersions"""

    # get to the right place
    os.chdir('/Users/clarkson/Data/scratch/validate')

    flis = glob.glob('Tuan*.dat')
    for ThisFile in flis:
        TestCompareTuan(ThisFile)

def TestCompareTuan(InFile = 'TuanProf_Rc0.14_Rt1.00_M31000.dat'):

    """Test dispersion projection routines against Tuan's routines"""
    
    A = King()
    
    # set up parameters
    A.ModelRc   = N.float(InFile.split('Rc')[1].split('_')[0])
    A.ModelRt   = N.float(InFile.split('Rt')[1].split('_')[0])
    A.ModelMass = N.float(InFile.split('M')[1].split('.dat')[0])
    A.ValidationFile = InFile

    if not os.access(InFile, os.R_OK):
        print "Comparison file %s not readable" % (InFile)
        return

    # set the validation png file
    stem = InFile.split('.dat')[0]
    A.ValidationImg = stem+'.png'

    # load the infile and use its radii for comp to tuan
    rdum, dum = N.loadtxt(InFile, unpack=True)
    A.Radii4Proj = rdum

    # test the model and plot
    timestarted = time.time()
    A.GetRhoM3dSigma2()
    print "time elapsed: %.2f" % (time.time() - timestarted)
    A.ShowVDisps()

def TestEvalParams():

    """Test routine to evaluate parameters at certain projected radii"""

    A = King()
    A.DataReadCaptured()
    A.LoadSurfaceDensity()
    A.EvalMotions()
    A.ScaleSurfDensToData()
    A.ShowRhoM3dSigma2()
#    A.ShowModelCompData()
    A.GetChisqModel()

def TestLoop(Debug = True, BigRun = False, MedRun = False, \
                 SubRange = False, Coarse=False):

    """Loop through King model parameters and evaluate chisq for
    each"""

    A = King()
    if BigRun:
#        A.NumRt = 75
        A.NumRc = 75
        A.NumMass = 75
        A.MaxRt = 100  # parsecs - go out to large distances for coarse run
        A.NumRt = 75

    if MedRun:
        A.NumRc = 30
        A.NumMass = 30
        A.MaxRt = 30
        A.NumRt = 30

    if Coarse:
        A.MaxMass = 1.0e5
        A.MaxRt   = 50.0

    SurfDensFile = 'Espinoza_Reread.txt'
    if SubRange:
        SurfDensFile = 'Espinoza_10-30Msol.txt'

    A.DataReadCaptured()
    A.LoadSurfaceDensity(SurfDensFile)
    A.LoopEvalMotions(Debug = Debug)
    A.ComputeMoreChisq()
    s = ''
    if Debug:
        s = '_DBG'
    A.WriteGridToFits("King_Trials_2D_Iso%s.fits" % (s))

# assess the trials that result
def TestAssessChi(NRetrieve = -1):

    """Assess resulting trials"""

    A = King()
    A.AssessChisqBubbles('King_Trials_2D_Iso.fits')
    A.AssembleChiRegionsTables()
#    A.WriteChiRegionsTables('ChisqTable.fits')
    A.WriteChi2LaTeX()
    A.CurveSubsetInBubble(ChiLev = 7.82, \
                              InFits='King_Trials_2D_Iso.fits', \
                              NRetrieve = NRetrieve)

def TestCurveSubset(UseKinem = True, NRetrieve = 100, ChiLev = 7.82):

    """Standalone test routine - plot curves for trial
    paramsets. Defaults to kinematic data only"""

    A = King()
    A.CurveSubsetInBubble(ChiLev = ChiLev, InFits='King_Trials_2D_Iso.fits', \
                              NRetrieve = NRetrieve, UseKinem = UseKinem)

def TestRewriteLaTeX():

    """Having made a small tweak to the latex-generating code, rerun
    it"""

    TopDir = os.getcwd()
    subdirs = ['', '/m05_m25', '/m05_m15', '/p15_m25']
    
    for SubDir in subdirs:
        ThisDir = TopDir+'/'+SubDir
        os.chdir(ThisDir)
        TestWriteLaTEX()

    os.chdir(TopDir) 

def TestWriteLaTEX():

    """Write output tables to .tex"""

    A = King()
    A.AssessChisqBubbles('King_Trials_2D_Iso.fits')
    A.WriteChi2LaTeX()

def TestPlotProfiles(ChiLev = 999.9, NumToPlot = 100, SurfFile = None, \
                         UseKinem = False):

    """Given a fitsfile of evaluated profiles, plot them over the
    dataset"""

    A = King()

    TrialsFile = 'ProfilesEval.fits'
    if UseKinem:
        TrialsFile = 'ProfilesEvalKinem.fits'
    A.LoadEvalTrials(TrialsFile)
    A.ShowEvalAsSpline = True
#    A.CurveSubsetInBubble(ChiLev = 7.82, \
#                              InFits='King_Trials_2D_Iso.fits', \
#                              NRetrieve = 101)

    # load surface density data for overplot
    A.LoadSurfaceDensity(SurfFile = SurfFile)
    A.DataReadCaptured()
    A.PlotEvalTrials(ChiLev, NumToPlot, UseKinem=UseKinem)

#    print N.size(A.DEvalProfiles['ChiTot'])

def TestRunOvernight(BigRun=False, Coarse=False, MedRun = True, SubRange=False):

    """Run coarse-grid mass model on range of centers and
    masses. Output rho0 in physical units."""

    TopDir = '/Users/clarkson/Data/scratch/coarse/fullrange'
    if SubRange:
        TopDir = '/Users/clarkson/Data/scratch/coarse/10-30Msol'
    
    SubDirs = ['', 'm05_m25', 'p15_m25', 'm05_m15']
    
    for subdir in SubDirs:
        ThisDir = "%s/%s" % (TopDir, subdir)
        os.chdir(ThisDir)
        TestLeaveGoing(BigRun = BigRun, Coarse=Coarse, MedRun = MedRun, \
                           SubRange = SubRange, )

    os.chdir(TopDir)

    # now try assembling master latex table
    A = King()
    A.AssembleMasterTables()


def TestTryCenters(BigRun = False):

    """Loop through selection of cluster center locations"""

    TopDir  = '/Users/clarkson/Data/scratch'
    SubDirs = ['m05_m25', 'p15_m25', 'm05_m15']

    for subdir in SubDirs:
        ThisDir = "%s/%s" % (TopDir, subdir)
        os.chdir(ThisDir)
        TestLeaveGoing(BigRun = BigRun)

    os.chdir(TopDir)

def TestLeaveGoing(BigRun = False, MedRun = True, SubRange = False, Coarse=False):

    print "Starting Run:"
    starttime = time.time()

    logObj = open('Runlog.log', 'w')

    TestLoop(Debug=False, BigRun = BigRun, MedRun = MedRun, SubRange=SubRange, \
                 Coarse=Coarse)
    TimeLine = "Finished Loops after %.2f seconds" % (time.time() - starttime) 
    logObj.write("%s \n" % (TimeLine))

    TestAssessChi()
    TimeLine = "Finished Assessing after %.2f seconds" % (time.time() - starttime)
    logObj.write("%s \n" % (TimeLine))
    print TimeLine

    TestPlotProfiles(7.82, 500)
    TimeLine = "Finished plotting after %.2f seconds" % (time.time() - starttime)

    print TimeLine
    logObj.write("%s \n" % (TimeLine))    
    logObj.close()
    
def TestReadLatex():

    """Read latex tables into dictionary"""

    A = King()
    DThis = A.ReadLaTeXTables()

    return DThis

def TestAssembleLaTeX(TopDir=None):

    """Assemble latex master table"""

    A = King()
    A.AssembleMasterTables(TopDir=TopDir)
