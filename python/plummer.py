#
# plummer.py - given a set of motions and an image center, find the
# limits of a plummer mass model that fits the tangential and radial
# motions
#

# at this point all the integrators are in place and tested. Now need
# to read in the data, compare it with the model, and actually go
# through the values.

import numpy as N
import pylab as P
import os, sys, time, string
from scipy.integrate import quad, dblquad, Inf # integration
from scipy import interpolate
from scipy.optimize import leastsq
import pyfits
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

rc('text', usetex=False)

class Plummer(object):

    """Class defining plummer model, data, results"""

    def __init__(self):

        # couple of control variables
        self.Verbose = True
        self.DoSurfDens2D = True
        self.UseFiniteUpper = False
        self.DoCalcWhileFitting = False
        self.CutUnphysical = False
        self.PlotMinimizing = False
        self.PlotNum = 8
        self.UseSurfDens = True
        self.CalcFast = False # ignore mproj?

        # coordinate-list
        self.MotionsFile = None

        # image center
        self.ImageCenter = N.zeros(2)

        # data
        self.MotionRad = N.array([])
        self.MotionTan = N.array([])
        self.RadiiRad = N.array([])
        self.RadiiTan = N.array([])

        # model shape parameters
        self.ModelR0 = None
        self.ModelNr = 1.0
        self.ModelNt = 1.0

        # model central values
        self.ModelDensity0    = None
        self.ModelSurfdens0   = None
        self.ModelVDisp0 = 1.0    # default for testing shape

        # mass within projected fiducial radius
        self.ModelMFiducial = None

        # radius range for checking physicality of model
        #
        # (just evaluating a function so can be large)
        self.r3PhysicalMin = 0.01
        self.r3PhysicalMax = 10.0
        self.r3PhysicalNum = 1000
        self.r3PhysicalRad = N.array([])

        # flag values
        self.ModelIsPhysical = True
        self.BadChisq = 1.0e9
        self.BadParam = -99.9

        # when simulating, will be inserting a "truth" set of
        # parametesr
        self.TestModelR0 = 0.3
        self.TestModelNr = 1.0
        self.TestModelNt = 1.0
        self.TestModelDensity0  = 1.0e4
        self.TestModelSurfdens0 = 1.0
        self.TestModelVdisp0    = 1.0
        self.TestObsRadii = None

        # projected mass within fiducial radius
        self.TestModelMFiducial = 10000.0   

        # observational error in model
        self.TestModelErrorRad = 0.05
        self.TestModelErrorTan = 0.05

        # using projected mass as a trial variable?
        self.UseMProj = False

        # fiducial radius for evaluating the projected mass (if using
        # Mproj as a varioable)
        self.FiducialRProj =  0.4

        # fiducial for evaluating the 3D mass 
        self.FiducialR3D   =  1.0  

        # radii
        self.r3d = N.array([])    # "r"
        self.r2D = N.array([])    # "R"

        # enclosed masss
        self.ModelEnclosedMass3DKinem = N.array([]) # M (< r)
        self.ModelEnclosedMass2DKinem = N.array([]) # M (< R)
        self.ModelSurfDens2DKinem = N.array([])

        # evaluate enclosed mass 3D and 2D from plummer model alone
        self.ModelEnclosedMass3D = N.array([])
        self.ModelEnclosedMass2D = N.array([]) # M (< R)

        # volume density evaluation
        self.ModelVolumeDensity = N.array([])

        # evaluation of the projected radial and transverse velocity
        # dispersions given the model
        self.EvalVDispRad = N.array([])
        self.EvalVDispTan = N.array([])

        # evaluation of parameters of interest
        self.EvalDensity0 = None        
        self.EvalMFiduc3d = None   # mass within 3D fiducial radius

        # parts of the density model (useful if when taking Mproj M <
        # 0.4 pc as our stepping-variable)
        self.EvalRhoConstant = None
        self.EvalRhoIntegral = None

        # data dispersions
        self.DataVDispRad = N.array([])
        self.DataVDispTan = N.array([])
        self.DataEDispRad = N.array([])
        self.DataEDispTan = N.array([])

        # chisq vector
        self.EvalChisqRad = None
        self.EvalChisqTan = None
        self.EvalChisqBoth = None
        self.EvalChisqSurf = None

        # details for stepping thru parameters
        self.TrialFiducialRProj = 0.4
        self.TrialMProjEval = None # evaluated mproj with s0 = 1
        self.TrialMProjMin  = 1.0e3
        self.TrialMProjMax  = 1.0e5
        self.TrialMProjStep = 1.0e3
        self.TrialMProjValues = N.array([])
        self.TrialMProj     = None

        # conversion factors
        self.GravConstant = 6.67e-11

        # M(<r) in M_Sol from r(pc), sigma0 (km/s), INCLUDES G
        self.ConvMassToSolar = 231.33  
        self.ConvMassToSolarNoG = 1.54e-8  # does not include G

        # rho(<r) in M_Sol / pc3 from r0 (pc), sigma0 (km/s), INCLUDES
        # G 
        self.ConvRhoToSolar = 18.40

#        self.ConvRhoToSolarNoG = 1.227e-9 # does not include G, DOES Include 4pi


        # parameter-grids for parameter-steps
        self.TrialGrids = None
        self.TrialVects = {}
        self.TrialGridShape = ()
        self.OptGridShape = ()

        # surface density data
        self.SurfDensRProj  = N.array([])
        self.SurfDensNum    = N.array([])
        self.SurfDensNumErr = N.array([])

        # indices for paramgrid
        self.iR0 = 0
        self.iNr = 0
        self.iNt = 0
        self.is0 = 0

        # results value-grids
        self.GridVDisp0    = N.array([])
        self.GridMFiduc2D  = N.array([])
        self.GridChisqRad  = N.array([])
        self.GridChisqTan  = N.array([])
        self.GridChisqSurf = N.array([])
        self.GridSurfCentr = N.array([])
        self.GridPhysical  = N.array([])
        self.GridRho0      = N.array([])
        self.GridMFiduc3d  = N.array([])  # total mass, evaluated at some
                                       # fiducial radius
        
        # dictionary of grids of 

        # we will also calculate chisq_R, chisq_T for optimal sigma_0
        # (which will also be stored). This will lie in a 3D array-set
        # and be output to a separate file
        self.OptGridR0        = N.array([])
        self.OptGridNr        = N.array([])
        self.OptGridNt        = N.array([])
        self.OptGridVDisp0    = N.array([])
        self.OptGridChisqRad  = N.array([])
        self.OptGridChisqTan  = N.array([])
        self.OptGridChisqSurf = N.array([])
        self.OptGridPhysical  = N.array([])

        # 2D version of the results-array if desired
        self.GridResults2D = N.array([])

        self.GridFitsFile = 'RunGridTrials.fits'

        self.NumToWrite = 1000  # write every this trials

        # some status information
        self.TimeStarted = None
        self.CompletedLoops = False

        # set up results arrays etc. for variables we want to keep
        # while optimizing over the rest
        self.SubsetKeysVary  = []
        self.SubsetKeysFixed = []
        self.SubsetVDisp0 = N.array([])
        self.SubsetMFiduc2D = N.array([])
        self.SubsetChisqRad = N.array([])
        self.SubsetChisqTan = N.array([])
        self.SubsetChisqSurf = N.array([])
        self.SubsetRho0     = N.array([])
        self.SubsetMFiduc3d = N.array([])
        self.SubsetDoPlot  = False
        self.SubsetPlotLog = False

        # Fitting-parameters tried
        self.SubsetFitR0 = N.array([])
        self.SubsetFitNr = N.array([])
        self.SubsetFitNt = N.array([])
        self.SubsetFitFinal = N.array([])
        self.SubsetChisqOld = N.array([])

        self.SubsetParams  = {'R0':self.SubsetFitR0, \
                                  'Nr':self.SubsetFitNr, \
                                  'Nt':self.SubsetFitNt, \
                                  'Final':self.SubsetFitFinal}

        # Information when lifting a subset of variables over which to
        # minimize
        self.SubspaceIndices = None
        self.SubspaceKeysFixed = []
        self.SubspaceIndsFixed = N.array([])
        self.SubspaceValsFixed = N.array([])
        self.SubspaceXVec = N.array([])
        self.SubspaceYVec = N.array([])
        self.SubspaceXGrid = N.array([])
        self.SubspaceYGrid = N.array([])
        self.SubspaceZOrig = N.array([])
        self.SubspaceZGrid = N.array([])
        self.SubspaceUGrid = N.array([])
        self.SubspaceXMin = None
        self.SubspaceYMin = None
        self.SubspaceZMin = None
        self.SubspaceHasAnyGood = True
        self.SubspaceDCalc = {}

        # interpolated calculated values
        #
        # store as three dictionaries:
        # X, Y_interp, Chi_Interp
        # for calculated objects called by keyword
        self.SubspaceDInterpX = {}
        self.SubspaceDInterpY = {}
        self.SubspaceDInterpZ = {}
        self.SubspaceDInterpU = {}

        # 1D (i.e. marginal distribution) chisq collapsed values
        self.CollapsedDInterpValue = {}
        self.CollapsedDInterpChisq = {}

        # variables for plot-evaluation
        self.DProfiles = {}
        self.PlotRadii = N.array([])
        self.PlotRadiiN = None
        self.PlotRadiiMax = 2.0
        self.PlotRadiiMin = 0.01
        self.NumProfileSamples = None
        self.iRandomProfiles = N.array([])
        self.iSample = None

        # keys to pass across to evaluated parts
        self.Pars4Eval = ['R0', 'Nr', 'Nt', 's0','SurfDens0']

        # fixing the scale for surfdens?
        self.FixedCentralSurfDens = None

    def SetupTrialRanges(self):

        """Set up the number of trials to use"""

        self.TrialGrids = {'R0':{'Lo':0.4, 'Hi':0.4, 'Num':1}, \
                               'Nr':{'Lo':1.0, 'Hi':1.0, 'Num':1}, \
                               'Nt':{'Lo':1.0, 'Hi':1.0, 'Num':1}, \
                               's0':{'Lo':0.5, 'Hi':10.0, 'Num':10}, \
                               'MProj':{'Lo':500.0, 'Hi':30000.0, 'Num':10}\
                               }

    def SetupTrialParamVecs(self):

        """Generate parameter-vectors"""

        if not N.any(self.TrialGrids):
            return

        KeysToSet = ['R0', 'Nr', 'Nt', 's0', 'MProj']
        for Key in KeysToSet:
            self.TrialVects[Key] = N.linspace(start = self.TrialGrids[Key]['Lo'], \
                                                  stop = self.TrialGrids[Key]['Hi'], \
                                                  num = self.TrialGrids[Key]['Num'], \
                                                  endpoint = True)

    def SetupTrialGridShape(self):

        """Generate shape-vector for grid of trials"""

        # this shape-vector will be used to store:
        #
        # projected mass

        MassKey = 's0'
        if self.UseMProj:
            MassKey = 'MProj'

        try:
            nR = self.TrialGrids['R0']['Num']
            nA = self.TrialGrids['Nr']['Num']
            nB = self.TrialGrids['Nt']['Num']
            nM = self.TrialGrids[MassKey]['Num']
            self.TrialGridShape = (nR, nA, nB, nM)
            self.OptGridShape   = (nR, nA, nB)
        except:
            return

    def SetupResultsGrid(self):

        """Set up grid of results if stepping through trials"""

        if N.size(self.TrialGridShape) < 1:
            return

        self.GridVDisp0    = N.zeros(self.TrialGridShape)
        self.GridMFiduc2D  = N.zeros(self.TrialGridShape)
        self.GridChisqRad  = N.zeros(self.TrialGridShape)
        self.GridChisqTan  = N.zeros(self.TrialGridShape)
        self.GridChisqSurf = N.zeros(self.TrialGridShape)
        self.GridSurfCentr = N.zeros(self.TrialGridShape)
        self.GridPhysical  = N.zeros(self.TrialGridShape)
        self.GridRho0      = N.zeros(self.TrialGridShape)
        self.GridMFiduc3d  = N.zeros(self.TrialGridShape)

    def SetupOptGrid(self):

        """Set up grid of "optimized" grid with s0 scaled
        automatically to fit the data"""
        
        if N.size(self.OptGridShape) < 1:
            return

        self.OptGridR0        = N.zeros(self.OptGridShape)
        self.OptGridNr        = N.zeros(self.OptGridShape)
        self.OptGridNt        = N.zeros(self.OptGridShape)
        self.OptGridVDisp0    = N.zeros(self.OptGridShape)
        self.OptGridChisqRad  = N.zeros(self.OptGridShape)
        self.OptGridChisqTan  = N.zeros(self.OptGridShape)
        self.OptGridChisqSurf = N.zeros(self.OptGridShape)
        self.OptGridPhysical  = N.zeros(self.OptGridShape)

        return

        # populate the parameter-arrays (used for output)
        R0Vec = N.copy(self.TrialVects['R0'])
        NrVec = N.copy(self.TrialVects['Nr'])
        NtVec = N.copy(self.TrialVects['Nt'])

        for iR0 in range(self.OptGridShape[0]):
            self.OptGridR0[iR0, :, : ] = N.copy(R0Vec)

        for iNr in range(self.OptGridShape[1]):
            self.OptGridNr[:,iNr,:] = N.copy(NrVec)
            
        for iNt in range(self.OptGridShape[2]):
            self.OptGridNt[:,:,iNt] = N.copy(NtVec)
        

    def OptGridWriteColumns(self, OptFile = 'RunOptTrials2D.fits'):

        """Given a set of "optimized" results-grids, write to
        column-file for convenient viewing with favorite fits
        viewer"""

        # construct record array
        LArrays = [N.ravel(self.OptGridR0), \
                       N.ravel(self.OptGridNr), N.ravel(self.OptGridNt), \
                       N.ravel(self.OptGridVDisp0), \
                       N.ravel(self.OptGridChisqRad), \
                       N.ravel(self.OptGridChisqTan), \
                       N.ravel(self.OptGridChisqSurf), \
                       N.ravel(self.OptGridPhysical)]

        LUnits = [('R0', N.float), \
                      ('Nr', N.float), ('Nt', N.float), \
                      ('s0', N.float), \
                      ('ChiR', N.float), \
                      ('ChiT', N.float), \
                      ('ChiSurf', N.float), \
                      ('IsPhys', N.float)]

        pyfits.writeto(OptFile, \
                           N.rec.fromarrays(LArrays, LUnits), \
                           clobber=True)

    def GridConvertTo2D(self):
        
        """Translate the results grid into a one-row-per-trial
        format"""

        if not N.any(self.TrialGridShape):
            return

        print N.shape(self.GridChisqRad), N.sum(self.GridChisqRad)
        if not N.any(self.GridChisqRad):
            return

        # some convenience functions
        NumRows  = N.prod(self.TrialGridShape)
        NumR0    = self.TrialGridShape[0]
        NumNr    = self.TrialGridShape[1]
        NumNt    = self.TrialGridShape[2]
        NumMProj = self.TrialGridShape[3]

        # generate record array for trials
        aResults = N.zeros(NumRows, [('R0', N.float), ('Nr', N.float), ('Nt', N.float), \
                                         ('s0', N.float), ('MProj', N.float), \
                                         ('Rho0', N.float), ('M3d', N.float), \
                                         ('ChiR', N.float), ('ChiT', N.float), \
                                         ('SurfDens0', N.float), \
                                         ('ChiSurf', N.float), \
                                         ('IsPhys', N.int)])

        TimeStarted = time.time()
        RowsWritten = 0
        for iR0 in range(NumR0):
            for iNr in range(NumNr):
                for iNt in range(NumNt):

                    try:
                        TheseIndices = (iR0, iNr, iNt)
                        lcount = N.arange(NumMProj,dtype='int')
                        IndicesOut = N.asarray(RowsWritten + lcount, 'int')
                        
                        # pass the input parameters
                        aResults['R0'][IndicesOut]   = self.TrialVects['R0'][iR0]
                        aResults['Nr'][IndicesOut]   = self.TrialVects['Nr'][iNr]
                        aResults['Nt'][IndicesOut]   = self.TrialVects['Nt'][iNt]

                        # pass the results of the fitting
                        aResults['s0'][IndicesOut]     = self.GridVDisp0[TheseIndices][lcount]
                        aResults['MProj'][IndicesOut]  = self.GridMFiduc2D[TheseIndices][lcount]
                        aResults['Rho0'][IndicesOut]   = self.GridRho0[TheseIndices][lcount]
                        aResults['M3d'][IndicesOut]    = self.GridMFiduc3d[TheseIndices][lcount]
                        aResults['ChiR'][IndicesOut]   = self.GridChisqRad[TheseIndices][lcount]
                        aResults['ChiT'][IndicesOut]   = self.GridChisqTan[TheseIndices][lcount]
                        aResults['ChiSurf'][IndicesOut] \
                            = self.GridChisqSurf[TheseIndices][lcount]
                        aResults['SurfDens0'][IndicesOut] \
                            = self.GridSurfCentr[TheseIndices][lcount]
                        aResults['IsPhys'][IndicesOut] = N.asarray( \
                            self.GridPhysical[TheseIndices][lcount], 'int')

                        # end of the loop - increment RowsWritten and continue
                        RowsWritten = RowsWritten + NumMProj
                    except(KeyboardInterrupt):
                        print "GridConvertTo2D INFO - Keyboard interrupt detected. Returning."
                        return


        TimeElapsed = time.time() - TimeStarted
        if self.Verbose:
            print "Conversion of %i elements took %.2f seconds" % (RowsWritten, TimeElapsed)
        # if we get here, pass the results to the class
        self.GridResults2D = aResults

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
        

    def ModelInitialiseResults(self):

        """Initialize important model parameters"""

        self.EvalVDispRad = N.array([])
        self.EvalVDispTan = N.array([])
        self.ModelEnclosedMass2D = N.array([])
        self.EvalDensity0  = None
        self.ModelDensity0 = None  # poorly-named variable
        self.EvalMFiduc3d  = None
        self.EvalChisqRad  = None
        self.EvalChisqTan  = None
        self.EvalChisqSurf = None
        self.EvalChisqBoth = None

        # variables for checking whether model is physical
        self.ModelIsPhysical = True
        self.ModelVolumeDensity = N.array([])  
        
    def GridGetThisModel(self):

        """Transfer current model parameters from the grid"""
        
        self.ModelR0        = self.TrialVects['R0'][self.iR0]
        self.ModelNr        = self.TrialVects['Nr'][self.iNr]
        self.ModelNt        = self.TrialVects['Nt'][self.iNt]
        if not self.UseMProj:
            self.ModelVDisp0    = self.TrialVects['s0'][self.is0]
            self.ModelMFiducial = 1.0
        else:
            self.ModelVDisp0 = 1.0
            self.ModelMFiducial = self.TrialVects['MProj'][self.is0]
            

        # ensure the correct mass-variable is being used in the trial
        self.ModelDispAndMProj()

        # pass this vdisp and fiducial-mass to the results grid
#        self.GridVDisp0 = N.copy(self.ModelVDisp0)
#        self.GridMFiduc = N.copy(self.ModelMFiducial)

    def GridInsertBadVals(self):

        """If model is unphysical, or for whatever other reason,
        insert the badvals into the results-grid"""
        
        self.EvalChisqRad  = self.BadChisq
        self.EvalChisqTan  = self.BadChisq
        self.EvalChisqBoth = self.BadChisq
        self.EvalDensity0  = self.BadParam
        self.EvalMFiduc3d  = self.BadParam

    def GridEvalThisModel(self):

        """Handed some model-parameters, evaluate its fit to data and
        some ancilliary info"""


        # first, check if the model is physical
        self.CheckModelIsPhysical()
        if not self.ModelIsPhysical:
            self.GridInsertBadVals()
            return

        # Evaluate figure of merit
        self.CompareModelToData()

        if not self.DoCalcWhileFitting:
            return

        # evaluate the additional desired parameters. First central
        # density:
        self.PointCentralVolumeDensityFull()
        self.EvalDensity0 = N.copy(self.ModelDensity0)
        
        # then 3D mass interior to 3d fiducial radius
        self.EvalMFiduc3d = self.ArrayEnclosedMass3DFull(self.FiducialR3D)
    
    

    def GridPassEvalToGrid(self):

        """Given a set of evaluations, pass them up to the trial-grid"""

        ThisIndex = (self.iR0, self.iNr, self.iNt, self.is0)

        # whatever we are doing, will need chisq, so pass them up
        self.GridChisqRad[ThisIndex] = N.copy(self.EvalChisqRad)
        self.GridChisqTan[ThisIndex] = N.copy(self.EvalChisqTan)
        self.GridPhysical[ThisIndex] = int(self.ModelIsPhysical)
        self.GridVDisp0[ThisIndex]   = N.copy(self.ModelVDisp0)

        if not self.DoCalcWhileFitting and not self.UseMProj:
            return

        self.GridMFiduc2D[ThisIndex] = N.copy(self.ModelMFiducial)
        self.GridRho0[ThisIndex]     = N.copy(self.EvalDensity0)
        self.GridMFiduc3d[ThisIndex] = N.copy(self.EvalMFiduc3d)

    def GridSetIndices(self, iR0 = 0, iNr = 0, iNt = 0, is0 = 0):

        """Convenience function to set grid indices from arguments
        without interrupting flow during loop"""

        self.iR0 = iR0
        self.iNr = iNr
        self.iNt = iNt
        self.is0 = is0

    def GridTryParamSet(self):

        """Assuming the set of indices is correct, evaluate the model
        at these indices"""

        self.ModelInitialiseResults()
        self.GridGetThisModel()
        self.GridEvalThisModel()
        self.GridPassEvalToGrid()

    def GridRunLoopsVec(self):

        """Vectorize sigma_0 when going through the trials"""

        # this is for a run where we only care about parameters and
        # chisq values, and will be computing everything else in a
        # subsequent pass

        if N.size(self.TrialGridShape) < 1:
            return

        # ensure surface density information is present if needed
        if self.UseSurfDens and not N.any(self.SurfDensRProj):
            self.LoadSurfaceDensity()

        # this is only worth doing if not self.UseMProj. assert here
        self.UseMProj = False
        self.DoCalcWhileFitting = False
        
        # vector of s0, will be used for all trials
        s0Vec = N.copy(self.TrialVects['s0'])

        # data vectors - replicate into grids for rapid computing of
        # chisq point-by-point
        dum, VRadGridData  = N.meshgrid(s0Vec, self.DataVDispRad)
        dum, VRadGridError = N.meshgrid(s0Vec, self.DataEDispRad)
        dum, VTanGridData  = N.meshgrid(s0Vec, self.DataVDispTan)
        dum, VTanGridError = N.meshgrid(s0Vec, self.DataEDispTan)

        gshape = self.TrialGridShape
        RunTotal = gshape[0] * gshape[1] * gshape[2]
        RunSoFar = 0
        self.TimeStarted = time.time()

        # loop thru R0, Nr and Nt
        is0 = 0
        for iR0 in range(gshape[0]):

            # evaluate chisq rad - will be the same for all the points
            #
            # I find it less confusing to pass the radius straight in
            # rather than set parts of the model at two places in the
            # master-loop
            if self.UseSurfDens:
                ThisR0 = self.TrialVects['R0'][iR0]
                SurfChisq, SurfScale = self.EvalChisqSurfDensPoint(ThisR0)
                self.GridSurfCentr[iR0]  = SurfScale # pass central surf dens too
                self.GridChisqSurf[iR0]  = SurfChisq
                self.OptGridChisqSurf[iR0] = SurfChisq

            for iNr in range(gshape[1]):
                for iNt in range(gshape[2]):

                    try:
                        # indices for the grid - leave s0 blank 
                        ThisVecIndices = (iR0, iNr, iNt)

                        # whatever happens, need to pass s0 vector to its
                        # appropriate gridspace
                        self.GridVDisp0[ThisVecIndices] = s0Vec

                        # is0 is a dummy 
                        self.ModelInitialiseResults()
                        self.GridSetIndices(iR0, iNr, iNt, 0)
                        self.GridGetThisModel()
                        self.ModelVDisp0 = 1.0  # dummy for projection-integral
                    
                        # if R0, Nr, Nt lead to unphysical density, cannot
                        # continue. Propogate badchisq into ALL the chisq
                        # for s0 for this particular combination
                        self.CheckModelIsPhysical()
                        self.GridPhysical[ThisVecIndices] = int(self.ModelIsPhysical)
                        self.OptGridPhysical[ThisVecIndices] \
                            = int(self.ModelIsPhysical) 
                        # propogate through chisq even if unphysical
                        # so we can decide what to do later
#                        if not self.ModelIsPhysical:
#                            self.GridChisqRad[ThisVecIndices] = self.BadChisq
#                            self.GridChisqTan[ThisVecIndices] = self.BadChisq
#x                            continue
                    
                        # evaluate projection-integrals for each of the
                        # datapoints, where we have set s0 == 1
                        self.ArrayDispOnSky()
                        s0Grid, VRadGridEval = N.meshgrid(s0Vec, self.EvalVDispRad)
                        dum,    VTanGridEval = N.meshgrid(s0Vec, self.EvalVDispTan)

                        # actually scale the evaluation by s0 upfront
                        VRadGridEval = VRadGridEval * s0Grid
                        VTanGridEval = VTanGridEval * s0Grid
                    
                        # now find the ingredients of chisq for every
                        # datapoint at every s0 for this paramset...
                        ChiRadParts = ((VRadGridData - VRadGridEval) / VRadGridError)**2
                        ChiTanParts = ((VTanGridData - VTanGridEval) / VTanGridError)**2

                        # ... and sum along the data-vector
                        ChiRadVec = N.sum(ChiRadParts,0)
                        ChiTanVec = N.sum(ChiTanParts,0)

                        # Propogate the resulting chisq up to the
                        # class results values
                        self.GridChisqRad[ThisVecIndices] = N.copy(ChiRadVec)
                        self.GridChisqTan[ThisVecIndices] = N.copy(ChiTanVec)

                        # find the miminum chisq and pass to the optgrid
                        #
                        # note this performs the same was as the
                        # passing to the full grid above, since the
                        # final index does not need to be specified
                        # when passing a vector.
                        iBestS0 = N.argmin(ChiRadVec + ChiTanVec)
                        self.OptGridVDisp0[ThisVecIndices]   = s0Vec[iBestS0]
                        self.OptGridChisqRad[ThisVecIndices] = ChiRadVec[iBestS0]
                        self.OptGridChisqTan[ThisVecIndices] = ChiTanVec[iBestS0]

                        # pass parameters actually fit across to the
                        # param-grids for output
                        self.OptGridR0[ThisVecIndices] = self.ModelR0
                        self.OptGridNr[ThisVecIndices] = self.ModelNr
                        self.OptGridNt[ThisVecIndices] = self.ModelNt

                    except(KeyboardInterrupt):
                        print "GridRunLoopsVec DBG -- Detected keyboard interrupt. Returning after %i of %i trials (%.2f percent)" % (RunSoFar, RunTotal, 100.0*RunSoFar / (1.0*RunTotal))
                        return

                    RunSoFar = RunSoFar + 1

                    # write during the trials so can check
                    if RunSoFar % self.NumToWrite == 0 \
                            and RunSoFar > self.NumToWrite:
                        self.GridWriteFits()
                        if self.Verbose:
                            TimeElapsed = time.time() - self.TimeStarted
                            TimePerTrial = TimeElapsed / (1.0 * RunSoFar) 
                            StringEnd = self.StatusTimeString(RunSoFar, RunTotal)

                            sys.stdout.write("\r GridRunLoopsVec INFO: %i of %i - elapsed %.1f s, avg per trial %.2f s ; expect finish %s" \
                                                 % (RunSoFar, RunTotal, \
                                                        TimeElapsed, TimePerTrial, \
                                                        StringEnd))
                            sys.stdout.flush()

        # final status variable
        self.CompletedLoops = True

        # write at end of all the trials
        self.GridWriteFits()

    def GridRunLoops(self):

        """Run the loops through trial values"""

        if N.size(self.TrialGridShape) < 1:
            return

        gshape = self.TrialGridShape
        RunTotal = N.prod(gshape)
        RunSoFar = 0
        self.TimeStarted = time.time()
        for iR0 in range(gshape[0]):
            for iNr in range(gshape[1]):
                for iNt in range(gshape[2]):
                    for is0 in range(gshape[3]):

                        try:
                            self.GridSetIndices(iR0, iNr, iNt, is0)
                            self.GridTryParamSet()
                        except(KeyboardInterrupt):
                            print "GridRunLoops INFO -- Detected keyboard interrupt. Returning after %i of %i trials (%.2f percent)" % (RunSoFar, RunTotal, 100.0*RunSoFar / (1.0*RunTotal))

                            return
#                        except:
#                            dumdum = 1

                        RunSoFar = RunSoFar + 1

                        # write during the trials so can check
                        if RunSoFar % self.NumToWrite == 0 \
                                and RunSoFar > self.NumToWrite:
                            self.GridWriteFits()
                            if self.Verbose:
                                TimeElapsed = time.time() - self.TimeStarted
                                TimePerTrial = TimeElapsed / (1.0 * RunSoFar) 
                                StringEnd = self.StatusTimeString(RunSoFar)

                                sys.stdout.write("\r GridRunLoops INFO: %i of %i - elapsed %.1f s, avg per trial %.2f s ; expect finish %s" \
                                                     % (RunSoFar, RunTotal, \
                                                            TimeElapsed, TimePerTrial, \
                                                            StringEnd))
                                sys.stdout.flush()

        self.CompletedLoops = True

        # write at end of all the trials
        self.GridWriteFits()

    def StatusTimeString(self, RunSoFar = 0, NTotal = None, TimeStarted = None):

        """Set up string to report time till routine finishes"""

        if not TimeStarted:
            TimeStarted = self.TimeStarted
        
        if not TimeStarted:
            return ''

        if NTotal:
            RunTotal = N.copy(NTotal)
        else:
            RunTotal = N.prod(self.TrialGridShape)

        TimeElapsed = time.time() - TimeStarted
        TimePerTrial = TimeElapsed / (1.0 * RunSoFar)
        TimeRemain = (RunTotal - RunSoFar) * TimePerTrial 
        TimeEnd = TimeStarted + TimeElapsed + TimeRemain
        StruEnd = time.localtime(TimeEnd)
        
        StrYea = string.zfill("%i" % (StruEnd[0]), 4)
        StrMon = string.zfill("%i" % (StruEnd[1]), 2)
        StrDay = string.zfill("%i" % (StruEnd[2]), 2)
        StrHou = string.zfill("%i" % (StruEnd[3]), 2)
        StrMin = string.zfill("%i" % (StruEnd[4]), 2)
        StrSec = string.zfill("%i" % (StruEnd[5]), 2)

        TimeString = "%s-%s-%s at %s:%s:%s" % \
            (StrYea, StrMon, StrDay, StrHou, StrMin, StrSec)

        return TimeString

    def GridWriteFits2D(self, OutFits = None):

        """Write 2D version of trials-file to fits"""

        if not OutFits:
            return
        
        if N.size(self.GridResults2D) < 2:
            return

        pyfits.writeto(OutFits, self.GridResults2D, clobber = True)

    def GridWriteFits(self):

        """Write trials to fits file"""

        if os.access(self.GridFitsFile, os.R_OK):
            os.remove(self.GridFitsFile)

        # write out the param vectors first...
        pyfits.writeto(self.GridFitsFile,  self.TrialVects['R0'], clobber=True)
        pyfits.append(self.GridFitsFile,  self.TrialVects['Nr'])
        pyfits.append(self.GridFitsFile,  self.TrialVects['Nt'])

        if self.UseMProj:
            pyfits.append(self.GridFitsFile, self.TrialVects['MProj'])
        else:
            pyfits.append(self.GridFitsFile, self.TrialVects['s0'])

        # write out the chisq values
        pyfits.append(self.GridFitsFile, self.GridChisqRad)
        pyfits.append(self.GridFitsFile, self.GridChisqTan)
        pyfits.append(self.GridFitsFile, self.GridChisqSurf)
        pyfits.append(self.GridFitsFile, self.GridPhysical)
        pyfits.append(self.GridFitsFile, self.GridSurfCentr)

        # this should be initialised already even if not populated
        pyfits.append(self.GridFitsFile, self.GridMFiduc2D)
        pyfits.append(self.GridFitsFile,  self.GridVDisp0)
        pyfits.append(self.GridFitsFile,  self.GridRho0)
        pyfits.append(self.GridFitsFile,  self.GridMFiduc3d)

        # construct record array from data...
        LData = [self.R2D, \
                     self.DataVDispRad, self.DataVDispTan, \
                     self.DataEDispRad, self.DataEDispTan]
        LUnits = [('R', N.float), \
                      ('DispRad', N.float), ('DispTan', N.float), \
                      ('E_DispRad', N.float), ('E_DispTan', N.float)]
        RecWrite = N.rec.fromarrays(LData, LUnits)

        # ... and append to trials-file
        pyfits.append(self.GridFitsFile, RecWrite)

        # then the param vectors
#        pyfits.append(self.GridFitsFile,  self.TrialVects['R0'])
#        pyfits.append(self.GridFitsFile,  self.TrialVects['Nr'])
#        pyfits.append(self.GridFitsFile,  self.TrialVects['Nt'])

        # include whichever param was used as a mass deviate
#        if self.UseMProj:
#            pyfits.append(self.GridFitsFile, self.TrialVects['MProj'])
#        else:
#            pyfits.append(self.GridFitsFile, self.TrialVects['s0'])
            
    def GridReadFitsLegacy(self, InFile = None):

        """Read trials file"""

        if not InFile:
            return

        if not os.access(InFile,os.R_OK):
            return

        hdulist = pyfits.open(InFile)
        self.GridMFiduc2D  = hdulist[0].data
        self.GridVDisp0    = hdulist[1].data
        self.GridRho0      = hdulist[2].data
        self.GridMFiduc3d  = hdulist[3].data
        self.GridChisqRad  = hdulist[4].data
        self.GridChisqTan  = hdulist[5].data

        self.TrialVects = {}
        self.TrialVects['R0'] = hdulist[6].data
        self.TrialVects['Nr'] = hdulist[7].data
        self.TrialVects['Nt'] = hdulist[8].data
        self.TrialVects['Final'] = hdulist[9].data

        self.TrialGridShape = N.shape(self.GridChisqTan)

        hdulist.close()
        
    def GridReadFits(self, InFile = None):

        """Read trials file. Now includes chisqsurf (which will always
        be present even if not filled)"""

        if not InFile:
            return

        if not os.access(InFile,os.R_OK):
            return

        hdulist = pyfits.open(InFile)

        self.TrialVects = {}
        self.TrialVects['R0'] = hdulist[0].data
        self.TrialVects['Nr'] = hdulist[1].data
        self.TrialVects['Nt'] = hdulist[2].data
        self.TrialVects['Final'] = hdulist[3].data

        self.GridChisqRad  = hdulist[4].data
        self.GridChisqTan  = hdulist[5].data
        self.GridChisqSurf = hdulist[6].data
        self.GridPhysical  = hdulist[7].data
        self.GridSurfCentr = hdulist[8].data

        self.TrialGridShape = N.shape(self.GridChisqTan)

        if len(hdulist) > 9:
            self.GridMFiduc2D = hdulist[9].data
        if len(hdulist) > 10:
            self.GridVDisp0   = hdulist[10].data
        if len(hdulist) > 11:
            self.GridRho0     = hdulist[11].data
        if len(hdulist) > 12:
            self.GridMFiduc3d = hdulist[12].data

        hdulist.close()

    def CheckParsSet3D(self):
        
        """Check if all parameters exist for 3D enclosed mass-model"""
        
        RetVal = True
        try:
            dummy = 1.0 * self.ModelR0
            dummy = 1.0 * self.ModelNr
            dummy = 1.0 * self.ModelNt
            dummy = 1.0 * self.ModelVDisp0
        except:
            RetVal = False

        return RetVal

    def EvalEnclosedMass3DKinem(self, r_in = N.array([]) ):
        
        """Find the mass enclosed in a plummer model with kinematic
        information"""

        # check if all params are set
        if N.any(r_in):
            rUse = r_in
        else:
            rUse = self.r3d

        if not N.any(rUse):
            return

        if not self.CheckParsSet3D():
            return

        # convenience variables
        r0 = self.ModelR0
        nr = self.ModelNr
        nt = self.ModelNt
        sigma0 = self.ModelVDisp0
        
        s = rUse/r0
        u = (1.0 + s**2)

        GM = rUse * sigma0**2 * ( \
            s**2 * (5.0 + nr) * u**(0.0 - 0.5 * (nr + 2.0)) \
                + 2.0 * u**(-0.5 * nt) - 2.0 * u**(-0.5 * nr)
            )

        # with sigma_0 in km/s, r in pc, conversion factor is 463.2
        self.ModelEnclosedMass3DKinem = GM * 463.2

    # convention for integrating functions:
    #
    # "Point*" means "evaluated at a single limit"
    #
    # "Array" means "evaluated over an array of values"

    # get enclosed 3D mass for spheres of radius r. Vectorized.
    def ArrayEnclosedMass3D(self, inrvec = N.array([]), inrho = None):

        """For an array of radii, evaluate the plummer enclosed mass
        within each radius in 3D"""

        if N.any(inrvec):
            rvec = inrvec
        else:
            rvec = self.r3d

        if not N.any(rvec):
            return

        # for a vector application will usually be taking the model
        # parameters.
        if N.any(inrho):
            rho0 = inrho
        else:
            if not N.any(self.ModelDensity0):
                self.PointCentralVolumeDensityFull()
            rho0 = self.ModelDensity0

        try:
            r0 = 1.0 * self.ModelR0
        except:
            return

        # evaluate analytic - this is already vectorized
        MAnalytic =  rvec**3 * (rvec**2 + r0**2)**(-3.0/2.0) 

        # scale by the mass at infinity
        MInfty = self.PointEnclosedMass3D(Inf, rho0)

        return MAnalytic * MInfty[0]

    # evaluate 3D mass integral
    def PointEnclosedMass3D(self, inr = None, inrho = None):

        """From plummer model parameters, evaluate the mass enclosed
        within a sphere of radius inr"""

        try:
            dum = 1.0 * inr
        except:
            return

        # if evaluating the mass, input inrho. If evaluating the mass
        # apart from rho0 (in order to find rho0), set rho0 = 1.0
        if N.any(inrho):
            rho0 = inrho
        else:
            rho0 = 1.0
           
        try:
            r0 = 1.0 * self.ModelR0
        except:
            return

        # Now can conduct the integral
        Integ3Dr = quad(lambda r: r**2 * (1.0 + (r/r0)**2)**(-5.0/2.0) , \
                            0, inr)

        Integ3Dr = N.asarray(Integ3Dr) * 4.0 * N.pi * rho0

        return Integ3Dr

    # evaluate projected mass integral
    def PointEnclosedMass2D(self, inR = None):

        """From Plummer model parameters, evaluate the mass enclosed
        within a cylinder of radius R"""

        try:
            dum = 1.0 * inR
        except:
            return

        # don't want to use self.r2D here - will confuse if vector
        RUse = N.copy(inR)

        if not N.any(self.ModelDensity0):
            self.PointCentralVolumeDensityFull()
        
        # check that all parameters are present
        try:
            rho0 = 1.0 * self.ModelDensity0
            r0   = 1.0 * self.ModelR0
        except:
            return

        # now evaluate the integral
        #
        # symbols: "R" --> "R2"  ; "r" --> r3
        Integral = dblquad(lambda r3, R2: \
                               (R2 * r3) \
                               / (N.sqrt(r3**2 - R2**2) \
                                      * (1.0 + (r3/r0)**2)**(5.0/2.0)), \
                               0, RUse, \
                               lambda x: x, lambda x: Inf)

        Integral = Integral * 4.0 * N.pi * rho0

        return Integral

    def ArraySurfDensInteg(self, inRadii = None):
        
        """Evaluate the denominator in the velocity-projection
        expressions in a vectorized way."""

        # This is n(r) of tracers, not rho(r) of the model.

        try: 
            radii = inRadii * 1.0
        except:
            return

        # we can do this by noting the curve shape is a known function
        # and multiplying up by the value evaluated at r=0

        # curve shape
        DensCalc = (1.0 + (radii/self.ModelR0)**2)**(-2.0)

        # lower limit
        LowLimit = self.PointSurfDensInteg(0.0)

        return DensCalc * LowLimit[0]

    def PointSurfDensInteg(self, inR = None):

        """Evaluate the denominator in the velocity-projection
        expressions"""

        # we want to be able to pass in zero lower limit for radius...
        try:
            Rproj = inR * 1.0
        except:
            return

        try:
            r0 = 1.0 * self.ModelR0
        except:
            return

        IntegSurf = quad( lambda r: r * (1.0 + (r/r0)**2)**(-2.5) \
                              / N.sqrt(r**2 - Rproj**2), \
                              Rproj, Inf)

        return IntegSurf

    def ArrayDispOnSky(self, inRadii = N.array([])):

        """Given a plummer model, evaluate the velocity dispersions
        on-sky"""

        if N.size(inRadii) > 0:
            R = inRadii
        else:
            R = self.R2D

        try:
            dum = 1.0 * R
        except:
            return
        
        # initialise output vectors
        NumerRad = N.copy(R) * 0.0
        NumerTan = N.copy(NumerRad)
        Denom    = N.copy(NumerRad)
        VelRad   = N.copy(NumerRad)
        VelTang  = N.copy(NumerRad)

        # the same denominator for both: the integral inside the
        # surface-density profile
        Denom = self.ArraySurfDensInteg(R)
        
        # now populate both the radial and tangential arrays
        for iRad in range(N.size(R)):
            RThis = R[iRad]
            NumerRad[iRad] = self.PointDispRadInteg(RThis)[0]
            NumerTan[iRad] = self.PointDispTanInteg(RThis)[0]

        VelRad = NumerRad / Denom
        VelTan = NumerTan / Denom

#        print "Arraydisponsky DBG:", N.min(VelRad), N.max(VelRad), self.ModelVDisp0

        # want the sigma not the variance
        self.EvalVDispRad = N.sqrt(VelRad) * self.ModelVDisp0
        self.EvalVDispTan = N.sqrt(VelTan) * self.ModelVDisp0

    def ArrayDispRadInteg(self, inRadii = None, DoPlot = False):

        """For a set of radii, evaluate the projected radial
        velocities"""

        # this is nearly identical to ArrayDispTangInteg - consider
        # bringing both under the same function. Could then share the
        # denominator between the two

        try:
            R = inRadii
        except:
            R = self.R2D

        if N.size(R) < 1:
            return
            

        # initialise output vectors
        Numer = N.copy(inRadii) * 0.0
        Denom = N.copy(Numer)
        VelRad = N.copy(Numer)

        # compute the denom first, it's vectorized
        #
        # (this could be passed to the class as it's exactly the same
        # for both radial and tangential.
        Denom = self.ArraySurfDensInteg(inRadii)

        for iRad in range(N.size(inRadii)):
            dum = self.PointDispRadInteg(inRadii[iRad])
            Numer[iRad] = dum[0]

        VelRad = Numer / Denom

        if DoPlot:
            try:
                P.close()
            except:
                dum = 1
            P.figure()
            P.subplots_adjust(hspace = 0.3)
            P.clf()
            P.subplot(211)
            P.plot(inRadii, Numer, 'go')
            P.ylabel('Numerator')
            P.title('Projections of velocities on radial')

#            P.subplot(312)
            P.plot(inRadii, Denom, 'bo')
            P.ylabel('Components')
            
            P.subplot(212)
            P.plot(inRadii, VelRad, 'ks')
            P.xlabel('R')
            P.ylabel('Sigma_R(R)')
            
        return VelRad


    def PointDispRadInteg(self, inR = None, DoPlot = False):

        """Evaluate the numerator of the radial-velocity projection"""

        try:
            Rproj = 1.0 * inR
        except:
            return

        try:
            nr = 1.0 * self.ModelNr
            nt = 1.0 * self.ModelNt
            sigma0 = 1.0 * self.ModelVDisp0
            r0 = 1.0 * self.ModelR0
        except:
            return

        IntegRad = quad( lambda r: self.DispIntegrandRad(r,Rproj) ,
                         Rproj, Inf)
        
        # expression E3 in the paper, row 2
#        IntegRad = quad( lambda r: r * (1.0 + (r/r0)**2)**(-5.0/2.0) \
#                             * ( \
#                (Rproj / r)**2 \
#                    * (1.0 + (r/r0)**2)**(-nr/2.0) \
#                    + (1.0 - (Rproj/r)**2) \
#                    * (1.0 + (r/r0)**2)**(-nt/2.0) \
#                    ) \
#                             /  N.sqrt(r**2 - Rproj**2), \
#                             Rproj, Inf)

#        FunctionRad = N.asarray(IntegRad) * sigma0**2

        return N.asarray(IntegRad)

    def DispIntegrandRad(self, r3d = None, R2D = None):
        
        """For the radial dispersions, this gives the functional form
        to integrate"""

        u = 1 + (r3d / self.ModelR0)**2
        Part1 = r3d / N.sqrt(r3d**2 - R2D**2)
        Part2a = (R2D / r3d)**2.0     * u**(-0.5*(5.0 + self.ModelNr))
        Part2b = (1.0 - (R2D/r3d)**2) * u**(-0.5*(5.0 + self.ModelNt))

        FunctionRad = Part1 * (Part2a + Part2b)

        return FunctionRad        

    def DispIntegrandTan(self, r3d = None, R2D = None):

        """For the tangential dispersions, this gives the functional
        form to integrate"""

        u = 1 + (r3d / self.ModelR0)**2
        Part1 = r3d / N.sqrt(r3d**2 - R2D**2)
        Part2 = u**(-0.5*(5.0 + self.ModelNt))

        return Part1 * Part2

    def ArrayDispTanInteg(self, inRadii = None, DoPlot = False):

        """For a set of radii, evaluate the projected tangential
        velocities"""

        if not N.any(inRadii):
            return

        # initialise output vectors
        Numer = N.copy(inRadii) * 0.0
        Denom = N.copy(Numer)
        VelTang = N.copy(Numer)

        # compute the denom first, it's vectorized
        Denom = self.ArraySurfDensInteg(inRadii)
            
        # now loop through the radii
        for iRad in range(N.size(inRadii)):
            dum = self.PointDispTanInteg(inRadii[iRad])
            Numer[iRad] = dum[0]

        VelTang = Numer / Denom

        if DoPlot:
            try:
                P.close()
            except:
                dum = 1
            P.figure()
            P.subplots_adjust(hspace = 0.3)
            P.clf()
            P.subplot(211)
            P.plot(inRadii, Numer, 'go')
            P.ylabel('Numerator')
            P.title('Projection on tangential motions')

#            P.subplot(312)
            P.plot(inRadii, Denom, 'bo')
            P.ylabel('Components')
            
            P.subplot(212)
            P.plot(inRadii, VelTang, 'ks')
            P.xlabel('R')
            P.ylabel('Sigma_T(R)')
            
        return VelTang

    def PointDispTanInteg(self, inR = None):

        """Evaluate the numerator of the tangential-velocity
        projection"""

        # is the input projected radius useful?
        try:
            Rproj = 1.0 * inR
        except:
            return

        # are all the needed parameters set?
        try:
            nt = 1.0 * self.ModelNt
            sigma0 = 1.0 * self.ModelVDisp0
            r0 = 1.0 * self.ModelR0
        except:
            return

        # conduct the integral
#        IntegTang = quad(lambda r: r * (1.0 + (r/r0)**2)**(-0.5*(5.0 + nt)) \
#                             / N.sqrt(r**2 - Rproj**2), \
#                             Rproj, Inf)

        IntegTan = quad(lambda r: self.DispIntegrandTan(r, Rproj), \
                            Rproj, Inf)

        return IntegTan

    def ArrayVDisp3Dr(self, inr = N.array([])):

        """Under Plummer, evaluate the radial and tangential
        dispersions"""
        
        try:
            r = 1.0 * inr
        except:
            return

        u = (1.0 + ( r /self.ModelR0 ) )**2

        return self.ModelVDisp0 * u**(-self.ModelNr / 4.0)

    def ArrayVDisp3Dt(self, inr = N.array([])):

        """Under Plummer, evaluate the radial and tangential
        dispersions"""
        
        try:
            r = 1.0 * inr
        except:
            return

        u = (1.0 + ( r /self.ModelR0 ) )**2

        return self.ModelVDisp0 * u**(-self.ModelNr / 2.0)


    def ArrayVDisp3Dr(self, inr = N.array([])):
    
        """Under Plummer, evaluate the radial and tangential
        dispersions"""
        
        try:
            r = 1.0 * inr
        except:
            return

        u = (1.0 + ( r /self.ModelR0 ) )**2

        return self.ModelVDisp0 * u**(-self.ModelNt / 2.0)         


    def ArrayEnclosedMass3DFull(self, inr = N.array([])):

        """Evaluate the full form of the enclosed mass in the plummer
        model"""

        if not N.any(inr):
            r = self.r3d
        else:
            r = inr

        if not N.any(r):
            return

        try:
            dum = 1.0 * r
        except:
            return

        # convenience variables
        r0 = self.ModelR0
        Nr = self.ModelNr
        Nt = self.ModelNt
        s0 = self.ModelVDisp0

        u = 1.0 + (r/r0)**2
        part1 =  (r/r0)**2 * (5.0 + Nr) * u**(-0.5*(Nr + 2.0))
        part2 =  2.0 * u**(-0.5 * Nt)
        part3 = -2.0 * u**(-0.5 * Nr)

        GM = r * s0**2 * (part1 + part2 + part3)
        
        # uses msol
        MEnclosed3D = GM * self.ConvMassToSolar

#        MEnclosed3D = GM / self.GravConstant
        return MEnclosed3D

    def PointCentralVolumeDensityFull(self):

        """Given a set of parameters, estimate the central volume
        density"""

        # convenience variables
        try:
            s0 = 1.0 * self.ModelVDisp0
            r0 = 1.0 * self.ModelR0
            Nr = 1.0 * self.ModelNr
            Nt = 1.0 * self.ModelNt
        except:
            return

#        RhoPart1 = s0**2 / (4.0 * N.pi * r0**2 * self.GravConstant)
        RhoPart1 = s0**2 * self.ConvRhoToSolar / (r0**2)  
        RhoPart2 = 15.0 + 5.0*Nr - 2.0*Nt

        self.ModelDensity0 = RhoPart1 * RhoPart2

    def ArrayCheckPhysicalSimplified(self):

        """Dummy function - set alpha = beta and see if rho(r) < 0 anywhere"""

        u = 1.0 + ( self.r3d / self.ModelR0)**2
        brack = 3.0 - (self.r3d / self.ModelR0)**2 * (self.ModelNr + 2.0) / u
        
        return brack
        

    def ArrayVolumeDensityFull(self, inr = N.array([]), DoPassUp = False):

        """Given a set of model-parameters and radii, evaluate the
        mass density profile"""

        if N.any(inr):
            r = inr
        else:
            r = self.r3d

        try:
            dum = 1.0 * r
        except:
            return

        try:
            s0 = 1.0 * self.ModelVDisp0
            Nr = 1.0 * self.ModelNr
            Nt = 1.0 * self.ModelNt
            r0 = 1.0 * self.ModelR0
        except:
            return


        # call the two parts of this function here
        Const = self.EvalRho3DConstant()
        Brack = self.EvalRho3DIntegral(r)
        
        rho = Brack * Const
        
        # longhand might be faster...
        DoLongHand = False
        if DoLongHand:
            u = (1.0 + (r/r0)**2)
            Const = s0**2 / (4.0 * N.pi * r0**2 * self.GravConstant)

            Argu1 = u**(-0.5 * (Nr + 2.0))
            Brac1 = 2.0 * Nr + (5.0 + Nr)*(3.0 - (r/r0)**2 * (Nr+2.0) / u )
            
            Argu2 = u**(-0.5 * (Nt + 2.0))
            Brac2 = -2.0 * Nt

            # MISSES BRACKET 3 - CHECK!!

            rho = Const * ( Argu1*Brac1 + Argu2*Brac2)

        # useful if called alone rather than part of an integral...
        if DoPassUp:
            self.ModelVolumeDensity = N.copy(rho)

        return rho

    def ArrayEnclosedMass2DFull(self, InRadii=N.array([])):

        """Evaluate enclosed mass versus radius"""

        # ensure the input radius vector is properly set
        try:
            R = 1.0 * InRadii
        except:
            R = 1.0 * self.r2D

        try:
            dum = 1.0 * R
        except:
            return
            
        NPoints = N.size(R)
        self.ModelEnclosedMass2DKinem = N.zeros(NPoints)

        if self.DoSurfDens2D:
            self.ModelSurfDens2DKinem = N.zeros(NPoints)

        RMax = N.max(R)

        for iPoint in range(NPoints):
            if self.Verbose:
                sys.stdout.write("\r ArrayEnclosedMass2D: at radius %.2f of %.2f; iteration %i of %i" % (R[iPoint], RMax, iPoint, NPoints))
                sys.stdout.flush()
            ThisInteg = self.PointEnclosedMass2DFull(R[iPoint])
            self.ModelEnclosedMass2DKinem[iPoint] = N.copy(ThisInteg[0])

            # might be doing the 2D surface density as well...
            if not self.DoSurfDens2D:
                continue

            ThatInteg = self.PointSurfDens2DFull(R[iPoint])
            self.ModelSurfDens2DKinem[iPoint] = ThatInteg[0]

    def PointEnclosedMass2DFull(self, inR = None, FindingSigma = False):

        """Evaluate the full projected mass interior to radius R using
        the full set of model-parameters. FindingSigma sets sigma_0 to
        1 (we might be trying to evaluate this number)"""

        # want to be able to try at R = 0 too, to ensure this goes
        # down to 0 at R = 0.0

        # is the input projected radius set?
        try:
            Rup = 1.0 * inR
        except:
            return
        
        # model parameters will need to be correctly set for the
        # integration to work
        try:
            dum = 1.0 * self.ModelVDisp0
            dum = 1.0 * self.ModelR0
            dum = 1.0 * self.ModelNr
            dum = 1.0 * self.ModelNt
        except:
            return

        UpperLim = Inf
        if self.UseFiniteUpper:
            UpperLim = 5000.0
        
            # do double-integral? or do single-integral then integrate
            # numerically??

        # OK now try the double-integral
#$        MProj = dblquad(lambda r3d, R2d: (R2d * r3d / N.sqrt(r3d**2  - R2d**2)) \
 #           * self.ArrayVolumeDensityFull(r3d), \
 #                           0.0, Rup, \
 #                           lambda x: x, lambda x:UpperLim)

        # evaluate the double-integral without the up-front constants
        MInteg = dblquad(lambda r3d, R2d: (R2d * r3d / N.sqrt(r3d**2  - R2d**2)) \
                             * self.EvalRho3DIntegral(r3d), \
                             0.0, Rup, \
                             lambda x: x, lambda x:UpperLim)
        
        # evaluate the up-front constant
        MConst = self.EvalRho3DConstant(EstimatingSigma = FindingSigma) 

        # that gives the constants fed through from the volume
        # density. Need the 4pi upfront to finish the projection
        MProj = N.copy(MInteg) * MConst * 4.0 * N.pi

        return MProj

    def ArraySurfDens2DFull(self, InRadii = N.array([])):

        """Generate array of surface densities given projected radii
        R"""

        try:
            R = 1.0 * InRadii
        except:
            R = 1.0 * self.r2D

        try:
            dum = 1.0 * R
        except:
            return
            
        NPoints = N.size(R)
        self.ModelSurfDens2DKinem = N.zeros(NPoints)

        for iPoint in range(NPoints):
            ThatInteg = self.PointSurfDens2DFull(R[iPoint])
            self.ModelSurfDens2DKinem[iPoint] = ThatInteg[0]

    def PointSurfDens2DFull(self, inR = None):

        """Evaluate the surface density at projected radius R from
        full kinematic model"""

        # is the input projected radius set?
        try:
            RThis = 1.0 * inR
        except:
            return
        
        # model parameters will need to be correctly set for the
        # integration to work
        try:
            dum = 1.0 * self.ModelVDisp0
            dum = 1.0 * self.ModelR0
            dum = 1.0 * self.ModelNr
            dum = 1.0 * self.ModelNt
        except:
            return

        # 2 * integ_R^Inf (r/sqrt * volume density)

        SurfDens = quad(lambda r: r/N.sqrt(r**2 - RThis**2) \
                            * self.ArrayVolumeDensityFull(r), \
                            RThis, Inf) 

        return SurfDens * N.array([2.0, 2.0])
       
    def EvalRho3DIntegral(self, inr = None):

        """Evaluate the 3D density function without the up-front
        constants"""

        # ensure radius-array is set in some way
        try:
            r = 1.0 * inr
        except:
            r = self.r3d

        try:
            dum = 1.0 * r
        except:
            return

        # are the radius-variables set?
        try:
            Nr = 1.0 * self.ModelNr
            Nt = 1.0 * self.ModelNt
            r0 = 1.0 * self.ModelR0
        except:
            return

        u = (1.0 + (r/r0)**2)
        Argu1 = u**(-0.5 * (Nr + 2.0))
        Brac1 = 2.0 * Nr + (5.0 + Nr)*(3.0 - (r/r0)**2 * (Nr+2.0) / u )

        Argu2 = u**(-0.5 * (Nt + 2.0))
        Brac2 = -2.0 * Nt

        # don't forget the anisotropy arguments
        Argu3 = +2.0
        Brac3 = (r0 / r)**2 * (u**(-0.5 * Nt) - u**(-0.5*Nr) )

        RhoEval = Argu1*Brac1 + Argu2*Brac2 + Argu3*Brac3

        # return as an argument AND pass up to the class
        self.EvalRhoIntegral = N.copy(RhoEval)
        return RhoEval

    def EvalRho3DConstant(self, EstimatingSigma = False):
 
        """Evaluate the up-front constants in rho(r)"""

        try:
            r0 = 1.0 * self.ModelR0
            s0 = 1.0 * self.ModelVDisp0
        except:
            return

        if EstimatingSigma:
            s0 = 1.0 

#        RhoConstant = s0**2 / (4.0 * N.pi * self.GravConstant * r0**2)

        # constants to take integral to msol / pc^3 are now pre-evaluated.
        RhoConstant = self.ConvRhoToSolar * s0**2 / r0**2

        # pass up to the class AND return so can use this in the integrator
        self.EvalRhoConstant = N.copy(RhoConstant)
        return RhoConstant

    def GeneratingToModel(self,ParsToPass = []):

        """Pass generating-function parameters to the model for use
        when computing"""

        self.ModelR0        = N.copy(self.TestModelR0)
        self.ModelNr        = N.copy(self.TestModelNr)
        self.ModelNt        = N.copy(self.TestModelNt)
        self.ModelVDisp0    = N.copy(self.TestModelVdisp0)
        self.ModelDensity0  = N.copy(self.TestModelDensity0)
        self.ModelMFiducial = N.copy(self.TestModelMFiducial)

    def ModelDispAndMProj(self):

        """Given a model, compute central velocity dispersion from
        fiducial mass, or vice versa"""

        if self.UseMProj:
            self.ModelGetS0FromMProj()
        else:
            # modelmfiducial takes nearly half a second to run - cut
            # it out by default
            if not self.DoCalcWhileFitting:
                return
            self.ModelMFiducial = self.PointEnclosedMass2DFull(self.FiducialRProj, FindingSigma = False)[0]

    def GenerateDataFromModel(self, InRadii = N.array([])):

        """When testing, take test model components and produce an
        "observed" velocity profile with errors"""

        # radii to use
        if N.any(InRadii):
            R = InRadii
        else:
            nsamples = 10
            R = N.arange(nsamples)/N.float(nsamples) * (5.0)

        # ensure that the evaluating radii are recorded
        self.TestObsRadii = N.copy(R)

        # pass the parameters to the model
        self.GeneratingToModel()

        # ensure the mass or vdisp is populated
        self.ModelDispAndMProj()

        # produce dataset
        self.R2D = R
        self.r3d = R
#        print "GenerateDataFromModel:", self.ModelR0, self.ModelNr, self.ModelNt, self.ModelVDisp0
        self.ArrayDispOnSky()
        
        # datasets
        self.DataVDispRad = N.copy(self.EvalVDispRad)
        self.DataVDispTan = N.copy(self.EvalVDispTan)

        # convenience variables
        nrad = N.size(self.DataVDispRad)
        ntan = N.size(self.DataVDispTan)

        # errors
        self.DataEDispRad = N.repeat(self.TestModelErrorRad, nrad)
        self.DataEDispTan = N.repeat(self.TestModelErrorTan, ntan)

        # perturb by error
        PerturbRad = N.random.normal(size=nrad)
        PerturbTan = N.random.normal(size=ntan)
        self.DataVDispRad = self.DataVDispRad + self.DataEDispRad * PerturbRad
        self.DataVDispTan = self.DataVDispTan + self.DataEDispTan * PerturbTan

        print "GenerateDataFromModel DBG:", N.std(PerturbRad) * N.sqrt(nrad / (1.0 + nrad)), \
            N.min(PerturbRad), N.max(PerturbRad)
        print "GenerateDataFromModel DBG:", N.std(PerturbTan) * N.sqrt(ntan / (1.0 * ntan)), \
            N.min(PerturbTan), N.max(PerturbTan)


    def TryIntegratingDensity(self):

        """Try integrating the density function with arguments passed
        from the class"""

        OuterLimit = 1.0

        # THIS SYNTAX APPEARS TO WORK!!
        Integ1D = quad(lambda r: self.ArrayVolumeDensityFull(r), \
                           0.0, OuterLimit)

        print "DBG - outer limit: %f " % (OuterLimit)

        return Integ1D[0]


    # python double-integral test
    def TestInteg2(self, n = 1.0):

        """Test double integration within a class"""

        Integ1 = dblquad( lambda t, x: N.exp(-t * x) / t**n, \
                              0, Inf, \
                              lambda dum: 1, lambda dum: Inf)
                              
        Integ2 = dblquad( lambda y, x: 1.0 / (1.0 - y * x), \
                              0, 1, lambda dum: 0, lambda dum: 1)

        print n, Integ1[0], Integ2[0], N.pi**2 / 6.0

    def ComputeChisqVel(self):

        """Given a set of model dispersions, estimate the chisq"""

        # check the physicality of the density-model
        self.CheckModelIsPhysical()

        self.EvalChisqRad = self.ComputeChisquared(self.DataVDispRad, \
                                                       self.DataEDispRad, \
                                                       self.EvalVDispRad)

        self.EvalChisqTan = self.ComputeChisquared(self.DataVDispTan, \
                                                       self.DataEDispTan, \
                                                       self.EvalVDispTan)

        if self.CutUnphysical:
            if not self.ModelIsPhysical:
                self.EvalChisqRad = self.BadChisq
                self.EvalChisqTan = self.BadChisq

        self.EvalChisqBoth = self.EvalChisqRad + self.EvalChisqTan

    def ComputeChisquared(self, InData = N.array([]), \
                              InError = N.array([]), \
                              InEval = N.array([]), \
                              DoStrict=False):
        
        """Given data, error, and evaluations of the model, return
        chisquared"""
        
        # initialise returnval
        chisq = None

        try:
            y = 1.0 * InData
            e = 1.0 * InError
            f = 1.0 * InEval

            # DoStrict will usually be set: does NOT insert dummy
            # errors for equal-weighting if unset
            if not DoStrict:
                if not N.any(InError):
                    e = N.ones(N.size(y))

            # ensure the dimensions agree
            dum = y + e + f
        except:
            return chisq

        chisq = N.sum( ( (y - f)/e)**2)
        return chisq

    def CheckModelBuildRadii(self):

        """Build radius-vector for model physicality-checking"""
        
        self.r3PhysicalRad = N.linspace(self.r3PhysicalMin, \
                                            self.r3PhysicalMax, \
                                            num = self.r3PhysicalNum, \
                                            endpoint = True)

    def CheckModelIsPhysical(self, DEBUG = False):

        """Check some standard things for the validity of the model"""

        # rho might not be set yet... use current model parameters to
        # set it if so
        
        if not N.any(self.ModelVolumeDensity):
            if not N.any(self.r3PhysicalRad):
                self.CheckModelBuildRadii()
            self.ModelVolumeDensity = self.ArrayVolumeDensityFull(\
                self.r3PhysicalRad)

        if DEBUG:
            print self.ModelVolumeDensity[0:5], self.ModelVolumeDensity[-1]

        self.ModelIsPhysical = True
        self.ModelIsUnphysical = 0
        if not self.DensityAllPositive():
            self.ModelIsPhysical = False
            self.ModelIsUnphysical = self.ModelIsUnphysical + 1

        # evaluate the central density
        if not N.any(self.ModelDensity0):
            self.PointCentralVolumeDensityFull()

        if self.ModelDensity0 < 0:
            self.ModelIsPhysical = False
            self.ModelIsUnphysical = self.ModelIsUnphysical + 2

    def DensityAllPositive(self, inRho = N.array([])):

        """Check if the volume density is everywhere >= 0"""

        if N.any(inRho):
            rho = N.copy(inRho)
        else:
            rho = N.copy(self.ModelVolumeDensity)
    
        try:
            dum = 1.0 * rho
        except:
            return

        if not N.any(rho):
            return False

        iTooLow = N.where(rho < 0)[0]
        if N.size(iTooLow) > 0:
            return False

        # if we have dropped this far, then all values must be >0
        return True

    def ModelGetS0FromMProj(self, InMProj = None):

        """Given a mass within a fiducial radius and a fiducial
        radius, estimate the central velocity dispersion"""

        if N.any(InMProj):
            MProjMatch = InMProj
        else:
            MProjMatch = self.ModelMFiducial

        if not N.any(MProjMatch):
            return

        # evaluate enclosed mass at the fiducial radius with s0 == 1
        MProj = self.PointEnclosedMass2DFull(self.FiducialRProj, FindingSigma = True)
        
        sigmasq = MProjMatch / MProj[0]
        self.ModelVDisp0 = N.sqrt(sigmasq)

    def TrialBuildMProjValues(self):

        """Given min, max, step values for projected mass, construct
        array of trial projected-mass values."""

        try:
            MMin = self.TrialMProjMin
            MMax = self.TrialMProjMax
            MStep = self.TrialMProjStep
        except:
            return

        MassVec = N.linspace(MMin, MMax + MStep, MStep)
        self.TrialMProjValues = N.copy(MassVec)

    def SubsetTestMinimize(self, Key1='R0', Key2='Final', i0 = 0, i1 = 3):

        """Test the subset routines"""

        self.SubsetSetupKeys(Key1, Key2)
        self.SubsetSetupGrids()
        self.SubsetI0 = i0
        self.SubsetI1 = i1
        self.SubsetFindBest2D()
        self.SubsetPassBest()
        self.SubsetPlotMinimizer()

    def SubsetSetupKeys(self, Key1 = 'R0', Key2='Final'):

        """When optimizing a subset of parameters, initialise the
        keywords to hold fixed"""
        
        self.SubsetKeysVary = self.TrialVects.keys()
        self.SubsetKeysFixed = []
        for Key in [Key1, Key2]:

            # Key actually valid?
            if not Key in self.SubsetKeysVary:
                return

            self.SubsetKeysFixed.append(Key)
            self.SubsetKeysVary.remove(Key)

    def SubsetSetupGrids(self):

        """Given the keys to hold fixed and vary, set up the results
        grids"""

        if len(self.SubsetKeysFixed) <> 2:
            return

        if len(self.SubsetKeysVary) < 1:
            return

        if not N.any(self.TrialGridShape):
            return

        # which index is which in the shape array?
        DShapeIndices = {'R0':0,'Nr':1,'Nt':2,'Final':3}

        size0 = self.TrialGridShape[DShapeIndices[self.SubsetKeysFixed[0]]]
        size1 = self.TrialGridShape[DShapeIndices[self.SubsetKeysFixed[1]]]
        
        SubsetGridShape = (size0, size1)
        self.SubsetVDisp0   = N.zeros(SubsetGridShape)
        self.SubsetMFiduc2D = N.zeros(SubsetGridShape)
        self.SubsetChisqRad = N.zeros(SubsetGridShape)
        self.SubsetChisqTan = N.zeros(SubsetGridShape)
        self.SubsetRho0     = N.zeros(SubsetGridShape)
        self.SubsetMFiduc3d = N.zeros(SubsetGridShape)

        # Fitting-parameters tried
        self.SubsetFitR0 = N.zeros(SubsetGridShape)
        self.SubsetFitNr = N.zeros(SubsetGridShape)
        self.SubsetFitNt = N.zeros(SubsetGridShape)
        self.SubsetFitFinal = N.zeros(SubsetGridShape)
        self.SubsetChisqOld = N.zeros(SubsetGridShape)

        self.SubsetParams  = {'R0':self.SubsetFitR0, \
                                  'Nr':self.SubsetFitNr, \
                                  'Nt':self.SubsetFitNt, \
                                  'Final':self.SubsetFitFinal}

        # copy across the input parameters for the fixed params
        KeyFixed0 = self.SubsetKeysFixed[0]
        KeyFixed1 = self.SubsetKeysFixed[1]

        Fixed0, Fixed1 = N.meshgrid(self.TrialVects[KeyFixed0], self.TrialVects[KeyFixed1])
        Fixed0 = N.transpose(Fixed0)
        Fixed1 = N.transpose(Fixed1)
        self.SubsetParams[KeyFixed0] = Fixed0
        self.SubsetParams[KeyFixed1] = Fixed1
                                              
    def SubsetLoopMinimize(self):

        """Having set up subset of values to hold constant, loop
        through these values and find the bestfit varying parameters
        in each case"""

        ShapeTrials = N.shape(self.SubsetFitR0)
        NVec0 = ShapeTrials[0]
        NVec1 = ShapeTrials[1]
        
        NTotal = NVec0 * NVec1

        print "SubsetLoopMinimize DBG: %i" % (NTotal)

        NDone = 0
        for i0 in range(NVec0):
            for i1 in range(NVec1):
                self.SubsetI0 = i0
                self.SubsetI1 = i1

                try:
                    self.SubsetFindBest2D()
                    if not self.SubspaceHasAnyGood:
                        continue

                    self.SubsetPassBest()
                    self.SubsetPlotMinimizer()
                except(KeyboardInterrupt):
                    print "SubsetLoopMinimize INFO - keyboard interrupt %i of %i ; returning" \
                        % (NDone, NTotal)
                    return

                NDone = NDone + 1

    def SubsetCalcMinimized(self):

        """Once the aux parameters have been minimized for each of the
        "fixed" parameters in the subset, compute the physical
        quantities of interest."""

        # pass to subspace calculation dictionary
        self.SubsetMinToSubspace()
        self.Subspace2DDoCalc()
        
        xflat = N.ravel(self.SubspaceDCalc[self.SubspaceKeysFixed[0]])
        yflat = N.ravel(self.SubspaceDCalc[self.SubspaceKeysFixed[1]])
        zflat = N.ravel(self.SubspaceZGrid)
        pflat = N.ravel(self.SubspaceDCalc[self.SubspaceKeysVary[0]])
        qflat = N.ravel(self.SubspaceDCalc[self.SubspaceKeysVary[1]])

        # construct vectors
        LVecs = [xflat, yflat, zflat, pflat, qflat]
        LUnits= [(self.SubspaceKeysFixed[0],N.float), \
                     (self.SubspaceKeysFixed[1], N.float), \
                     ('Chisq', N.float), \
                     (self.SubspaceKeysVary[0], N.float), \
                     (self.SubspaceKeysVary[1], N.float)]

        for ThisKey in ['Rho0', 'MProj', 'M3d']:
            if ThisKey in self.SubspaceDCalc.keys():
                rflat = N.ravel(self.SubspaceDCalc[ThisKey])
                LVecs.append(rflat)
                LUnits.append((ThisKey, N.float))

        RecOut = N.rec.fromarrays(LVecs, LUnits)
        pyfits.writeto('TryCalcRho.fits',RecOut,clobber=True)

        self.SubspaceWriteUnbinned('TryCalcRho2D.fits')

        # Rho0 has quite a high dynamic range
        self.Subspace2DInterpCalc('Rho0', 1000, CutClosest = False)
        self.Subspace2DInterpCalc('M3d' , 500, CutClosest = False)
        self.Subspace2DInterpCalc('MProj', 500)

        self.Subspace2DInterpWrite('InterpCalcMin.fits')

        # collapse along the param axis for each interpolated quantity
        self.SubspaceCollapseInterp('M3d')
        self.SubspaceCollapseInterp('Rho0')
        self.SubspaceCollapseInterp('MProj')

        self.SubspaceWriteCollapsed()

    def SubspaceWriteCollapsed(self, FitsCollapse = 'TryCollapseValues.fits'):

        """Write collapsed-values to disk for plotting"""

        if len(self.CollapsedDInterpValue.keys()) < 1:
            return


        # initialise the output file
        PriHDU = pyfits.PrimaryHDU()
        hdulist = pyfits.HDUList([PriHDU])

        for ThisKey in self.CollapsedDInterpValue.keys():
            
            TheseValues = N.copy(self.CollapsedDInterpValue[ThisKey])
            TheseChisqs = N.copy(self.CollapsedDInterpChisq[ThisKey])

            # generate table
            c1 = pyfits.Column(name = ThisKey, array = TheseValues, format='D')
            c2 = pyfits.Column(name = 'Chisq', array = TheseChisqs, format='D')
            table_hdu = pyfits.new_table([c1, c2])
            table_hdu.name = ThisKey

            hdulist.append(table_hdu)

        hdulist.writeto(FitsCollapse, clobber=True)
        hdulist.close()

    def SubspaceWriteUnbinned(self, FitsOut = 'TryCalcRho2D.fits'):

        """Write the calculated values to 2D fits file for rapid
        testing of interpolation"""
        
        print "SubspaceWriteUnbinned INFO - writing 2D calculated values"
        
        pyfits.writeto(FitsOut, self.SubspaceDCalc[self.SubspaceKeysFixed[0]], clobber=True)
        pyfits.append(FitsOut, self.SubspaceZGrid)
        for ThisKey in ['Rho0', 'MProj', 'M3d']:
            if ThisKey in self.SubspaceDCalc.keys():
                pyfits.append(FitsOut, self.SubspaceDCalc[ThisKey])

    def SubspaceReadUnbinned(self, FitsRead = 'TryCalcRho2D.fits'):

        """Read in the subspace calculated values"""

        # only reads in: preserved quantity, chisq, calculated values

        if not os.access(FitsRead, os.R_OK):
            return

        hdulist=pyfits.open(FitsRead)

        # read in the keynames first
        LKeys = hdulist[0].data

        self.SubspaceKeysFixed = ['R0']

        self.SubspaceDCalc = {}
        self.SubspaceDCalc['R0']    = hdulist[0].data
        self.SubspaceZGrid          = hdulist[1].data
        self.SubspaceDCalc['Rho0']  = hdulist[2].data
        if len(hdulist) > 3:
            self.SubspaceDCalc['M3d'] = hdulist[3].data
        if len(hdulist) > 4:
            self.SubspaceDCalc['MProj'] = hdulist[4].data

        hdulist.close()


    def SubspaceCollapseInterp(self, ThisKey ='Rho0'):

        """Given a set of interpolated calculated parameters"""

        if not ThisKey in self.SubspaceDInterpX.keys():
            return

        # interpolated parameters
        GridY = self.SubspaceDInterpY[ThisKey]
        GridX = self.SubspaceDInterpX[ThisKey]
        GridZ = self.SubspaceDInterpZ[ThisKey]
        GridU = self.SubspaceDInterpU[ThisKey]

#        print N.shape(GridX)

        # initialise results vectors:
        YVecOut = GridY[0,:]
        ZVecOut = N.copy(YVecOut) * 0.0
        ZVecRaw = N.copy(YVecOut) * 0.0
        
        # only a subset of objects will be used. loop through the
        # interpolated values.
        for iY in range(N.size(YVecOut)):
            UThis = GridU[:,iY]
            ZThis = GridZ[:,iY]            
            XThis = GridX[:,iY]

            Goods = N.where(UThis > 0)[0]
 #           print N.sum(Goods)

            if N.size(Goods) < 1:
                continue

            # find the raw minimum
            iMin = N.argmin(XThis[Goods])
            ZVecRaw[iY] = ZThis[Goods[iMin]]

#            if N.abs(iY / N.float(N.size(YVecOut)) - 0.5) < 0.05:  
#                try:
#                    P.close()
#                except:
#                    dum = 1
            
#                P.figure(figsize=(6,6))
#                P.plot(XThis[Goods], ZThis[Goods],'k.')
#                print XThis[Goods[iMin]], ZVecRaw[iY]
#                return
 

        # pass the result up to the class
        self.CollapsedDInterpValue[ThisKey] = N.copy(YVecOut)
        self.CollapsedDInterpChisq[ThisKey] = N.copy(ZVecRaw)

    def SubsetMinToSubspace(self):

        """Once the subset "fixed" values have been optimized for the
        "vary" parameters, pass the results across to the subspace in
        order to calculate parameters of interest."""

        # self.SubsetParams[ThisKey] contains the results of the
        # minimization. Pass to subspace dictionary for calculation.

        self.SubspaceDCalc = {}
        
        # params that were held fixed while the others varied
        for iKey in range(len(self.SubsetKeysFixed)):
            ThisKey = self.SubsetKeysFixed[iKey] 
            self.SubspaceDCalc[ThisKey] = self.SubsetParams[ThisKey]

        for iKey in range(len(self.SubsetKeysVary)):
            ThisKey = self.SubsetKeysVary[iKey]
            self.SubspaceDCalc[ThisKey] = self.SubsetParams[ThisKey]
            
        # that's the coordinates... now for the chisq values
        # themselves corresponding to the original parameters
        #
        # SubsetChisqOld already matches the shape of the parameters
        # held fixed.
        self.SubspaceZGrid = N.copy(self.SubsetChisqOld)

            
    def SubsetFindBest2D(self):

        """Given an index for objects within the subspace, find the
        best parameter-pair in the subspace that matches a given
        combination of fixed parameters"""

        # set up the subspace to minimize
        #
        # convenience variables
        KeyFixed0 = self.SubsetKeysFixed[0]
        KeyFixed1 = self.SubsetKeysFixed[1]
        i0 = self.SubsetI0
        i1 = self.SubsetI1
        self.Subspace2DInitialise()
        self.Subspace2DGetSubset(KeyFixed0, KeyFixed1, i0, i1)
        self.Subspace2DMakeGrids()
        
        if not self.SubspaceHasAnyGood:
            return

        # find the minimum varying param-values
        xmin, ymin, zmin = self.FindMinimumTwo()  
        self.SubspaceXMin = xmin
        self.SubspaceYMin = ymin
        self.SubspaceZMin = zmin

    def SubsetPassBest(self):

        """Given an xmin, ymin from a trial, pass up to the class"""

        KeyVary0 = self.SubsetKeysVary[0]
        KeyVary1 = self.SubsetKeysVary[1]
        KeyFixed0 = self.SubsetKeysFixed[0]
        KeyFixed1 = self.SubsetKeysFixed[1]
        i0 = self.SubsetI0
        i1 = self.SubsetI1

        self.SubsetParams[KeyVary0][i0,i1] = self.SubspaceXMin
        self.SubsetParams[KeyVary1][i0,i1] = self.SubspaceYMin
        self.SubsetParams[KeyFixed0][i0,i1] = self.TrialVects[KeyFixed0][i0]
        self.SubsetParams[KeyFixed1][i0,i1] = self.TrialVects[KeyFixed1][i1]

        # calculate new chisq using these model parameters
        DThisModel = {}
        DThisModel[KeyVary0]  = self.SubspaceXMin
        DThisModel[KeyVary1]  = self.SubspaceYMin
        DThisModel[KeyFixed0] = self.TrialVects[KeyFixed0][i0]
        DThisModel[KeyFixed1] = self.TrialVects[KeyFixed1][i1]

        # translate this into model parameters to re-compute chisq
        # requires reading in data as well...

        self.SubsetChisqOld[i0,i1] = self.SubspaceZMin

        # finally, determine if the best-fit parameter combination
        # leads to a physical solution
        bob = 3

    def SubsetPlotMinimizer(self):

        """Convenience-function: plot chisq surface for pair of
        params"""

        if not self.SubsetDoPlot:
            return

        Name1 = self.SubspaceKeysFixed[0]
        Inde1 = self.SubspaceIndsFixed[0]
        Val1  = self.TrialVects[Name1][Inde1]
        Name2 = self.SubspaceKeysFixed[1]
        Inde2 = self.SubspaceIndsFixed[1]
        Val2  = self.TrialVects[Name2][Inde2]

        xmin = self.SubspaceXMin
        ymin = self.SubspaceYMin

        # find the minmum chisq too
        zmin = self.SubspaceZMin

        try:
            P.close()
        except:
            dum = 1

        P.figure()
        zgrid = N.copy(self.SubspaceZGrid)
        if self.SubsetPlotLog:
            zgrid = N.log10(zgrid)

        # offset - quadrilaterals are plotted from the lower-left...
        xoffset = 0.5 * (self.SubspaceXGrid[0][1] - self.SubspaceXGrid[0][0])
        yoffset = 0.5 * (self.SubspaceYGrid[1][0] - self.SubspaceYGrid[0][0])
        print xoffset, yoffset

        P.pcolor(self.SubspaceXGrid - xoffset, \
                     self.SubspaceYGrid - yoffset, \
                     zgrid)
        P.colorbar()
        P.plot(xmin,ymin,'wx', markersize=5)
        P.plot([N.min(self.SubspaceXGrid), N.max(self.SubspaceXGrid)], \
                   [ymin, ymin],'y--')
        P.plot([xmin, xmin], \
                   [N.min(self.SubspaceYGrid), N.max(self.SubspaceYGrid)],'y--')
        P.xlabel(self.SubspaceKeysVary[0])
        P.ylabel(self.SubspaceKeysVary[1])

        # find the minimum of the chisq array itself... is it close?
        iGMin = N.argmin(zgrid)        
        GMinX = N.ravel(self.SubspaceXGrid)[iGMin]
        GMinY = N.ravel(self.SubspaceYGrid)[iGMin]
        zminshow = N.min(zgrid)
        if self.SubsetPlotLog:
            zminshow = 10.0**zminshow
        
        print "%.2f %.2f %.2f" % (GMinX, GMinY, zminshow)
        print "%.2f %.2f %.2f" % (xmin, ymin, zmin)

        P.title("%s = %.3f, %s = %.3f, zmin = %.1f" % (Name1, Val1, Name2, Val2, N.log10(zmin)))
        P.savefig('ShowTrialExample.png')
        
    def Subspace2DInitialise(self):

        """Initialise the subspace"""

        self.SubspaceIndsFixed = N.array([])
        self.SubspaceValsFixed = N.array([])
        self.SubspaceXVec = N.array([])
        self.SubspaceYVec = N.array([])
        self.SubspaceXGrid = N.array([])
        self.SubspaceYGrid = N.array([])
        self.SubspaceZOrig = N.array([])
        self.SubspaceZGrid = N.array([])
        self.SubspaceUGrid = N.array([])
        self.SubspaceXMin = None
        self.SubspaceYMin = None
        self.SubspaceZMin = None
        self.SubspaceHasAnyGood = True
        self.SubspaceDCalc = {}

    def Subspace2DMakeGrids(self):

        """Once 2D subspace chisq and variable names have been
        selected, convert them into grid suitable for
        minimum-finding"""

        self.SubspaceXGrid, self.SubspaceYGrid = N.meshgrid(self.SubspaceXVec, self.SubspaceYVec)

        # transpose the chisq array so that it agrees with the
        # meshgrid; initialise the usegrid
        # 
        # keep separate variable to avoid flip-flopping
        self.SubspaceZGrid = N.transpose(self.SubspaceZOrig)
        #self.SubspaceUGrid = N.zeros(N.shape(self.SubspaceZGrid))

        # bring in the physical grid directly
        self.SubspaceUGrid = N.transpose(N.copy(self.SubspaceUOrig))

        if not self.CutUnphysical:
            self.SubspaceUGrid = N.copy(self.SubspaceZGrid) * 0 + 1
        else:
            prebads = N.where(self.SubspaceUGrid < 1)
            self.SubspaceZGrid[prebads] = N.max(self.SubspaceZGrid)

            #print "here - %i %f" % (N.size(self.SubspaceUGrid), N.sum(self.SubspaceUGrid))

        # this may be the problem
#        self.SubspaceZGrid = N.copy(self.SubspaceZOrig)
#        self.SubspaceUGrid = N.copy(self.SubspaceUOrig)
        # do some quality selepolyfitction
        goodvals = N.where(self.SubspaceZGrid < self.BadChisq)
        badsvals = N.where(self.SubspaceZGrid >= self.BadChisq)

        if N.size(goodvals) < 1:
            self.SubspaceHasAnyGood = False
            return
        else:
            dum = 1 
##            self.SubspaceUGrid[goodvals[0],goodvals[1]] = 1
#            self.SubspaceUGrid[goodvals] = 1

        if N.size(badsvals) < 1:
            return

        self.SubspaceZGrid[badsvals] = N.max(self.SubspaceZGrid[goodvals])
        
    def Subspace2DCalcDict(self):

        """Given a 2D subspace and fixed parameters, build grids of
        values from which to calculate outcome-parameters of
        interest"""

        # set up grid parameters as a dictionary, will make calling by
        # keyword name less nuts...
        
        # initialise the calculation dictionary
        self.SubspaceDCalc = {}

        # populate the entries that vary over the grid
        self.SubspaceDCalc[self.SubspaceKeysVary[0]] = self.SubspaceXGrid
        self.SubspaceDCalc[self.SubspaceKeysVary[1]] = self.SubspaceYGrid

        # retrieve the fixed-values for this subspace
        for iKey in range(len(self.SubspaceKeysFixed)):
            self.SubspaceDCalc[self.SubspaceKeysFixed[iKey]] = \
                self.SubspaceXGrid * 0.0 + self.SubspaceValsFixed[iKey]

#        self.SubspaceDCalc[self.SubspaceKeysFixed[0]] = \
#            self.SubspaceXGrid * 0.0 + self.SubspaceValsFixed[1]

    def SubspaceSetModelPars(self):

        """Given a calculation-dictionary, set the model parameters"""

        self.ModelR0 = self.SubspaceDCalc['R0']
        self.ModelNr = self.SubspaceDCalc['Nr']
        self.ModelNt = self.SubspaceDCalc['Nt']
        try:
            self.ModelVDisp0 = self.SubspaceDCalc['s0']
        except:
            self.ModelVDisp0 = self.SubspaceDCalc['Final']

    def Subspace2DDoCalc(self):

        """Armed with the calculation-dictionary, perform
        calculation"""

        self.SubspaceSetModelPars()

        # calculate rho and m3d
        self.Subspace2DCalcRho0()
        self.Subspace2DCalcM3d()

        if not self.CalcFast:
            print "Starting projected-mass calculations"
            self.Subspace2DCalcM2D()
            self.SubspaceSetModelPars()   # leave things as they were before

    def Subspace2DCalcM3d(self):

        """Calculate mass enclosed within 3d fiducial radius for each
        set of input parameters"""

        Mass3d = self.ArrayEnclosedMass3DFull(self.FiducialR3D)
        self.SubspaceDCalc['M3d'] = Mass3d

    def Subspace2DCalcRho0(self):

        """Calculate central density over subspace from input
        parameters"""

        print N.shape(self.ModelR0)

        # compute the central volume density
        self.PointCentralVolumeDensityFull()
        self.SubspaceDCalc['Rho0'] = self.ModelDensity0

    def Subspace2DCalcM2D(self):

        """Calculate enclosed projected mass over subspace from input
        parameters"""

        # Initialise output results
        ShapeModel = N.shape(self.SubspaceDCalc['R0'])
        ModelMassProj = N.zeros(ShapeModel)
    
        # what did we call the final parameter?
        SigKey = 'Final'
        try:
            dum = self.SubspaceDCalc[SigKey]
        except:
            SigKey = 's0'

        NumDoneSoFar = 0
        NumToDo = N.prod(ShapeModel)
        TimeStarted = time.time()
        for iPar1 in range(ShapeModel[0]):
            for iPar2 in range(ShapeModel[1]):

                try:
                    # set up model parameters
                    self.ModelR0 = self.SubspaceDCalc['R0'][iPar1][iPar2]
                    self.ModelNr = self.SubspaceDCalc['Nr'][iPar1][iPar2]
                    self.ModelNt = self.SubspaceDCalc['Nt'][iPar1][iPar2]
                    self.ModelVDisp0 = self.SubspaceDCalc[SigKey][iPar1][iPar2]

                    # estimate mass enclosed at fiducial radius
                    ThisMass = self.PointEnclosedMass2DFull(self.FiducialRProj, \
                                                                FindingSigma = False)

                    if NumDoneSoFar % 10 == 1:
                        TimeElapsed = time.time() - TimeStarted
                        MeanRate = TimeElapsed / N.float(NumDoneSoFar)
                        StringEnd = self.StatusTimeString(NumDoneSoFar, NumToDo, TimeStarted)
                        StringShow = "DBG: %i of %i - %.2f seconds elapsed. Per trial %.2f; finish %s" \
                            % (NumDoneSoFar, NumToDo, TimeElapsed, MeanRate, StringEnd)
                        sys.stdout.write("\r %s" % (StringShow))
                        sys.stdout.flush()

                    ModelMassProj[iPar1][iPar2] = ThisMass[0]
                except(KeyboardInterrupt):
                    print "Interrupt detected. returning."
                    return

                NumDoneSoFar = NumDoneSoFar + 1

        # pass the resulting mass estimates back up to the dictionary
        print "Finished estimating M2D"
        self.SubspaceDCalc['MProj'] = N.copy(ModelMassProj)

    def Subspace2DInterpCalc(self, CalcChoice = 'Rho0', NumInterp = None, \
                                 CutClosest = False, InSmooth = 2, UseClusters = True):

        """Chisq array and two arrays, one of which is
        gridded. Interpolate the non-gridded one to place it on a
        regular grid"""

        print self.SubspaceDCalc.keys()

        if not CalcChoice in self.SubspaceDCalc.keys():
            return

        # calc array under focus
        self.SubspaceCalcArray = self.SubspaceDCalc[CalcChoice]

        # self.SubspaceXGrid is the array carried through in the
        # calculation (s0 will be replaced with the calculated value)

        SubspaceShape = N.shape(self.SubspaceCalcArray)
        NPreserved = SubspaceShape[0]

        # Vector of calculated values at which to interpolate chisq
        # surface
        MaxCalc = N.max(self.SubspaceCalcArray)
        MinCalc = N.min(self.SubspaceCalcArray)
        if N.any(NumInterp):
            NumCalc = N.copy(NumInterp)
        else:
            NumCalc = SubspaceShape[0]
        VecCalc = N.linspace(MinCalc, MaxCalc, NumCalc, endpoint = True)

        # construct output calc array (and grid array and chisq array)
        # evaluated at the gridpoints
        SubspaceInterpXGrid = N.zeros((NPreserved, NumCalc))
        SubspaceInterpCGrid = N.copy(SubspaceInterpXGrid)
        SubspaceInterpZGrid = N.copy(SubspaceInterpXGrid)
        SubspaceInterpUGrid = N.ones(N.shape(SubspaceInterpXGrid))

        try:
            P.clf()
        except:
            dum = 1

        for iPreserved in range(NPreserved):

            ThisCalcVec = self.SubspaceCalcArray[iPreserved,:]
            ThisChisVec = self.SubspaceZGrid[iPreserved,:]

            # NB spline needs this to be sorted in increasing order of
            # ThisCalcVec
            VecSort = N.argsort(ThisCalcVec, kind='mergesort')
            UseArr = N.ones(N.size(VecSort))

            # distances from previous-lowest point. Sometimes the
            # spline-fit flails if two points are too close together.
            VDiffs = ThisCalcVec[VecSort] - N.roll(ThisCalcVec[VecSort],1)
            VDiffs[0] = 1e9
            VDiffs[1] = 1e9
            GoodDiffs = N.where(VDiffs < 1.0e8)[0]
            
            # find close-pairs of objects
#            print N.mean(VDiffs[GoodDiffs]), N.median(VDiffs[GoodDiffs])
            MinLag = N.median(VDiffs[GoodDiffs])*0.6
            ClosePairs = N.where(VDiffs[GoodDiffs] < MinLag)[0]
            if N.size(ClosePairs) > 0:
                ClosePairs = GoodDiffs[ClosePairs]

            # if close pairs have been found, count the number of
            # neighbours to each point that is part of a close pair
            CloseCounts = N.zeros(N.size(ClosePairs))
            for iClose in range(N.size(ClosePairs)):

                # count near-neighbours both above and below
                VsThis = VecSort[ClosePairs[iClose]]
                DiffsThis = N.abs(ThisCalcVec[VecSort] - ThisCalcVec[VsThis])
                NearThis = N.where( (DiffsThis > 0) & (DiffsThis < MinLag))[0]
                CloseCounts[iClose] = N.size(NearThis)
                
            # vector of clusters
            ClustersX = N.array([])
            ClustersY = N.array([])
            Use4Cluster = N.ones(N.size(VDiffs))

            # go down the list of clusters in descending size order
            DescendingClusterSize = N.argsort(CloseCounts)[::-1]

            for jClump in range(N.size(ClosePairs)):
                iCluster = DescendingClusterSize[jClump]

                # find points within close distance of this cluster-point
                #
                # include the original, which will be included in the cluster
                VecSortThis = VecSort[ClosePairs[iCluster]]
                if Use4Cluster[ClosePairs[iCluster]] < 1:
                    continue

                XThis = ThisCalcVec[VecSortThis]

                # find points not already accounted for that are close to this point
                CloseToThis = N.where( (N.abs(ThisCalcVec[VecSort] - XThis) < MinLag) \
                                           & (Use4Cluster > 0))[0]
                if N.abs(iPreserved - self.PlotNum) < 0.2:
                    print "DBG: neighbours %.2f %i" % (XThis, N.size(CloseToThis))
                if N.size(CloseToThis) < 2:
                    continue

                # CloseToThis refers to indices in VecSort. So does
                # Use4Cluster.
                #
                # don't want to use the same point in multiple means.
                Use4Cluster[CloseToThis] = 0

                if not UseClusters:
                    continue
                
                # find the mean position in X and Y for the points
                XThisCluster = N.mean(ThisCalcVec[VecSort[CloseToThis]])
                YThisCluster = N.mean(ThisChisVec[VecSort[CloseToThis]])

                ClustersX = N.hstack((ClustersX, XThisCluster))
                ClustersY = N.hstack((ClustersY, YThisCluster))

                # knock out the original points from the fit here...
                UseArr[CloseToThis] = 0

            if CutClosest:
                ClosestPair = N.argmin(VDiffs)
                UseArr[ClosestPair ]    = 0
                UseArr[ClosestPair -1 ] = 0            

            # construct arrays for interpolation
            FitSort = VecSort[N.where(UseArr > 0)[0]]
            
            XForFit = N.copy(ThisCalcVec[FitSort])
            YForFit = N.copy(ThisChisVec[FitSort])

            # now append any cluster points we have computed
            if N.size(ClustersX) > 1 and N.size(ClustersY) > 1:
                XForFit = N.hstack((XForFit, ClustersX))
                YForFit = N.hstack((YForFit, ClustersY))
                
            # sort the XForFit and YForFit
            SortForFit = N.argsort(XForFit, kind='quicksort')    
            XForFit = XForFit[SortForFit]
            YForFit = YForFit[SortForFit]

            # perform the interpolation. Our friend the spline again
#            print "Interp DBG: ", iPreserved, N.size(ThisCalcVec), N.size(ThisChisVec)
 #           print ThisCalcVec[FitSort]
#            ThisKnots = interpolate.splrep(ThisCalcVec[FitSort], \
#                                               ThisChisVec[FitSort],s=5.0, per=0)
#            ThisEvals =  interpolate.splev(VecCalc, ThisKnots, der = 0)

            PrintSmooth = False
            if N.abs(iPreserved - self.PlotNum) < 0.2:
                PrintSmooth = True

#                print VDiffs

            # polynomial backbone
            PolyCoeffs = N.polyfit(ThisCalcVec, ThisChisVec,5)
            PolyEvals  = N.polyval(PolyCoeffs, VecCalc)

#            ThisKnots, ThisEvals, UsedInput = \
#                self.SubspaceSplineSmooth(ThisCalcVec[FitSort], \
#                                              ThisChisVec[FitSort], VecCalc, \
#                                              smooth=InSmooth, NumTrials = 0, \
#                                              ShowRes = PrintSmooth)
#            FitSort = FitSort[UsedInput]

            # do with the actual input vector we have constructed
            ThisKnots, ThisEvals, UsedInput = \
                self.SubspaceSplineSmooth(XForFit, YForFit, VecCalc, \
                                              smooth=InSmooth, NumTrials = 0, \
                                              ShowRes = PrintSmooth)

            # evaluate the splinefit once more
           # ThisEvals = interpolate.splev(VecCalc, ThisKnots)

            # try barycentric
#            ThisEvals = interpolate.barycentric_interpolate(ThisCalcVec[FitSort], \
#                                                                ThisChisVec[FitSort], \
#                                                                VecCalc)

            # try polyfit
#            coeffs = N.polyfit(ThisCalcVec[VecSort], \
#                                   ThisChisVec[VecSort], 7)
#            ThisEvals = N.polyval(coeffs,VecCalc)
#            ThisResid = N.polyval(coeffs, ThisCalcVec) - ThisChisVec

            # "good" points 
            UseFine = N.ones(N.size(VecCalc))

            # only apply this to regions WITHIN which there is data
            # coverage
            BadLo = N.where( VecCalc < N.min(ThisCalcVec))[0]
            iAtMin = N.argmin(ThisCalcVec)
            iAtMax = N.argmax(ThisCalcVec)
            ChiAtMin = ThisChisVec[iAtMin]
            ChiAtMax = ThisChisVec[iAtMax]
            if N.size(BadLo) > 1:
                ThisEvals[BadLo] = ChiAtMin
                UseFine[BadLo] = 0

#            # is the spline still flailing?
#            FlailHi = N.where(ThisEvals > N.max(ThisChisVec)*1.2)[0]
#            FlailLo = N.where(ThisEvals < N.min(ThisChisVec)*0.8)[0]
#            if N.size(FlailLo) > 0:
#                UseFine[FlailLo] = 0
#                ThisEvals[FlailLo] = ChiAtMin

#            if N.size(FlailHi) > 0:
#                UseFine[FlailHi] = 0
#                ThisEvals[FlailHi] = ChiAtMax

            BadHi = N.where( VecCalc > N.max(ThisCalcVec))[0]
            if N.size(BadHi) > 1:
                ThisEvals[BadHi] = ChiAtMax
                UseFine[BadHi] = 0

            GoodEvals = N.where(UseFine > 0)[0]

            CheckResids = False
            ResidThresh = 0.1
            if CheckResids:
                fResids = ( PolyEvals[GoodEvals] - ThisEvals[GoodEvals] ) \
                    / PolyEvals[GoodEvals]
                Flailed = N.where(N.abs(fResids) > ResidThresh)[0]
                if N.size(Flailed) > 0:
                    Flailed = GoodEvals[Flailed]
                    ThisEvals[Flailed] = PolyEvals[Flailed]

            # pass to master array
            ThisXValue = self.SubspaceDCalc[self.SubspaceKeysFixed[0]][iPreserved,0]
            SubspaceInterpXGrid[iPreserved,:] = ThisXValue
            SubspaceInterpCGrid[iPreserved,:] = N.copy(VecCalc)
            SubspaceInterpZGrid[iPreserved,:] = N.copy(ThisEvals)
            SubspaceInterpUGrid[iPreserved,:] = N.copy(UseFine)

            # pick an example where the calc vectors do not fully span
            # the interpolant space
            if N.abs(iPreserved - self.PlotNum) < 0.2:

                try:
                    P.subplot(311)
#                    P.plot(self.SubspaceDCalc[self.SubspaceKeysFixed[0]], \
#                               self.SubspaceDCalc[CalcChoice],'k.')
                    P.plot(self.SubspaceDCalc[self.SubspaceKeysFixed[0]][iPreserved,:], \
                               self.SubspaceDCalc[CalcChoice][iPreserved,:],'rs')
                    P.subplot(312)
                    P.plot(ThisCalcVec[VecSort], ThisChisVec[VecSort],'ko')
#                    P.plot(ThisCalcVec[VecSort], ThisChisVec[VecSort],'r-')
                    
                    P.plot(XForFit, YForFit+5,'bx')                    

                    # any close points?
                    if N.size(ClosePairs) > 0:
                        print N.mean(VDiffs[GoodDiffs])
                        P.plot(ThisCalcVec[VecSort[ClosePairs]], \
                                   ThisChisVec[VecSort[ClosePairs]], 'rs')

                    if N.size(ClustersX) > 0:
                        P.plot(ClustersX, ClustersY, 'b^')

                        print MinLag
                        for i in range(N.size(CloseCounts)):
                            print "DBG2", ThisCalcVec[VecSort[ClosePairs[i]]], \
                                           CloseCounts[i]

                    # good points?
                    GoodEvals = N.where(UseFine > 0)[0]

#                    P.plot(VecCalc[GoodEvals], ThisEvals[GoodEvals],'g.')
                    P.plot(VecCalc[GoodEvals], ThisEvals[GoodEvals],'g-')

                    P.plot(VecCalc[GoodEvals], PolyEvals[GoodEvals], 'r-')
#                    P.plot(VecCalc[GoodEvals], PolyEvals[GoodEvals], 'r.')

                    P.subplot(313)

                    # try a histogram
                    P.cla()
                    g = N.where(VDiffs < 1e8)[0]
                    print N.mean(VDiffs[g])
                    freq, edges = N.histogram(VDiffs[g], bins = 100)
                    print VDiffs[g]
                    midpts = edges + 0.5 * (edges - N.roll(edges,1))
                    midpts = midpts[1::]
                    P.plot(midpts, freq,'ks')

#                    P.plot(VecCalc[GoodEvals], \
#                               (PolyEvals[GoodEvals] - ThisEvals[GoodEvals]) \
#                               / PolyEvals[GoodEvals], 'k.')

##                    for i in range(N.size(VDiffs)):
##                        j = VecSort[i]
##                        print ThisCalcVec[j], VDiffs[i]

#                    print CutClosest

                    # closest-pair indices
#                    if CutClosest:
#                        i0 = VecSort[ClosestPair]
#                        i1 = VecSort[ClosestPair-1]

#                        P.plot(ThisCalcVec[i0], \
#                                   ThisChisVec[i0], 'rs')
#                        P.plot(ThisCalcVec[i1], \
#                                   ThisChisVec[i1], 'rs')
                    


                    #print ThisCalcVec[VecSort]
                    

                    P.xlabel(CalcChoice)
                    P.ylabel('ChiSquared')
#                    P.title("%.2f" % (self.ModelR0[0,iPreserved][0]))

#                    P.subplot(313)
#                    P.plot(ThisCalcVec, ThisResid,'k.')

                except:
                    dum = 1
                    
#                return

        # transfer this set of interpolants up to the dictionary
        self.SubspaceDInterpX[CalcChoice] = N.copy(SubspaceInterpXGrid)
        self.SubspaceDInterpY[CalcChoice] = N.copy(SubspaceInterpCGrid)
        self.SubspaceDInterpZ[CalcChoice] = N.copy(SubspaceInterpZGrid)
        self.SubspaceDInterpU[CalcChoice] = N.copy(SubspaceInterpUGrid)

    def SubspaceSplineSmooth(self, X, Y, XEval, smooth=0.0, NumTrials = 5, \
                                 ShowRes = False):

        """Spline-smoothing of points"""

        UseInput = N.ones(N.size(X))

        XMin = N.min(X)
        XMax = N.max(X)

        # only evaluate where there are points
        GoodsFine = N.where( (XEval >= XMin) & (XEval <= XMax) )[0]
        XFine = XEval[GoodsFine]

        for iTrial in range(NumTrials+1):
            GoodInput = N.where(UseInput > 0)[0]

            ThisKnots = interpolate.splrep(X[GoodInput], Y[GoodInput], s=smooth,  per = 0 )
            ThisEvals = interpolate.splev(XFine, ThisKnots, der = 0 )
        
            # minmax of spline to look for flailing
            iEvalMin = N.argmin(ThisEvals)
            iEvalMax = N.argmax(ThisEvals)
            iInpuMin = N.argmin(Y)
            iInpuMax = N.argmax(Y)

            # find the nearest input point to the max and min eval
            NearestToMax = N.argmin(N.abs(X[GoodInput] - XFine[iEvalMax]))
            NearestToMin = N.argmin(N.abs(X[GoodInput] - XFine[iEvalMin]))
            NearestToMin = GoodInput[NearestToMin]
            NearestToMax = GoodInput[NearestToMax]

            if ThisEvals[iEvalMax] > N.max(Y)*1.1:
                if ShowRes:
                    print "INFO:", ThisEvals[iEvalMax], N.max(Y)*1.1
                UseInput[NearestToMax] = 0
            
            if ThisEvals[iEvalMin] < N.min(Y)*0.8:
                UseInput[NearestToMin] = 0

            ThisEvals = interpolate.splev(XEval, ThisKnots)

        return ThisKnots, ThisEvals, GoodInput
        
    def Subspace2DInterpRead(self, FitsRead = 'TryInterp2D.fits', ThisKey = 'M3d'):
        
        """Read in the multidimen array for rapid access"""

        if not os.access(FitsRead, os.R_OK):
            print "Subspace2DInterpRead WARN - cannot find infile %s" % (FitsRead)
            return

        # initialise the dictionary of interpolated objects
        self.SubspaceDInterpX = {}
        self.SubspaceDInterpY = {}
        self.SubspaceDInterpZ = {}
        self.SubspaceDInterpU = {}
        
        self.SubspaceDInterpX[ThisKey] = pyfits.getdata(FitsRead, 0)
        self.SubspaceDInterpY[ThisKey] = pyfits.getdata(FitsRead, 1)
        self.SubspaceDInterpZ[ThisKey] = pyfits.getdata(FitsRead, 2)
        self.SubspaceDInterpU[ThisKey] = pyfits.getdata(FitsRead, 3)

    def Subspace2DInterpWrite(self, FitsInterp = 'TryInterp.fits', ThisKey='M3d'):

        """Write interpolated dictionary to fits file for viewing in
        favorite table viewer"""

        if len(self.SubspaceDInterpX.keys()) < 1:
            return

        # write to 2D so we can import while testing the collapse
        # function
        Fits3D = 'TryInterp2D.fits'
        pyfits.writeto(Fits3D, self.SubspaceDInterpX[ThisKey], clobber=True)
        pyfits.append(Fits3D,  self.SubspaceDInterpY[ThisKey])
        pyfits.append(Fits3D,  self.SubspaceDInterpZ[ThisKey])
        pyfits.append(Fits3D,  self.SubspaceDInterpU[ThisKey])

        NumWritten = 0
        for ThisKey in self.SubspaceDInterpX.keys():
            LWrite = [N.ravel(self.SubspaceDInterpX[ThisKey]), \
                          N.ravel(self.SubspaceDInterpY[ThisKey]), \
                          N.ravel(self.SubspaceDInterpZ[ThisKey]), \
                          N.ravel(self.SubspaceDInterpU[ThisKey])]

            LUnits = [(self.SubspaceKeysFixed[0], N.float), \
                          (ThisKey,  N.float), \
                          ('ChiTot', N.float), \
                          ('Use', N.int)]

            RecWrite = N.rec.fromarrays(LWrite, LUnits)
            if NumWritten < 1:
                pyfits.writeto(FitsInterp, RecWrite, clobber=True)
            else:
                pyfits.append(FitsInterp, RecWrite)
            NumWritten = NumWritten + 1
 
    def Subspace2DCalcShow(self):

        """Given a subspace, compute rho0 and show the results"""
        
        print self.SubspaceKeysVary

        if not 'Final' in self.SubspaceKeysVary:
            print "calculation not a parameter of interest"
            return

        # first build the dictionary
        self.Subspace2DCalcDict()
        
        # then perform the calculation
        self.Subspace2DDoCalc()

        # then show it
        if not N.any(self.SubspaceDCalc['Rho0']):
            print "Subspace2DCalcShow WARN - Rho0 not produced"
            return

        # try plotting results
        try:
            P.close()
        except:
            dum = 1

        xmin = self.SubspaceXMin
        ymin = self.SubspaceYMin

        P.subplots_adjust(hspace = 0.5, wspace = 0.5)
        P.subplot(221)
        zgrid = N.copy(self.SubspaceZGrid)
        if self.SubsetPlotLog:
            zgrid = N.log10(zgrid)
        P.pcolor(self.SubspaceXGrid,self.SubspaceYGrid, zgrid)
        P.colorbar()
        P.plot(xmin,ymin,'wx', markersize=5)
        P.plot([N.min(self.SubspaceXGrid), N.max(self.SubspaceXGrid)], \
                   [ymin, ymin],'y--')
        P.plot([xmin, xmin], \
                   [N.min(self.SubspaceYGrid), N.max(self.SubspaceYGrid)],'y--')
        P.xlabel(self.SubspaceKeysVary[0])
        P.ylabel(self.SubspaceKeysVary[1])
        P.title("%s = %.3f, %s = %.3f" % (self.SubspaceKeysFixed[0], 
                                          self.SubspaceValsFixed[0], \
                                              self.SubspaceKeysFixed[1],\
                                              self.SubspaceValsFixed[1]))

        P.subplot(222)
        P.plot(self.SubspaceDCalc[self.SubspaceKeysVary[0]], self.SubspaceDCalc['Rho0'], 'k.')

        # output flattened arrays for further examination
        xflat = N.ravel(self.SubspaceDCalc[self.SubspaceKeysVary[0]])
        yflat = N.ravel(self.SubspaceDCalc[self.SubspaceKeysVary[1]])
        zflat = N.ravel(self.SubspaceZGrid)

        # construct vectors
        LVecs = [xflat, yflat, zflat]
        LUnits= [(self.SubspaceKeysVary[0],N.float), \
                     (self.SubspaceKeysVary[1], N.float), \
                     ('Chisq', N.float)]

        try:
            rflat = N.ravel(self.SubspaceDCalc['Rho0'])
            LVecs.append(rflat)
            LUnits.append(('Rho0',N.float))
        except:
            dum = 1
            
        try:
            mflat = N.ravel(self.SubspaceDCalc['M3d'])
            LVecs.append(mflat)
            LUnits.append(('M3d',N.float))
        except:
            dum = 1 

#        RecOut = N.rec.fromarrays([xflat, yflat, zflat, rflat, mflat], \
#                                      [(self.SubspaceKeysVary[0],N.float), \
#                                           (self.SubspaceKeysVary[1], N.float), \
#                                           ('Chisq', N.float), \
#                                           ('Rho0', N.float)])

        RecOut = N.rec.fromarrays(LVecs, LUnits)

        # write to fits
        pyfits.writeto('TryCalc.fits',RecOut,clobber=True)

    def Subspace2DGetSubset(self, Key1 = None, Key2 = None, Ind1 = None, Ind2 = None):

        """Given keynames and indices to hold fixed, extract subsets
        from grid-arrays containing the params and chisq values"""

        if not N.any(self.GridChisqRad):
            return

        if not Key1 or not Key2:
            return


        VectKeys = self.TrialVects.keys()
        if not Key1 in VectKeys:
            return
        if not Key2 in VectKeys:
            return

        try:
            dum  = 1 * Ind1 * Ind2
        except:
            return
        
        GridShape = self.TrialGridShape

        # wrap the indices and keys into a dictionary:
        DCalled = {Key1:Ind1, Key2:Ind2}

        # I don;'t know how to pass the equivalent of, e.g., [:,0,2,:]
        # to another routine in a general way that will always
        # work. Do this in a canned way for now
        iSel = N.array([-1,-1,-1,-1])
        
        ArrayKeys = ['R0', 'Nr', 'Nt', 'Final']

        self.SubspaceKeysVary  = ['R0', 'Nr', 'Nt', 'Final']        
        self.SubspaceKeysFixed = []
        self.SubspaceIndsFixed = N.array([])
        self.SubspaceValsFixed = N.array([])
        for iKey in range(len(ArrayKeys)): 
            ThisKey = ArrayKeys[iKey]
            if ThisKey in DCalled.keys():
                # which of the trial-vector indices is this?
                IndexTrial = DCalled[ThisKey]

                # fill in the selected index for the trial-grid
                iSel[iKey] = IndexTrial                

                # ensure the subspace information (keyword,
                # trial-index, value) is appropriately updated
                self.SubspaceKeysFixed.append(ThisKey)
                self.SubspaceIndsFixed = N.append(self.SubspaceIndsFixed, IndexTrial)
                self.SubspaceValsFixed = N.append(self.SubspaceValsFixed, \
                                                      self.TrialVects[ThisKey][IndexTrial])
                self.SubspaceKeysVary.remove(ThisKey)

        # construct the subset X and Y vectors in order
        self.SubspaceXVec = self.TrialVects[self.SubspaceKeysVary[0]]
        self.SubspaceYVec = self.TrialVects[self.SubspaceKeysVary[1]]
    
        # total chisq
        ChisqTot = self.GridChisqRad + self.GridChisqTan

        if self.UseSurfDens:
            ChisqTot = ChisqTot + self.GridChisqSurf
        

        # I don't know how to pass the general slicing index [0,:,3,:]
        # across. So do the extractyion here.
        if iSel[0] > -1:
            if iSel[1] > -1:
                self.SubspaceZOrig = ChisqTot[iSel[0], iSel[1], :, : ]
                self.SubspaceUOrig = self.GridPhysical[iSel[0], iSel[1], :, : ]
                return
            if iSel[2] > -1:
                self.SubspaceZOrig = ChisqTot[iSel[0], :, iSel[2], : ]
                self.SubspaceUOrig = self.GridPhysical[iSel[0], :, iSel[2], :]
                return

            self.SubspaceZOrig = ChisqTot[iSel[0], :, :, iSel[3] ]
            self.SubspaceUOrig = self.GridPhysical[iSel[0], :, :, iSel[3]]
            return

        if iSel[1] > -1:
            if iSel[2] > -1:
                self.SubspaceZOrig = ChisqTot[:, iSel[1], iSel[2], : ]
                self.SubspaceUOrig = self.GridPhysical[:, iSel[1], iSel[2], :]
                return
            self.SubspaceZOrig = ChisqTot[:, iSel[1], :, iSel[3] ]
            self.SubspaceUOrig = self.GridPhysical[:, iSel[1], :, iSel[3] ]
            return

        self.SubspaceZOrig = ChisqTot[:, :, iSel[2], iSel[3] ]
        self.SubspaceUOrig = self.GridPhysical[:, :, iSel[2], iSel[3]]
        return

    def FindMinimumTwo(self , InX = N.array([]), \
                           InY = N.array([]), \
                           InZ = N.array([]), \
                           InU = N.array([]), \
                           badval = 1.0e8, \
                           nfine = 100, \
                           DoPlot = False):

        """Find minimum InZ; NB InX, InY, InZ must be gridded but
        values can be missing or invalid"""

        if N.any(InX):
            x = InX
        else:
            x = self.SubspaceXGrid
        
        if N.any(InY):
            y = InY
        else:
            y = self.SubspaceYGrid

        if N.any(InZ):
            z = InZ
        else:
            z = self.SubspaceZGrid

        if N.any(InU):
            u = InU
        else:
            u = self.SubspaceUGrid

        # debug
#        goods = N.where(u > 0)
#        bads = N.where(u < 1)
#        print "MinimizeTwo DBG: %i %i" % (N.size(goods), N.size(bads))

        # debug plot
        if self.PlotMinimizing:
            print "DBG -- ", N.shape(u), N.size(u), N.sum(u), N.any(InU)
            print N.min(u), N.max(u)

            print "plotting"
            P.clf()
            P.subplot(321)
            P.pcolor(x,y,z)
            P.title('Chisq array')
            P.colorbar()
            P.subplot(322)
            P.pcolor(x,y,u)
            P.title('Use-array')
            P.colorbar()
            P.annotate('Minfinding Debug', (0.5,0.95), xycoords='figure fraction', \
                           horizontalalignment = 'center', verticalalignment='middle')

            return

        StartTime = time.time()

        # convenience
        xVec = x[0,:]
        Nx = N.size(xVec)

        # vectors for zmin along y and yvalues at min 
        YMinZs = N.zeros(Nx)
        YAtMin  = N.zeros(Nx)
        YMinUses = N.ones(Nx)  # selection-vector for ymins
        for ix in range(Nx):

            YThis = y[:,ix]
            ZThis = z[:,ix]
            UThis = u[:,ix]
            
            # this syntax was just wrong
#            Goods = N.where(ZThis < badval)[0]
#            if N.any(InU):
#                Goods = N.where(UThis > 0)[0]

            Goods = N.where (UThis > 0)[0]
            if N.size(Goods) < 1:
                YMinUses[ix] = 0
                continue

#            print N.shape(UThis), N.shape(YThis)
            YFine = N.linspace(start = N.min(YThis[Goods]), \
                                   stop = N.max(YThis[Goods]), \
                                   num = nfine, \
                                   endpoint=True)
            tck = interpolate.splrep(YThis[Goods],ZThis[Goods],s=0)
            ZFine = interpolate.splev(YFine,tck,der=0)  # eval spline
            ZDeri = interpolate.splev(YFine,tck,der=1)  # eval deriv
            indMin = N.argmin(ZFine)

            YMinZs[ix] = ZFine[indMin]
            YAtMin[ix]  = YFine[indMin]

            # must actually be a minimum for this to work. Otherwise
            # report the the limit of the paramspace
            DerivDiff = N.sign(ZDeri * N.roll(ZDeri,1))
            DerivDiff[0] = 1

            SignChanges = N.where(DerivDiff < 0)[0]
            if N.size(SignChanges) < 1:
                if ZDeri[0] < 0:
                    iOrigMin = N.argmax(YThis)
                else:
                    iOrigMin = N.argmin(YThis)

                YMinZs[ix] = ZThis[iOrigMin]
                YAtMin[ix] = YThis[iOrigMin]
                YMinUses[ix]  = UThis[iOrigMin] # inherit the use-value
            # find the minimum
            DoDebug = False
            if DoDebug:
                yminraw = N.argmin(ZThis[Goods])
                yminraw = Goods[yminraw]
                print "DBG: X = %.1f -- YFine %.1f %.1f -- YGrid %.1f %.1f" \
                    % ( xVec[ix], YFine[indMin], ZFine[indMin], \
                            YThis[yminraw], ZThis[yminraw]), N.any(u)

            if self.PlotMinimizing:

#                if ix < 1:
#                    try:
#                        P.close()
#                    except:
#                        dum = 1
#                    P.figure()
                    
                if ix < Nx - 1:
                    ThisColor = 'k'
                else:
                    ThisColor = 'r'

                P.subplot(312)
                P.plot(YThis[Goods], ZThis[Goods], marker='o', color=ThisColor, \
                           linestyle='None')
                P.plot(YFine, ZFine,'g--')

                P.plot(YFine[indMin], ZFine[indMin], 'rs')

        # now find the minimum of the minima
        GoodX = N.where(YMinUses > 0)[0]

        if N.size(GoodX) < 3:
            return N.min(x), N.min(y), 10000.0

        xVec4Y = xVec[GoodX]
        YMinZs4Y = YMinZs[GoodX]
        YAtMin4Y = YAtMin[GoodX]
        XFine = N.linspace(N.min(xVec4Y), N.max(xVec4Y), nfine)


        try:
            tckX = interpolate.splrep(xVec4Y,YMinZs4Y,s=0)
            ZX = interpolate.splev(XFine,tckX,der=0)
        except:
            coeffs = N.polyfit(xVec4Y, YMinZs4Y, 2)
            ZX = N.polyval(coeffs, XFine)

        # find the index in XFine that minimizes chisq at ymin
        iXMin = N.argmin(ZX)

        MinX = XFine[iXMin]
        ZAtMin = ZX[iXMin]

#       what happens if the physical region has no minimum...
#        if iXMin >= N.size(ZX) - 1:
#            MinX = N.max(xVec)
#            ZAtMin = 1.0e3

        if DoPlot:
            P.subplot(325)
            P.plot(xVec4Y, YMinZs4Y, 'ko')

            P.plot(XFine,ZX,'g-')

        # approximate the ymin curve
        try:
            tckY = interpolate.splrep(xVec4Y, YAtMin4Y, s=0)
            MinY = interpolate.splev(MinX, tckY)
        except:
            coeffs = N.polyfit(xVec4Y, YAtMin4Y, 2)
            MinY = N.polyval(coeffs, MinX)

        # return the minima: X, Y, Z
        return MinX, MinY, ZAtMin

    def FindMinimumIn2D(self, InX = N.array([]), \
                            InY = N.array([]), \
                            InZ = N.array([]), \
                            badval = 1.0e8, \
                            DoPlot = True, \
                            nfine = 100):

        """Given a set of chisq values at each of a set of parameters
        X,Y, find the parameters that minimize Z"""

        # parse the input
        if not N.any(InX) or not N.any(InY) or not N.any(InZ):
            return

        if N.size(InX) <> N.size(InY):
            return

        if N.size(InZ) <> N.size(InX):
            return

        # select out the objects < badval
        Goods = N.where(InZ < badval)[0]

        if N.size(Goods) < 10:
            return

        x = InX[Goods]
        y = InY[Goods]
        z = InZ[Goods]

        # for now we find the minimum over the entire space
        
        # produce the spline representation
        stime = time.time()
        tck = interpolate.bisplrep(x,y,z, s=0.001)
        print time.time() - stime
        
        # generate grid for evaluating spline
        xmin = N.min(x)
        xmax = N.max(x)
        ymin = N.min(y)
        ymax = N.max(y)
        
        xvec = N.linspace(xmin,xmax,nfine, endpoint=True)
        yvec = N.linspace(ymin,ymax,nfine, endpoint=True)
        xfine, yfine = N.meshgrid(xvec,yvec)

        # evaluate the spline on this grid
        zfine = interpolate.bisplev(xvec, yvec, tck)

        starttime = time.time()
        if not DoPlot:
            return
        
        try:
            P.close()
        except:
            dum = 1

        P.figure(figsize=(8,5))
        P.subplot(121)
        P.pcolor(x,y,z)
        P.colorbar()
        P.title('Function')

        P.subplot(122)
        P.pcolor(xfine,yfine,zfine)
        P.colorbar()
        P.title('Smoothed')
        
        print time.time() - starttime
        


    def CompareModelToData(self):

        """Given a set of model parameters, project onto the sky and
        compare to data"""
    
        # project the model onto the sky
        self.ArrayDispOnSky()

        # Evaluate chisq
        self.ComputeChisqVel()

    def LoadSurfaceDensity(self, SurfFile = 'Espinoza_Reread.txt', \
                               DoPlot = False, ShowKing = True):

        """Load espinoza et al. surface density data"""

        # old_surffile = 'Espinoza_10-30Msol.txt'

#        SurfFile = 'Espinoza_Reread.txt'

        if not os.access(SurfFile, os.R_OK):
            return

        # format is R, log10(counts), lower lim
        RProj, LogNum, LogMinus, LogPlus = N.loadtxt(SurfFile, unpack = True)
        
        # lognum was offset by +0.5 dex for clarity
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

        if not DoPlot:
            return

        for iRow in range(N.size(Num)):
            print "%.3f %8.2f %8.2f %6.2f %11.2f" \
                % (RProj[iRow], \
                       N.abs(Num[iRow] - NumLow[iRow]), \
                       N.abs(Num[iRow] - NumHigh[iRow]), \
                       N.sqrt(Num[iRow]), Num[iRow])

        # try curve fitting
        errfunc = lambda P, X, Y, Err: (Y - self.FunctionPlummerSurfOwn(P, X)) / Err
        PInit = [Num[0], 0.5]
        Coeffs = leastsq(errfunc, PInit, \
                             args = (RProj, Num, NumErr), full_output = 0)

        Coeffs = Coeffs[0]

        # try king profile
        errKing = lambda P, X, Y, Err: (Y - self.FunctionKingSurfOwn(P,X)) / Err
        PInitKing = [Num[0], 0.2, 5.0]
        CoeffsKing = leastsq(errKing, PInitKing, \
                                 args = (RProj, Num, NumErr), full_output=0)
        CoeffsKing = CoeffsKing[0]

        # restricted king - two parameters, fixed tidal radius
        RTid = 10.0
        errKingPart = lambda P, X, Y, Err: (Y - self.FunctionKingSurfOwn([P[0], P[1], RTid],X)) / Err
        PInitKingPart = [Num[0], 0.2]
        CoeffsKingPart = leastsq(errKingPart, PInitKingPart, \
                                     args = (RProj, Num, NumErr), \
                                     full_output=0)
        CoeffsKingPart = CoeffsKingPart[0]

        # evaluate chisq at the best-fitting coeffs
        EvalBest = self.FunctionPlummerSurfOwn(Coeffs, RProj)
        Chisq = N.sum( ( (Num - EvalBest) / NumErr)**2)
        print "DBG: CHISQ at best fit %.2f %.2f" \
            % (Chisq, Chisq/(1.0 * (N.size(RProj)-2.0)))

        
        Coeffs2 = leastsq(errfunc, PInit, \
                             args = (RProj, Num, Num**0.5), full_output = 0)

        Coeffs2 = Coeffs2[0]

        # expand xdum if /show_king
        xdummax = N.max(RProj*1.5)
        if ShowKing:
            xdummax = RTid * 0.75
            print CoeffsKing

        xdum = N.linspace(0.01, xdummax, 500)
        ydum = self.FunctionPlummerSurfOwn(Coeffs,xdum)
        ydum2 = self.FunctionPlummerSurfOwn(Coeffs2,xdum)

        self.FreeFigure()
        P.figure(figsize=(6,8))
        P.subplots_adjust(hspace = 0.4, wspace=0.4)
        P.subplot(211)
        P.errorbar(RProj, Num, yerr = NumErr, fmt = 'k.')
        P.plot(xdum,  ydum, 'g--')

#        if ShowKing:
#            P.plot(xdum, self.FunctionKingSurfOwn(CoeffsKingPart, xdum), 'bo')

#        P.plot(xdum, ydum2, 'r--' )
        P.title('10 < M < 30 MSol')
        P.ylabel('N')
        P.xlabel('R, pc')
        plotax = N.copy(P.axis())
#        plotax[0] = 0.01
#        plotax[1] = 1.0
#        P.axis(plotax)

        self.FreeFigure()
        subp  = '111'
        figsz = (7,6)
        if ShowKing:
            figsz = (10,6)
            subp = '121'

        P.figure(figsize=figsz)
        P.subplot(subp)
        P.loglog(RProj, Num, 'k.')
        P.errorbar(RProj, Num, yerr = NumErr, fmt = 'k.')

        P.plot(xdum,  ydum, 'g--')
#        P.plot(xdum, ydum2, 'r--' )

        if ShowKing:
            CoefTry = [CoeffsKingPart[0], CoeffsKingPart[1], RTid]
            ydumking = self.FunctionKingSurfOwn(CoefTry, xdum)
            P.plot(xdum, ydumking, 'b-')
            plotax = N.copy(P.axis())
            plotax[1] = 10.0
            plotax[2] = 1.0
#            P.axis(plotax)

        P.ylabel('Stars per pc^2')
        P.xlabel('R, pc')
        P.title('Espinoza et al. (2009): 10 < M < 120 MSol')
        print "Errors as read in:", Coeffs
        print "Poisson errors", Coeffs2

        if not ShowKing:
            P.savefig('SurfDensProfile.png')
            return

        # if ShowKing, evaluate the enclosed mass profiles in the
        # analytic case
        R0Backup = N.copy(self.ModelR0)
        self.ModelR0 = N.copy(Coeffs[1])

        # evaluate the unnormalized enclosed masses
        Enclosed2DIsoPlummer = self.EvalProfileMPlummer(xdum)
        Enclosed2DIsoKing    = self.EvalProfileMKing(xdum, CoefTry)

        # set the normalization
        Enclosed2DPlummer = Enclosed2DIsoPlummer * N.pi * Coeffs[0] * Coeffs[1]**2

#        KingK = CoeffsKing[0] / (1.0 - 1.0/N.sqrt(1.0 + (CoeffsKing[2]/CoeffsKing[1])**2) )**2
#        Enclosed2DKing = Enclosed2DIsoKing * KingK * N.pi * CoeffsKing[1]**2 * CoeffsKing[0]

        Enclosed2DKing = Enclosed2DIsoKing * N.pi * CoeffsKingPart[1]**2 \
            * CoeffsKingPart[0]

        P.subplot(122)
        P.loglog(xdum, Enclosed2DPlummer, 'g--')
        P.loglog(xdum, Enclosed2DKing, 'b-')
    
        P.xlabel("R, pc")
        P.ylabel("Stars within R")
        P.title("Enclosed number of stars")

        print Enclosed2DKing[-1] / Enclosed2DPlummer[-1]

        # restore things how they were
        self.ModelR0 = N.copy(R0Backup)

    def EvalChisqSurfDensArray(self, Dummy = True, DoPlot = False):

        """Over an array of trial-R0 values, evaluate chisq at each"""

        if Dummy:
            VecR0 = N.linspace(0.1, 0.8, 50, endpoint = True)
        else:
            try:
                VecR0 = self.TrialVects['R0']
            except:
                return

        if not N.any(VecR0):
            return

        self.SurfDensChisq = N.zeros(N.size(VecR0))        

        if DoPlot:
            P.figure(figsize=(6,8))
            P.subplots_adjust(hspace=0.3, wspace=0.3)
            P.subplot(211)
            P.loglog(self.SurfDensRProj, \
                         self.SurfDensNum, 'k.')

            P.errorbar(self.SurfDensRProj, \
                           self.SurfDensNum, \
                           yerr = self.SurfDensErr, \
                           fmt = 'k.')
            P.xlabel('R (pc)')
            P.ylabel('Num')

        for iR0 in range(N.size(VecR0)):
            self.ModelR0 = VecR0[iR0]
            ThisChisq, ThisScale = self.EvalChisqSurfDensPoint()            
            self.SurfDensChisq[iR0] = N.copy(ThisChisq)
        
            if not DoPlot:
                continue

            ThisEval = self.FunctionPlummerSurfOwn([ThisScale, VecR0[iR0]], \
                                                       self.SurfDensRProj)

            P.plot(self.SurfDensRProj, ThisEval,'g-')

        if DoPlot:
            P.subplot(212)
            P.plot(VecR0, self.SurfDensChisq, 'k')

    def EvalChisqSurfDensPoint(self, inR0 = None):

        """Given a set of surface-densities, evaluate the chi-square
        with a given R0. Find the scale-factor by optimal scaling"""

        # ensure the needed params are set
        if not N.any(self.SurfDensRProj) \
                or not N.any(self.SurfDensNum) \
                or not N.any(self.SurfDensErr):
            return

        if inR0:
            R0 = inR0
        else:
            try:
                R0 = 1.0 * self.ModelR0
            except:
                return

        fitfunc = lambda p, x: self.FunctionPlummerSurfOwn([p,R0], x)
        errfunc = lambda par, x, y, err: (y - fitfunc(par,x)) / err
        if not self.FixedCentralSurfDens:
        
            PatternPars = N.array([ 1.0, R0 ])           
            YPattern = self.FunctionPlummerSurfOwn(PatternPars, \
                                                       self.SurfDensRProj)

            PInit = self.SurfDensNum[0]
            ScaleBest = leastsq(errfunc, PInit, args=(self.SurfDensRProj, \
                                                          self.SurfDensNum, \
                                                          self.SurfDensErr))

            ScaleBest = N.float(ScaleBest[0])
        else:
            # if central surface density is fixed, use that instead
            ScaleBest = N.copy(self.FixedCentralSurfDens)

        # compute chisq 
        ParsFull = N.array([ ScaleBest, R0 ])
        ThisChisValues = errfunc(ScaleBest, self.SurfDensRProj, \
                                     self.SurfDensNum, self.SurfDensErr)

        ThisChisqTotal = N.sum(ThisChisValues**2)

        return ThisChisqTotal, ScaleBest

    def FreeFigure(self):

        """Convenience function to close out existing figure window"""
            
        try:
            P.close()
        except:
            dum = 1
            
    def FunctionPlummerSurfOwn(self, P, X):

        """Plummer surface-density curve"""

        return P[0] * (1.0 + (X/P[1])**2)**(-2.0)
        

    def FunctionKingSurfOwn(self, P, X):

        """King surface-density curve"""

        Rc = P[1]
        Rt = P[2]

        return P[0] * ( 1.0/N.sqrt(1.0 + (X/Rc)**2) - \
                            1.0/N.sqrt(1.0 + (Rt/Rc)**2 ) )**2

    def EstimateChiSurface(self, ChiLev = 4.7, CutUnphysical = True):

        """Estimate chisq surface"""

        if not N.any(self.GridChisqRad):
            return

        # calculate total chisquared
        GridChisqTotal = self.GridChisqRad + self.GridChisqTan + self.GridChisqSurf
        VecChisqTotal = N.ravel(GridChisqTotal)

        # keep an index for points within the bubble
        self.GridInBubble = N.copy(self.GridChisqRad) * 0.0

        # find chisq min
        MinChi = N.min(GridChisqTotal)

        if not CutUnphysical:
            WithinThisChi = N.where(GridChisqTotal - MinChi - ChiLev < 0.0)
        else:
            WithinThisChi = N.where((GridChisqTotal - MinChi - ChiLev < 0.0) & \
                                        (self.GridPhysical > 0))
                                        
        if N.size(WithinThisChi) < 1:
            return

        self.GridInBubble[WithinThisChi] = 1

        # ok now construct parameter-grid
        self.ParamGridFromVects()

        # rho0 and M3d are evaluations, so can quickly be performed
        # over the entire paramgrid
        self.ModelR0     = N.copy(self.GridParR0)
        self.ModelNr     = N.copy(self.GridParNr)
        self.ModelNt     = N.copy(self.GridParNt)
        self.ModelVDisp0 = N.copy(self.GridParS0)

        M3d = self.ArrayEnclosedMass3DFull(self.FiducialR3D)
        self.PointCentralVolumeDensityFull()

        # pass these up to the grid
        self.GridMFiduc3D = N.copy(M3d)
        self.GridRho0 = N.copy(self.ModelDensity0)
        self.GridMFiduc2D = N.copy(self.GridChisqRad) * 0.0

        # now calculate the enclosed mass setting sigma = 1
        self.ModelVDisp0 = 1.0

        # break the indices down into separates
        IndicesR0 = N.copy(WithinThisChi[:][0])
        IndicesNr = N.copy(WithinThisChi[:][1])
        IndicesNt = N.copy(WithinThisChi[:][2])
        IndicesS0 = N.copy(WithinThisChi[:][3])
        NumMatched = N.size(IndicesR0)
        AlreadyCalc = N.zeros(NumMatched)
        
        TimeStarted = time.time()
        NumDone = 0
        for iMatch in range(NumMatched):

            if iMatch % 100 == 1:
                sys.stdout.write("\r EstimateChiSurface INFO - %i of %i : time elapsed %.2f sec rate %.2f" \
                                     % (NumDone, NumMatched, time.time() - TimeStarted, \
                                     NumDone / ( time.time()-TimeStarted ) ))
                sys.stdout.flush()

            if AlreadyCalc[iMatch] > 0:
                continue

            
            try:

                
                ThisIndexOrig = (IndicesR0[iMatch], \
                                     IndicesNr[iMatch], \
                                     IndicesNt[iMatch], \
                                     IndicesS0[iMatch])

                

                self.ModelR0 = self.GridParR0[ThisIndexOrig]
                self.ModelNr = self.GridParNr[ThisIndexOrig]
                self.ModelNt = self.GridParNt[ThisIndexOrig]
                self.ModelVDisp0 = 1.0

                # compute the enclosed mass at this point
                MProjIntegral = self.PointEnclosedMass2DFull(self.FiducialRProj)

                # find indices of points with the same r0, Nr, Nt
                # indices at this point
                iR0Diffs = N.copy(IndicesR0)-IndicesR0[iMatch]
                iNrDiffs = N.copy(IndicesNr)-IndicesNr[iMatch]
                iNtDiffs = N.copy(IndicesNt)-IndicesNt[iMatch]
                TotalDiffs = N.abs(iR0Diffs) + N.abs(iNrDiffs) + N.abs(iNtDiffs)
                Sames = N.where((TotalDiffs < 0.5) & (AlreadyCalc < 1))[0]

                if N.size(Sames) < 1:
                    continue

                # pass everything matching this set to the output
                for iSame in range(N.size(Sames)):
                    ThisIndexOut = (IndicesR0[iMatch], \
                                        IndicesNr[iMatch], \
                                        IndicesNt[iMatch], \
                                        IndicesS0[Sames[iSame]])

                    self.GridMFiduc2D[ThisIndexOut] = MProjIntegral[0]

                # Don't need to duplicate the calculation
                AlreadyCalc[Sames] = 1

                NumDone = NumDone + N.size(Sames)

            except(KeyboardInterrupt):
                print "EstimateChiSurface INFO - keyboard interrupt detected. Exitting."
                return
                
        # don't forget -- write the output to file OK that's the
        # *integral* of the M2D done. Now convert these to M2D
        self.GridMFiduc2D = self.GridMFiduc2D * self.GridParS0**2

        # get past the stdout.flush
        print "\n"

        # write these to fits file for examination
        Mass2D = self.GridMFiduc2D[WithinThisChi]
        print "RESULT: %.2f %.2f" % (N.min(Mass2D), N.max(Mass2D))

        # now write to output
        ResFile   = 'ChisqEval.fits'
        ResFile2D = 'ChisqEval2D.fits'
        pyfits.writeto(ResFile2D, self.GridParR0, clobber=True)
        pyfits.append(ResFile2D,  self.GridParNr)
        pyfits.append(ResFile2D,  self.GridParNt)
        pyfits.append(ResFile2D,  self.GridParS0)

        pyfits.append(ResFile2D,  self.GridChisqRad)
        pyfits.append(ResFile2D,  self.GridChisqTan)
        pyfits.append(ResFile2D,  self.GridChisqSurf)
        pyfits.append(ResFile2D,  GridChisqTotal)
        pyfits.append(ResFile2D,  self.GridSurfCentr)

        pyfits.append(ResFile2D,  self.GridPhysical)
        pyfits.append(ResFile2D,  self.GridInBubble)

        pyfits.append(ResFile2D,  self.GridMFiduc3D)
        pyfits.append(ResFile2D,  self.GridMFiduc2D)
        pyfits.append(ResFile2D,  self.GridRho0)

        # indices to write (grids are getting rather large)
        IWrite = WithinThisChi
        
        # now build record array to write to fits
        LData = [N.ravel(self.GridParR0[IWrite]), \
                     N.ravel(self.GridParNr[IWrite]), \
                     N.ravel(self.GridParNt[IWrite]), \
                     N.ravel(self.GridParS0[IWrite]), \
                     N.ravel(self.GridChisqRad[IWrite]), \
                     N.ravel(self.GridChisqTan[IWrite]), \
                     N.ravel(self.GridChisqSurf[IWrite]), \
                     N.ravel(GridChisqTotal[IWrite]), \
                     N.ravel(self.GridSurfCentr[IWrite]), \
                     N.ravel(N.asarray(self.GridPhysical, 'int')[IWrite]), \
                     N.ravel(N.asarray(self.GridInBubble, 'int')[IWrite]), \
                     N.ravel(self.GridMFiduc3D[IWrite]), \
                     N.ravel(self.GridMFiduc2D[IWrite]), \
                     N.ravel(self.GridRho0[IWrite])]

        LUnits = [('R0', N.float), \
                      ('Nr', N.float), \
                      ('Nt', N.float), \
                      ('s0', N.float), \
                      ('ChiRad',  N.float), \
                      ('ChiTan',  N.float), \
                      ('ChiSurf', N.float), \
                      ('ChiTot',  N.float), \
                      ('SurfDens0', N.float), \
                      ('IsPhys',   N.float), \
                      ('InBubble', N.float), \
                      ('M3D',  N.float), \
                      ('MProj', N.float), \
                      ('Rho0', N.float)]

        pyfits.writeto(ResFile, N.rec.fromarrays(LData, LUnits), \
                           clobber = True)

        return

        print N.shape(WithinThisChi)

        # report minmax ranges
        print N.min(self.ModelDensity0), N.max(self.ModelDensity0)
        print N.min(M3d), N.max(M3d)

        # find the number of elements that don't share sigma
        NUnique = 0
        IndicesR0 = N.copy(WithinThisChi[:][0])
        IndicesNr = N.copy(WithinThisChi[:][1])
        IndicesNt = N.copy(WithinThisChi[:][2])

        # create unique number for each index
        
        NSets = 0
        AlreadyRead = N.zeros(N.size(IndicesR0))
        for iMatch in range(N.size(IndicesR0)):
        
            if AlreadyRead[iMatch] > 0:
                continue

            try:
                iR0Diffs = N.copy(IndicesR0)-IndicesR0[iMatch]
                iNrDiffs = N.copy(IndicesNr)-IndicesNr[iMatch]
                iNtDiffs = N.copy(IndicesNt)-IndicesNt[iMatch]
                TotalDiffs = N.abs(iR0Diffs) + N.abs(iNrDiffs) + N.abs(iNtDiffs)
                
                Sames = N.where((TotalDiffs < 0.5) & (AlreadyRead < 1))[0]

                if N.size(Sames) > 0:
                    NSets = NSets + 1
                AlreadyRead[Sames] = 1
            except(KeyboardInterrupt):
                print NSets

                # save grid with binary variable - inside chisq bubble
                # or not?  then loop thru unique Nr, Nt, r0
                # combinations to compute the MProj integral for each
                #
                # then multiply by MProj

        print "DBG -- number of unique iR0, iNr, iNt combinations: %i" % (NSets)
            

    def ParamGridFromVects(self):

        """Given vectors of trial parameters, turn them into a grid
        with the same dimensions as the results-grids"""

        if not N.any(self.TrialGridShape):
            return

        self.GridParR0 = N.zeros(self.TrialGridShape)
        self.GridParNr = N.copy(self.GridParR0)
        self.GridParNt = N.copy(self.GridParR0)
        self.GridParS0 = N.copy(self.GridParR0)
    
        for iR0 in range(self.TrialGridShape[0]):
            self.GridParR0[iR0, :, :, : ] = N.copy(self.TrialVects['R0'][iR0])
        
        for iNr in range(self.TrialGridShape[1]):
            self.GridParNr[:, iNr, :, : ] = N.copy(self.TrialVects['Nr'][iNr])

        for iNt in range(self.TrialGridShape[2]):
            self.GridParNt[:, :, iNt, : ] = N.copy(self.TrialVects['Nt'][iNt])

        for iS0 in range(self.TrialGridShape[3]):
            try:
                self.GridParS0[:, :, :, iS0] = N.copy(self.TrialVects['s0'][iS0])
            except:
                self.GridParS0[:, :, :, iS0] = N.copy(self.TrialVects['Final'][iS0])

    def ShowExampleSurface(self, ThisKey='M3d', vmax = None):

        """Show interpolated calculated values"""
        
        if not ThisKey in self.SubspaceDInterpX.keys():
            return

        X = self.SubspaceDInterpX[ThisKey]
        Y = self.SubspaceDInterpY[ThisKey]
        Z = self.SubspaceDInterpZ[ThisKey]
        U = self.SubspaceDInterpU[ThisKey]

        S = N.copy(Z)
        bads = N.where(U < 1)
        goods = N.where(U > 0)
        if N.size(bads) > 0:
            S[bads] = N.max(Z[goods])
                
        self.FreeFigure()
        P.figure(figsize=(8,6))
        P.subplot(121)

        S = N.log10(S)
        if vmax:
            vtop = N.log10(vmax)

        P.pshow(X, Y, S, vmax = vtop)

    # plotting routines come here
    def DrawRandomsWithinChiLev(self, ChiLev = 4.7):

        """Generate indices corresponding to uniform random samples
        within delta-chisq bubble to desired delta-chi"""
        
        """Draw random samples from parameter-space that is within the
        chisq bubble to desired delta-chi"""

        try:
            a = self.TrialsWithMProj
        except:
            return

        IsPhys = N.where( (a['IsPhys'] > 0) & \
                              (a['Nr'] > 0.1) & \
                              (a['Nt'] > 0.1) )[0]


        MinChi = N.min(a['ChiTot'][IsPhys])
        InThis = N.where(a['ChiTot'][IsPhys] - MinChi < ChiLev)[0]

        if N.size(InThis) < 1:
            return

        # ensure indices according to original ordering are used.
        InThis = IsPhys[InThis]

        # how many samples do we want?
        if not self.NumProfileSamples:
            self.NumProfileSamples = 10
        
        iUniform = N.random.random_integers(0, N.size(InThis)-1, \
                                                self.NumProfileSamples)
        self.iRandomProfiles = InThis[iUniform]

    def EvalProfilesLoop(self, Debug = False):

        """Loop through the set of random samples from the grid"""

        if not N.any(self.PlotRadii):
            print "EvalProfilesLoop FATAL - no plot radii set."
            return

        # ensure the radius vector is passed up to the
        # results-dictionary
        self.DProfiles['Radii'] = N.copy(self.PlotRadii)

        imax = self.NumProfileSamples
        if Debug:
            imax = 5

        for i in range(imax):
            self.iSample = i
            self.ProfileModelSelect()
            self.EvalProfileForPlot()

    def ProfilePlotIsotropic(self, outfile = 'IsoTest.dat'):

        """To test the routines, plot the enclosed mass under plummer
        model for isotropic case"""

        self.ModelR0 = 0.45
        self.ModelNr = 1.0
        self.ModelNt = 1.0
        self.ModelVDisp0 = 5.4
        
        if not os.access(outfile, os.R_OK):

            self.PlotRadiiN   = 40.0
            self.PlotRadiiMax = 10.0

            # set up log-spaced radii for plots
            self.PlotRadii = N.array([0.0])

            # midpoints 
            logmax = N.log10(self.ModelR0) + 1.5
            logmin = N.log10(self.ModelR0) - 2.0
            logspaces = N.linspace(logmin, logmax, 25, endpoint=True)
            addradii = 10.0**logspaces
            self.PlotRadii = N.hstack((self.PlotRadii, addradii))
            self.PlotRadiiN = N.size(self.PlotRadii)

            #        self.SetupRadiiForPlots()
            self.NumProfileSamples = 1
            self.SetupDOutForPlot()

            self.iSample = 0
            self.EvalProfileForPlot()

            IsoMass = self.EvalProfileMPlummer()
            CalMass = N.copy(self.ModelEnclosedMass2DKinem)

            ModelVectors = N.array([self.ModelR0, self.ModelNr, self.ModelNt, self.ModelVDisp0])

            N.savetxt(outfile, N.vstack((self.PlotRadii, IsoMass, CalMass)) )
        else:
            a = N.loadtxt(outfile)
            print N.shape(a)
            self.PlotRadii = a[0]
            IsoMass = a[1]
            CalMass = a[2]

        # find the scale factor for the calculated surface density
        if N.abs(N.max(IsoMass) - 1.0) < 0.1:
            
            # scale the calculated mass to produce mass in absolute
            # units
#            self.PointCentralVolumeDensityFull()
#            rho0 = self.ModelDensity0

#            Mtot = 4.0 * N.pi * rho0 * self.ModelR0**3 / 3.0
#            CalMass = CalMass * Mtot

            print "here"

            MassNorm = 1387.5562 * self.ModelVDisp0**2 * self.ModelR0 
            IsoMass = IsoMass * MassNorm
        

        # scale the enclosed masses to the same point
        maxpredict = IsoMass[-1]
        maxcalcul  = CalMass[-1]


        mnorm = maxpredict / maxcalcul
        mnorm = 1.0

        self.FreeFigure()
        P.figure(figsize=(7,7))
        P.subplots_adjust(hspace=0.4, wspace=0.4)
#        self.PlotRadii = self.PlotRadii / self.ModelR0
        P.subplot(211)
        P.loglog(self.PlotRadii, IsoMass, 'g')
        P.plot(self.PlotRadii, CalMass * mnorm, 'k.')

        plotax = N.copy(P.axis())
        plotax[3] = N.max(IsoMass)*100.0
        P.axis(plotax)

        P.xlabel('Radius in pc', fontsize=12)
        P.ylabel('M(<R) in Solar masses', fontsize=12)

           
        P.plot([3.0, 5.0], [40.0, 40.0], 'k.')
        P.plot([3.0, 5.0], [10.0, 10.0], 'g-')
        P.annotate('Numerical', (7.0, 35.0), xycoords='data', \
                       horizontalalignment='left')
        P.annotate('Analytic',  (7.0, 9.0), xycoords='data', \
                       horizontalalignment='left', color='green')

        # model parameteres
        P.annotate('R0 = %.2f pc' % (self.ModelR0), (0.1,0.9), \
                       xycoords='axes fraction', color='k', \
                       horizontalalignment='left')

        P.annotate('s0 = %.2f km/s' % (self.ModelVDisp0), (0.1,0.8), \
                       xycoords='axes fraction', color='k', \
                       horizontalalignment='left')
        
        P.annotate('M(R < Inf) = %.2e MSol' % (MassNorm), (0.95, 0.90), \
                       xycoords='axes fraction', \
                       horizontalalignment='right')
        

#        P.subplot(223)
#        P.plot(self.PlotRadii, IsoMass, 'g')
#        P.plot(self.PlotRadii, CalMass * mnorm, 'ko')

#        plotax = N.copy(P.axis())
#        plotax[3] = 1.5
#        P.axis(plotax)

#        P.xlabel('Radius (units of R_0)')
#        P.ylabel('Enclosed 2D mass')

        DoFour = False
        if DoFour:
            P.subplot(222)
            rfine = N.linspace(0.0, 1.1*N.max(self.PlotRadii), 1000)
            P.plot(rfine, self.EvalProfileMPlummer(rfine)*MassNorm, 'g')
            P.plot(self.PlotRadii, CalMass * mnorm, 'k.')

#        plotax = N.copy(P.axis())
#        plotax[3] = 1.5
#        P.axis(plotax)

#        P.title('Test of 2D enclosed-mass calculation')
            P.xlabel('Radius (pc)')
            P.ylabel('Enclosed 2D mass')

        # legends
#        P.annotate('Numerical', (0.65, 0.3), xycoords='axes fraction', \
#                       horizontalalignment='right')
#        P.annotate('Analytic',  (0.65, 0.2), xycoords='axes fraction', \
#                       horizontalalignment='right', color='green')

#        P.plot([25.0, 30.0], [0.48, 0.48], 'k.')
#        P.plot([25.0, 30.0], [0.33, 0.33], 'g-')

            P.subplot(223)
        else:
            P.subplot(212)

        goods = N.where((N.isfinite(1.0/IsoMass) > 0) & \
                            (N.isfinite(CalMass) > 0) )[0]

#        P.semilogx(self.PlotRadii[goods], (CalMass[goods] * mnorm - IsoMass[goods])/CalMass[goods], 'r.')
        P.semilogx(self.PlotRadii[goods], N.abs(CalMass[goods] * mnorm - IsoMass[goods]), 'r.')
#/CalMass[goods], 'r.')
        P.xlabel('Radius in pc', fontsize=12)
        P.ylabel('(Numerical - Analytic) in  Solar masses', fontsize=10)

        P.annotate('Test of 2D enclosed-mass calculation', (0.5,0.96), \
                       xycoords='figure fraction', \
                       horizontalalignment='center')

        P.annotate('Isotropic Plummer', (0.5, 0.925), \
                       xycoords='figure fraction', \
                       horizontalalignment='center')

        if DoFour:
            P.subplot(224)
            P.loglog(self.PlotRadii, N.abs(CalMass * mnorm - IsoMass)/IsoMass, 'ro')
            P.xlabel('Radius in pc')
            P.ylabel('|Numerical - Analytic|/Analytic')

        P.savefig('TestAgainstIso.png')


    def EvalProfileMPlummer(self, inrad = N.array([])):

        """Evaluate enclosed mass in plummer model FOR ISOTROPIC CASE"""

        if not N.any(inrad):
            rad = self.PlotRadii
        else:
            rad = inrad

        u = 1.0 + ( rad / self.ModelR0)**2
        return 1.0 - 1.0/u

    def EvalProfileMKing(self, inrad=None, P=N.array([])):

        """Evaluate the functional form of the enclosed mass in a king
        profile. Leave the normalization to the calling routine."""

        try:
            R = 1.0 * inrad
        except:
            return

        if not N.any(P):
            return inrad * 0.0

        # the King variables
        x  = ( R/P[1] )**2
        xt = ( P[2]/P[1] )**2

        part1 = N.log(1.0 + x)
        part2 = -4.0 * (N.sqrt(1.0 + x) - 1.0) / N.sqrt(1.0 + xt)
        part3 = x / (1.0 + xt)

        return part1 + part2 + part3

    def ProfilePlotLRF92(self):
        
        """Test the routines by finding radial profiles corresponding
        to LRF92"""

        self.PlotRadiiN   = 33
        self.PlotRadiiMax = 32.0

        self.SetupRadiiForPlots()
        self.NumProfileSamples = 1
        self.SetupDOutForPlot()

        self.ModelR0 = 3.966
        self.ModelNr = 0.190
        self.ModelNt = 0.531
        self.ModelVDisp0 = 8.60

        self.iSample = 0
        self.EvalProfileForPlot()

        print "\n"

    def PlotOneProfile(self, iprofile = 0):
        
        """Plot a single profile from profiles dictionary"""
        
        r = N.copy(self.DProfiles['Radii'])
        sigR = N.copy(self.DProfiles['sigR'][iprofile])
        sigT = N.copy(self.DProfiles['sigT'][iprofile])

        self.FreeFigure()
        fig = P.figure(figsize=(8,8))

        # dataset from LRF92
        DataR  = N.array([57.6, 160.9, 492.7])
        DataSR = N.array([0.027, 0.027, 0.022])*0.9
        DataER = N.array([0.003, 0.003, 0.003])*0.9
        DataST = N.array([0.024, 0.024, 0.018])*0.9
        DataET = N.array([0.003, 0.003, 0.003])*0.9

        # conversion factor - parsecs to arcseconds for M13:
        rconv = 28.5
        r = r * rconv

        # want velocity in arcsecs per century
        vconv = 4.71 * 7.24 * 10.0
        ax = fig.add_subplot(111)
        ax.plot(r, sigR / vconv)
        ax.errorbar(DataR, DataSR, yerr=DataER, fmt='wo', ecolor='b')
        ax.plot(r, sigT / vconv, 'g--')
        ax.errorbar(DataR, DataST, yerr=DataET, fmt='s', color='g', ecolor='g')


        ax.annotate('Parameters from Table 1 of LRF92', (0.95, 0.95), \
                       xycoords='axes fraction', \
                       horizontalalignment = 'right')

        ax.yaxis.set_major_locator(MaxNLocator(10))
#        ax.xaxis.set_major_locator(MaxNLocator(19))

        P.grid(which='both')

        P.title("Figure 2 of Leonard, Richer & Fahlman (1992)")
        P.xlabel('R (arcsec)')
        P.ylabel('Sigma_R, T, arcsec per century')

        P.axis([-50.0, 900.0, -0.005, 0.045])

        P.savefig('LRF92_reproduced.png')

    def EvalProfileForPlot(self):

        """Evaluate profile for a given set of parameters. Loop
        variable self.iProfile is set in calling routine
        EvalProfilesLoop"""

        # the enclosed mass, dispersion profile and volume-density
        # are analytic for all radii - evaluate these as a matter of
        # course
        TheseM3d = self.ArrayEnclosedMass3DFull(self.PlotRadii)
        TheseRho = self.ArrayVolumeDensityFull(self.PlotRadii, DoPassUp = False) 

        # 3D dispersion components
        TheseSigr = self.ArrayVDisp3Dr(self.PlotRadii)
        TheseSigt = self.ArrayVDisp3Dt(self.PlotRadii)

        # calculate dispersions on-sky
        self.ArrayDispOnSky(self.PlotRadii)

        # calculate projected mass interior to radius R
        self.ArrayEnclosedMass2DFull(self.PlotRadii)

        self.DProfiles['sigR'][self.iSample]  = N.copy(self.EvalVDispRad)
        self.DProfiles['sigT'][self.iSample]  = N.copy(self.EvalVDispTan)
        self.DProfiles['MProj'][self.iSample] = \
            N.copy(self.ModelEnclosedMass2DKinem)
        self.DProfiles['Rho'][self.iSample]  = N.copy(TheseRho)
        self.DProfiles['M3d'][self.iSample]  = N.copy(TheseM3d)
        self.DProfiles['Done'][self.iSample] = 1

    def ProfileModelSelect(self):

        """Ensure model parameters are populated"""

        try:
            iSample = self.iRandomProfiles[self.iSample]
        except:
            iSample = 1

        a = self.TrialsWithMProj
        self.ModelR0 = a['R0'][iSample]
        self.ModelNr = a['Nr'][iSample]
        self.ModelNt = a['Nt'][iSample]
        self.ModelVDisp0 = a['s0'][iSample]

        self.CheckModelIsPhysical()

        self.DProfiles['ChiTot'][self.iSample] \
            = N.copy(self.TrialsWithMProj['ChiTot'][iSample])
        
        # pass the parameters for this grid-point to the
        # master-dictionary
        for ParKey in self.Pars4Eval:
            try:
                self.DProfiles[ParKey][self.iSample] \
                    = N.copy(self.TrialsWithMProj[ParKey][iSample])
            except:
                dum = 1

        sys.stdout.write("\r ProfileModelSelect INFO: %i of %i -- %.2f, %.2f, %.2f, %.2f, %.2f, %s" \
            % (self.iSample, N.size(self.DProfiles['ChiTot']),
               self.ModelR0, self.ModelNr, \
                   self.ModelNt, self.ModelVDisp0, \
                   self.TrialsWithMProj['ChiTot'][iSample], \
                   self.ModelIsPhysical))
        sys.stdout.flush()

    def ReadTrialsWithMProj(self, infile = 'ChisqEval.fits'):
    
        """Load trials including mproj and chisq"""

        if not os.access(infile, os.R_OK):
            print "File %.2f not found" % (infile)
            return

        try:
            self.TrialsWithMProj = pyfits.getdata(infile)
        except:
            return

    def SetupRadiiForPlots(self):

        """Set up radii to evaluate profiles for results-plot"""

        if not N.any(self.PlotRadiiN):
            self.PlotRadiiN = 100

        if not N.any(self.PlotRadiiMax):
            self.PlotRadiiMax = 2.0

        self.PlotRadii = N.linspace(0.01, self.PlotRadiiMax, \
                                        self.PlotRadiiN, endpoint = True)

    def SetupDOutForPlot(self):

        """Given a results vector, initialise the results-array for
        fitted plummer profiles"""

        if not self.NumProfileSamples:
            self.NumProfileSamples = 10

        if not self.PlotRadiiN:
            self.PlotRadiiN = 10


        LKeys = ['sigR', 'sigT', \
                     'Rho0', 'M3d', 'MProj', 'Rho']

        resvec = N.zeros((self.NumProfileSamples, self.PlotRadiiN))

        self.DProfiles = {}
        for ThisKey in LKeys:
            self.DProfiles[ThisKey] = N.copy(resvec)

        # also include radii and a done/not done flag
        if N.any(self.PlotRadii):
            self.DProfiles['Radii'] = N.copy(self.PlotRadii)
        else:
            self.DProfiles['Radii']  = N.zeros(self.PlotRadii)

        self.DProfiles['Done']   = N.zeros(self.NumProfileSamples)
        self.DProfiles['ChiTot'] = N.zeros(self.NumProfileSamples)

        # set up parameter grid
        for ThisKey in self.Pars4Eval:
            self.DProfiles[ThisKey] = N.zeros(self.NumProfileSamples)

    def PlotProfile(self,X,Y,Indices, sf = 1.0, alphashow = 0.01, jmin = 0, jmax=0):

        """Convenience function to plot profiles"""

        colo='k'

        rgrid = N.linspace(0,N.max(X), 100, endpoint=True)

        for i in range(N.size(Indices)):
            j = Indices[i]

            # try spline interpolation
            tck = interpolate.splrep(X, Y[i]*sf, s=0)
            tev = interpolate.splev(rgrid, tck)
            if self.DoEPS:
                colo='%f' % (N.random.uniform(0.7,1.0))
            P.plot(rgrid, tev, 'k', alpha=alphashow, color=colo, zorder=1)

#            P.plot(X, Y[j]*sf, 'k', alpha=alphashow)

        P.plot(X, Y[jmin]*sf, 'r', zorder=2)
        P.plot(X, Y[jmax]*sf, 'b', zorder=2)

    def WriteEvalProfiles(self, FitsProfiles = 'ProfilesToPlot.fits'):

        """Having evaluated a large number of profiles, write them to
        disk"""

        PriHDU = pyfits.PrimaryHDU()
        hdulist = pyfits.HDUList([PriHDU])
        for ThisKey in self.DProfiles.keys():

            data_hdu = pyfits.ImageHDU(N.copy(self.DProfiles[ThisKey]))
            data_hdu.name = ThisKey

            hdulist.append(data_hdu)

        hdulist.writeto(FitsProfiles, clobber=True)
        hdulist.close()

    def ReadEvalProfiles(self, InFile = 'ProfilesToPlot.fits'):

        """Given a fits file with profiles to plot, read in and plot
        them"""

        if not os.access(InFile, os.R_OK):
            return

        hdulist = pyfits.open(InFile)
    
        LKeys = ['sigR', 'sigT', \
                     'Rho0', 'M3d', 'MProj', 'Rho', 'Done', 'Radii']

        LKeys.append(self.Pars4Eval)

        self.DProfiles = {}
        for ThisKey in LKeys:
            try:
                self.DProfiles[ThisKey] = N.copy(hdulist[ThisKey].data)
            except:
                dum = 1

        hdulist.close()

    def PlotEvalProfiles(self):

        """Show the profiles of dispersion and mass against projected radius"""

        if len(self.DProfiles.keys()) < 1:
            return

        # radii
        r = N.copy(self.DProfiles['Radii'])

        # "goods" array
        u = self.DProfiles['Done']
        
        Goods = N.where(u > 0)[0]
        if N.size(Goods) < 1:
            return

        # do a subset
#        Goods = Goods[0:20]

        ngoods = N.size(Goods)

        m3d = self.DProfiles['M3d']
        m2d = self.DProfiles['MProj']
        sigR = self.DProfiles['sigR']
        sigT = self.DProfiles['sigT']

        # minmax enclosed mass at radius 2 pc
        # 
        # evaluate radius closest to 0.4pc
        rmin = N.argmin(N.abs(r - 0.4))
        maxenclosed = N.copy(m2d[:,rmin])
        maxmass = N.argmax(maxenclosed[Goods])
        jmax = Goods[maxmass]
        minmass = N.argmin(maxenclosed[Goods])
        jmin = Goods[minmass]

        # colors for upper and lower limits
        colUp = 'r'
        colLo = 'b'

        # usetex starts here
        rc('text', usetex=True)

        # some syntax to make the .eps work correctly
        ZOrderData = 3

        # start with something basic
        self.FreeFigure()
        P.figure(figsize=(11,7))
        P.subplots_adjust(hspace=0.4, wspace=0.4)
        P.subplot(241)
        P.cla()

#        P.annotate('Enclosed 3D mass, projected mass, and projected dispersions', \
#                       (0.5,0.97), xycoords='figure fraction', \
#                       horizontalalignment='center')
        P.annotate(r'Parameter-sets within 95\% ("$2\sigma$") of the global minimum', \
                       (0.5, 0.95), xycoords='figure fraction', \
                       horizontalalignment='center')

        

        alphashade = 0.01

        # try plot
        self.PlotProfile(r, m3d, Goods, 1.0e-4, alphashade, jmin, jmax)
        P.title(r'M ($<$ r)')
        P.xlabel(r'r (pc)', fontsize=14)
        P.ylabel(r'$\times 10^{4}$ M$_{\odot}$')

        P.subplot(242)
        self.PlotProfile(r, m2d, Goods, 1.0e-4, alphashade, jmin, jmax)
        P.title(r'M($<$ R)')
        P.xlabel(r'R (pc)')
#        P.ylabel('x10^4 MSol')
        P.ylabel(r'$\times 10^{4}$ M$_{\odot}$')

       
        P.subplot(243)
        self.PlotProfile(r, sigR, Goods, 1.0, alphashade, jmin, jmax)
        if N.any(self.DataVDispRad):
            P.errorbar(self.R2D, self.DataVDispRad, \
                           yerr = self.DataEDispRad, fmt='wo', ecolor='k', \
                           zorder=ZOrderData)
        P.title(r'Radial')
        P.xlabel(r'R (pc)')
        P.ylabel(r'dispersion, km s$^{-1}$')

        P.subplot(244)
        self.PlotProfile(r, sigT, Goods, 1.0, alphashade, jmin, jmax)
        if N.any(self.DataVDispTan):
            P.errorbar(self.R2D, self.DataVDispTan, \
                           yerr = self.DataEDispTan, fmt='wo', ecolor='k', \
                           zorder=ZOrderData)

        P.title(r'Tangential')
        P.xlabel(r'R (pc)')
        P.ylabel(r'dispersion, km s$^{-1}$')

        P.subplot(245)
        self.PlotProfile(r, m3d, Goods, 1.0e-4, alphashade, jmin, jmax)
        # 1.2 was 1.4
        P.axis([0.7, 1.2, 0.0, 4.0])
        plotax = N.copy(P.axis())
        P.plot([1.0, 1.0], [plotax[2], plotax[3]], 'g--')
        P.title(r'M ($<$ r)')
        P.xlabel(r'r (pc)')
        P.ylabel(r'$\times 10^4$ M$_{\odot}$')

        P.subplot(246)
        self.PlotProfile(r,m2d, Goods, 1.0e-4, alphashade, jmin, jmax)
        P.title(r'M($<$ R)')
        P.xlabel(r'R (pc)')
        P.ylabel(r'$\times10^4$ M$_{\odot}$')
        
        P.axis([0.1, 0.6, 0.0, 2.0])
        plotax = N.copy(P.axis())
        P.plot([0.4, 0.4], [plotax[2], plotax[3]], 'g--')

        P.subplot(247)
        self.PlotProfile(r, sigR, Goods, 1.0, alphashade, jmin, jmax)
        if N.any(self.DataVDispRad):
            P.semilogx(self.R2D, self.DataVDispRad, 'w.')
            P.errorbar(self.R2D, self.DataVDispRad, \
                           yerr = self.DataEDispRad, fmt='wo', ecolor='k', \
                           zorder=ZOrderData)

        P.title(r'Radial')
        P.xlabel(r'R (pc)')
        P.ylabel(r'dispersion, km/s')

        P.subplot(248)
        self.PlotProfile(r, sigT, Goods, 1.0, alphashade, jmin, jmax)
        if N.any(self.DataVDispTan):
            P.semilogx(self.R2D, self.DataVDispTan, 'k.')
            P.errorbar(self.R2D, self.DataVDispTan, \
                           yerr = self.DataEDispTan, fmt='wo', ecolor='k', \
                           zorder=ZOrderData)

        P.title(r'Tangential')
        P.xlabel(r'R (pc)')
        P.ylabel(r'dispersion, km s$^{-1}$')
        
        # save figure
        P.savefig('RadProfiles.eps',format="eps_cmyk")
        



def FunctionPlummerSurfDens(P, X):

    """Plummer surface-density curve"""

    return P[0] * (1.0 + (X/P[1])**2)**(-2.0)

def TestFunction2(Y, X, n = 2.0):

    return N.exp(-Y*X)/X**n


def TestPlummer3D(alpha = 1.0, beta = 1.0, \
                      r0 = 0.4, rmin = 0.0, rmax = 10.0, \
                      sigma_0 = 4.0, npoints = 100):

    """Test routine for plummer model 3D enclosed mass"""

    rrad = N.arange(npoints)/(1.0*npoints) * (rmax - rmin) + rmin
    
    A = Plummer()
    A.ModelNr = alpha
    A.ModelNt = beta
    A.ModelR0 = r0
    A.ModelVdisp0 = sigma_0
    A.r3d = rrad

    A.EvalEnclosedMass3DKinem()
    M = A.ModelEnclosedMass3DKinem
    
    P.plot(rrad,M,'b.')
    P.title('Enclosed mass, 3D')

def TestIntegral(n = 2):

    """Tester for double integral in scipy called from within class"""

    A = Plummer()
    A.TestInteg2(n)

def TestMass3D(r0 = 0.4, rmax = 5.0, nradii = 50, rho0 = 2.0):

    """Perform 3D integral on plummer enclosed mass"""

    A = Plummer()
    A.ModelR0 = r0
    radii = (N.arange(nradii)+1.0)/(nradii*1.0) * rmax
    mencl = N.zeros(N.size(radii))
    for irad in range(N.size(radii)):
        dum = A.PointEnclosedMass3D(radii[irad], rho0)
        mencl[irad] = dum[0]

    MaxMass = A.PointEnclosedMass3D(Inf, rho0)[0]

#    # try the analytic form
#    MAnalytic = radii**3 * (radii**2 + r0**2)**(-3.0/2.0) 

#    print N.mean(mencl / MAnalytic), MaxMass

#    MCalc = MAnalytic * MaxMass

    # try analytic
    radfine = N.arange(50000)/50000.0 * rmax
    MCalc = A.ArrayEnclosedMass3D(radfine, rho0)

    # where is the half-mass in 3D space?
    ihalf = N.argmin(N.abs(MCalc / MaxMass - 0.5))
    print radfine[ihalf], MCalc[ihalf] / MaxMass, radfine[ihalf] / r0

    MCalc = A.ArrayEnclosedMass3D(radii, rho0)
    

    # so -- if evaluating a vector 

    P.clf()
    P.plot(radii, mencl, 'ko')
    P.xlabel("r")
    P.ylabel("Mass within sphere radius r")
    P.plot(radii, MCalc, 'g-')

def TestMassProj(r0 = 0.4, Rf = 0.4):

    """Test the mass within projected radius R integral"""

    A = Plummer()

    A.ModelDensity0 = 1.0

    nradii = 50.0
    radii = (N.arange(nradii)+1.0)/(nradii*1.0) * 5.0
    mencl = N.zeros(N.size(radii))

    A.ModelR0 = r0

    for iRadius in range(N.size(radii)):
        IntegTest = A.PointEnclosedMass2D(radii[iRadius])
        mencl[iRadius] = IntegTest[0]

    # find the total mass (at R = infinity)
    IntegInfty = A.PointEnclosedMass2D(Inf)

    # recast as the half-mass radius
    maxmass = IntegInfty[0]
    
    print maxmass
    closest = N.argmin(N.abs(mencl / maxmass - 0.5))
    print radii[closest], mencl[closest]/maxmass

    P.clf()
    P.plot(radii, mencl/maxmass, 'k.')
    print IntegTest
    
def TestDenominator(r0 = 0.4, \
                        npoints = 50, \
                        rmin = 0.0, rmax = 5.0):

    """Test the denominator expression in the velocity dispersions"""

    A = Plummer()
    A.ModelR0 = r0

    radii = N.arange(npoints) / (npoints * 1.0) * (rmax - rmin) + rmin
    sigma = radii * 0.0
    for irad in range(N.size(radii)):
        integ = A.PointSurfDensInteg(radii[irad])
        sigma[irad] = integ[0]

    P.clf()

    # evaluate at zero
    LowLimit = A.PointSurfDensInteg(0.0)
    DensCalc = (1.0 + (radii/r0)**2)**(-2.0)

    print N.mean(sigma / DensCalc), N.std(DensCalc / sigma), LowLimit[0]

    # call our one-shot routine
    radfine = N.arange(50000)/50000.0 * rmax
    TryTwo = A.ArraySurfDensInteg(radfine)

    P.plot(radii, sigma, 'k.')
    P.plot(radii, DensCalc * LowLimit[0], 'g-')
    P.plot(radfine, TryTwo, 'b.')
    
def TestDispTang(r0 = 0.4, beta = 1.0, sigma0 = 1.0, \
                     npoints = 50, \
                     rmin = 0.0, rmax = 5.0):

    """Test the projection sigma(T)"""

    A = Plummer()
    A.ModelR0 = r0
    A.ModelNt = beta
    A.VDisp0 = sigma0

    radii = N.arange(npoints) / (npoints * 1.0) * (rmax - rmin) + rmin
    A.ArrayDispTanInteg(radii, DoPlot = True)
    
    return


def TestDispRad(r0 = 0.4, beta = 1.0, sigma0 = 1.0, alpha = 1.0, \
                    npoints = 50, \
                    rmin = 0.0, rmax = 5.0):

    """Test the projection sigma(R)"""

    A = Plummer()
    A.ModelR0 = r0
    A.ModelNt = beta
    A.ModelNr = alpha
    A.VDisp0 = sigma0

    radii = N.arange(npoints) / (npoints * 1.0) * (rmax - rmin) + rmin
    A.ArrayDispRadInteg(radii, DoPlot = True)
    
    return

# generate projected components of velocity dispersion from model
def TestGenDisp(r0 = 0.4, sigma0 = 4.0, alpha = 1.0, beta = 1.0, \
                     npoints = 10, rmin = 0.05, rmax = 3.0, InError = 0.5, \
                    UseMProj = False):

    """Given a model, generate motion-dataset"""

    G = Plummer()
    G.TestModelR0 = r0
    G.TestModelNt = beta
    G.TestModelNr = alpha
    G.TestModelVdisp0 = sigma0
    G.TestModelErrorRad = InError
    G.TestModelErrorTan = InError
    G.TestModelMProj = 10000.00

    RObs = N.arange(npoints) / (npoints * 1.0) * (rmax - rmin) + rmin

    G.UseMProj = UseMProj
    G.GenerateDataFromModel(RObs)

    # estimate projected fiducial mass from the fitted sigma
    MProjCalc = G.PointEnclosedMass2DFull(G.FiducialRProj, FindingSigma = False)

    print "TestGenDisp DBG: MProj at fiducial, sigma_0: ", G.ModelVDisp0, G.TestModelMProj, MProjCalc[0]

    R = G.TestObsRadii

    try:
        P.close()
    except:
        dum = 1

    P.figure(figsize=(10,5))
    P.subplots_adjust(hspace=0.3, wspace = 0.3)
    P.subplot(121)
    P.errorbar(R, G.DataVDispRad, yerr = G.DataEDispRad, fmt='o', color='k')
    P.title('Radial')

    plotax = N.copy(P.axis())
    xmax = plotax[1] * 1.2
    ymax = 10.0
    plotax[1] = xmax
    plotax[2] = 0.0
    plotax[3] = ymax
    P.axis(plotax)

    P.subplot(122)
    P.errorbar(R, G.DataVDispTan, yerr = G.DataEDispTan, fmt='s', color='g')
    P.title('Tangential')
    plotax = N.copy(P.axis())
    plotax[1] = xmax
    plotax[2] = 0.0
    plotax[3] = ymax
    P.axis(plotax)

    # loop through parameters
    MProjLoop = N.arange(10)/10.0 * (20000.0) + 2000.0
    VDispLoop = N.arange(10)/10.0 * (10.0) + 0.4

    print "Looping through trials..."
    for iMass in range(N.size(MProjLoop)):

        # set the mass and ensure model is fully parameterised
        G.ModelMFiducial = MProjLoop[iMass]
        G.ModelVDisp0 = VDispLoop[iMass]
        G.ModelDispAndMProj()

        # project on-sky
        G.ArrayDispOnSky(RObs)

        # estimate chisq
        G.ComputeChisqVel()

        print "%.2f, %.2f, %.2f, %.2f" % (G.ModelVDisp0, G.ModelMFiducial, G.EvalChisqRad, G.EvalChisqTan)


        # and plot
        P.subplot(121)
        P.plot(RObs, G.EvalVDispRad,'r--')
        P.annotate("%.2f" % (G.EvalChisqRad), (xmax-0.01, G.EvalVDispRad[-1]), \
                       xycoords='data', horizontalalignment='right', \
                       verticalalignment='middle', color='r')

        P.subplot(122)
        P.plot(RObs, G.EvalVDispTan,'g--')
        P.annotate("%.2f" % (G.EvalChisqTan), (xmax-0.01, G.EvalVDispTan[-1]), \
                       xycoords='data', horizontalalignment='right', \
                       verticalalignment='middle', color='g')

    print "... done"

# test function for density and enclosed mass
def TestShowFunc(r0 = 0.4, sigma0 = 4.0, alpha = 1.0, beta = 1.0, \
                     npoints = 50, rmin = 0.0, rmax = 10.0):

    """Test function to see what the behaviour is of enclosed mass and
    volume density, both of which are analytic"""
    
    A = Plummer()
    A.ModelR0 = r0
    A.ModelNt = beta
    A.ModelNr = alpha
    A.ModelVDisp0 = sigma0

    radii = N.arange(npoints) / (npoints * 1.0) * (rmax - rmin) + rmin
    
    # send the radii to the class
    A.r3d = radii
    A.R2D = radii

    # code to do the plotting comes here. Now just a quick function call.
#    MassEnclosed  = A.ArrayEnclosedMass3DFull(radii)
#    VolumeDensity = A.ArrayVolumeDensityFull(radii)
    
    MassEnclosed =  A.ArrayEnclosedMass3DFull()
    VolumeDensity = A.ArrayVolumeDensityFull(DoPassUp = True)
    A.PointCentralVolumeDensityFull()

    print A.ModelDensity0, N.max(VolumeDensity)
    A.CheckModelIsPhysical(DEBUG = True)

    # evaluate the projection of this into velocities on the plane of
    # the sky
    A.ArrayDispOnSky()
    
#    # and plot them
    try:
        P.close()
    except:
        dum = 1

    P.figure()
    P.subplot(211)
    P.plot(radii, A.EvalVDispRad,'ko')
    P.title('Radial')
    P.subplot(212)
    P.plot(radii, A.EvalVDispTan,'gs')
    P.title('Transverse')
    P.savefig('Show_profiles.png')
    P.close()

#    return

    print "DBG - model check for physicality: %s" % (A.ModelIsPhysical)

    # try going through a range of parameter-sets
    P.clf()
    avalues = [1.0, 0.9, 0.5, 1.1, 1.10]
    bvalues = [1.0, 0.9, 0.9, 1.1, 0.80]

    DoShowProfiles = False
    if DoShowProfiles:
        for j in range(N.size(avalues)):
#        SurfDens2D = N.zeros(N.size(radii))

            A.ModelNr = avalues[j]
            A.ModelNt = bvalues[j]
            A.ArraySurfDens2DFull(radii)
            SurfDens2D = N.copy(A.ModelSurfDens2DKinem)

            print N.size(SurfDens2D), N.size(radii)

#        for iPoint in range(N.size(radii)):
#            A.ModelNr = avalues[j]
#            A.ModelNt = avalues[j]
#            SurfDens2D[iPoint] = A.PointSurfDens2DFull(radii[iPoint])[0]
            
            psym = 'ko'
            if j < 1:
                psym = 'wo'
            else:
                P.semilogy(radii,SurfDens2D*radii,'k--')

            P.semilogy(radii,SurfDens2D*radii,psym)
                
            P.annotate("%.2f, %.2f" % (avalues[j], bvalues[j]), \
                           (N.max(radii), SurfDens2D[-1]*radii[-1]*2.0), \
                           xycoords='data', horizontalalignment='right', \
                           verticalalignment='bottom')

        # overplot isotropic case
            if j < 1:
                yplummer = (1.0 + (radii/r0)**2)**(-2.0)
                yplummer = yplummer * SurfDens2D[0] / yplummer[0]
                P.semilogy(radii, yplummer*radii, 'g-')

        return

    try:
        P.close()
    except:
        dum = 1

    P.figure(figsize=(9,7))
    P.subplots_adjust(wspace=0.3, hspace=0.3)
    P.subplot(221)
    P.plot(radii, MassEnclosed, 'ko')
    P.ylabel('Enclosed mass')
    P.title('Enclosed Mass, 3D')

    print "DBG:",VolumeDensity[0:10]

    P.subplot(222)
    P.plot(radii, VolumeDensity, 'gs')
    P.xlabel('Radius')
    P.ylabel('Volume density')
    P.title('Volume density rho(r)')

    print "CHECK DBG: ", VolumeDensity[0] < 0

    # now evaluate the projected enclosed mass

    if not A.ModelIsPhysical:
        print "Model not physical (rho < 0 for some radius) - Not continuing with surface density"
        return

    print "Evaluating enclosed 2D mass..."
    A.DoSurfDens2D = True
    A.ArrayEnclosedMass2DFull(radii)
    print "... done."
    P.subplot(223)
    P.plot(radii, A.ModelEnclosedMass2DKinem, 'k.')
    P.title('Enclosed mass, 2D')

    # overplot the 3D version
    scaled3d = MassEnclosed #* A.ModelEnclosedMass2DKinem[-1] / MassEnclosed[-1]
    P.plot(radii, scaled3d,'r--')

    print A.ModelSurfDens2DKinem[0:5]

    if not N.any(A.ModelSurfDens2DKinem):
        return
    
    P.subplot(224)
    P.loglog(radii, A.ModelSurfDens2DKinem, 'bo')

    # overplot the plummer expectation
    yplummer = (1.0 + (radii/r0)**2)**(-2.0) 
    yplummer = yplummer * (A.ModelSurfDens2DKinem[0] / yplummer[0])
    P.loglog(radii, yplummer,'g--')

    P.title('Surface density Sigma(R)')
    P.xlabel('R')

    P.savefig('Show_masses.png')

    print A.ModelIsPhysical

# try setting up grid of parameters to search
def TestSetupGrid(r0 = 0.4, sigma0 = 5.4, alpha = 1.0, beta = 1.0, \
                      npoints = 7, rmin = 0.02, rmax = 1.0, InError = 0.5, \
                      UseMProj = True, ReadInData=True, DoFig = False, \
                      ReducedSet = False, NumR = 12, NumNr = 20, NumNt = 20, \
                      LoopFile = 'RunGridLoopsVec.fits', \
                      FixedDens0 = None):

    # example call:
    #
    # plummer.TestSetupGrid(UseMProj = False, ReadInData = False, npoints=7, rmax = 0.7, alpha = 0.5, beta = 0.75)

    A = Plummer()
    A.UseMProj = UseMProj
    A.SetupTrialRanges()
    A.UseSurfDens = True

    A.FixedCentralSurfDens = FixedDens0

#   don't calc while fitting... we defer this to later for the subset
#    of points within 4D chisq bubble 

#    if A.UseMProj:
#        A.DoCalcWhileFitting = True

    # monkey aroudn a bit with some of the parameters
    A.TrialGrids['R0']['Lo']  = 0.1
    A.TrialGrids['R0']['Hi']  = 0.8
    A.TrialGrids['R0']['Num'] = NumR
    
    A.TrialGrids['Nr']['Lo']  = 0.0
    A.TrialGrids['Nr']['Hi']  = 8.0 # was 2.5
    A.TrialGrids['Nr']['Num'] = NumNr
    A.TrialGrids['Nt']['Lo']  = 0.0
    A.TrialGrids['Nt']['Hi']  = 8.0  # was 2.5
    A.TrialGrids['Nt']['Num'] = NumNt

    # central velocity dispersion
    A.TrialGrids['s0']['Lo']  = 2.0
    A.TrialGrids['s0']['Hi']  = 8.0
    A.TrialGrids['s0']['Num'] = 44

    # projected mass grids
    A.TrialGrids['MProj']['Lo']  = 2000
    A.TrialGrids['MProj']['Hi']  = 30000.0
    A.TrialGrids['MProj']['Num'] = 28 # was 28

    # reduced set for testing?
    if ReducedSet:
        A.TrialGrids['R0']['Num'] = 4
        A.TrialGrids['Nr']['Num'] = 9
        A.TrialGrids['Nt']['Num'] = 9

    A.SetupTrialParamVecs()
    A.SetupTrialGridShape()
    A.SetupResultsGrid()
    A.SetupOptGrid()

    # set up some data
    if not ReadInData:
        A.TestModelR0 = r0
        A.TestModelNt = beta
        A.TestModelNr = alpha
        A.TestModelVdisp0 = sigma0
        A.TestModelErrorRad = InError
        A.TestModelErrorTan = InError
        A.TestModelMProj = 10000.00
        RObs = N.arange(npoints) / (npoints * 1.0) * (rmax - rmin) + rmin
        A.R2D = RObs
        A.r3d = RObs
        A.GenerateDataFromModel(RObs)

    else:
        print "DBG -- reading in data"
        A.DataReadCaptured()

    if DoFig:
        try:
            P.close()
        except:
            dumdum = 1

        RObs = A.R2D
        P.figure(figsize=(10,5))
        P.subplots_adjust(hspace=0.3, wspace = 0.3)
        P.subplot(121)
        P.errorbar(RObs, A.DataVDispRad, yerr = A.DataEDispRad, fmt='o', color='k')
        P.title('Radial')
        P.xlabel('R (pc)')

        plotax = N.copy(P.axis())
        xmax = plotax[1] * 1.2
        ymax = 10.0
        plotax[1] = xmax
        plotax[2] = 0.0
        plotax[3] = ymax
        P.axis(plotax)

        P.subplot(122)
        P.errorbar(RObs, A.DataVDispTan, yerr = A.DataEDispTan, fmt='s', color='g')
        P.title('Tangential')
        P.xlabel('R (pc)')
        plotax = N.copy(P.axis())
        plotax[1] = xmax
        plotax[2] = 0.0
        plotax[3] = ymax
        P.axis(plotax)

        P.savefig('ShowData.png')
        P.close()

    # loop throug hall four dimensions of the grid...
    PSizes = A.TrialGridShape
    print "DBG - starting loops: %i" % (PSizes[0] * PSizes[1] * PSizes[2])
    A.NumToWrite = 50
    
    # try vectorized
    A.GridFitsFile = LoopFile
    A.GridRunLoopsVec() # this also writes the trials to disk

    # write results out
    A.OptGridWriteColumns('RunGridTrials2D_Opt.fits')
    A.GridConvertTo2D()
    A.GridWriteFits2D('RunGridTrials2D.fits')

#    TestReadTrials(A.GridFitsFile,'RunGridLoopsVec2D.fits')

    # arrays for comparison
#    ChiRadVec = N.copy(A.GridChisqRad)
#    ChiTanVec = N.copy(A.GridChisqTan)


    return

    # run the long way...
    A.GridFitsFile = 'RunGridLoopsOld.fits'
    A.GridRunLoops()
    TestReadTrials(A.GridFitsFile,'RunGridLoopsOld2D.fits')

    if not A.CompletedLoops:
        print "Loops not finished. Not continuing"
        return
    print "... Done."

    # ... and compare the results
    print "Comparing results:"
    print N.sum(ChiRadVec - A.GridChisqRad)
    print N.sum(ChiTanVec - A.GridChisqTan)


    return

    # take a look at the results
    for iR0 in range(PSizes[0]):
        for is0 in range(PSizes[3]):
            print "%.2f %.2f -- %.2f %.2f" % (A.TrialVects['R0'][iR0], \
                                                  A.TrialVects['MProj'][is0], \
                                                  A.GridRho0[iR0][0][0][is0] / 1.0e4, \
                                                  A.GridChisqRad[iR0][0][0][is0])


def TestReadCaptured():

    A = Plummer()
    A.DataReadCaptured()

    try:
        P.close()
    except:
        dummy = 1

    P.figure(figsize=(7,7))
    P.subplots_adjust(hspace=0.4, wspace = 0.3)
    P.subplot(211)
    P.errorbar(A.R2D, A.DataVDispRad, yerr = A.DataEDispRad, fmt='o')
    P.axis([0.0, 0.25, 0.0, 10.0])
    P.title('Radial')
    P.xlabel('R (pc)')
    P.ylabel('Dispersion, km/s')

    P.subplot(212)
    P.errorbar(A.R2D, A.DataVDispTan, yerr = A.DataEDispTan, fmt='s')
    P.axis([0.0, 0.25, 0.0, 10.0])
    P.title('Tangential')
    P.xlabel('R (pc)')
    P.ylabel('Dispersion, km/s')

    
# try reading in a trial-file and converting its results to a 2D array
#
# then write out so that we can use e.g. TOPCAT to play with the
# results
def TestReadTrials(InFits='RunGridTrials.fits', OutFits = 'RunGridTrials2D.fits'):

    C = Plummer()
    C.GridReadFits(InFits)
    print N.shape(C.GridMFiduc2D)
    C.GridConvertTo2D()
    print "here", N.shape(C.GridResults2D)
    C.GridWriteFits2D(OutFits)

def TestMinimizeArr(Key1='R0', Key2='Final', InFits = 'RunGridTrials.fits', \
                        IndRep0 = None, IndRep1 = None, CutBads = True, \
                        iplot = 5, SigmaNudge = 0.0):

    A = Plummer()
    A.GridReadFits(InFits)
    A.SubsetSetupKeys(Key1, Key2)
    A.CutUnphysical = CutBads
    A.SubsetSetupGrids()

    # do not use this usually... this is to investigate the effect of adding 2 to the dispersions
    if SigmaNudge:
        A.TrialVects['Final'] = A.TrialVects['Final'] + SigmaNudge

    A.SubsetLoopMinimize()
    A.PlotNum = iplot

    A.UseSurfDens = True
    try:
        dum = 1.0 * IndRep1
        dum = 1.0 * IndRep0

        print N.shape(A.SubsetChisqOld)
        print N.log10(A.SubsetChisqOld[IndRep0, IndRep1])
    except:
        dum = 1

        
    # try showing the results
    try:
        P.close()
    except:
        dum = 1

    surf = A.SubsetChisqOld
    bads = N.where(surf > 1000) # this was using chisq <= 0 as condition
    goods = N.where(surf < 1000)
    if N.size(bads) > 0:
        surf[bads] = N.max(surf[goods])

    DTested = A.SubsetParams
    titlestring1 = A.SubsetKeysVary[0]
    titlestring2 = A.SubsetKeysVary[1]
    if 'Final' in titlestring2:
        titlestring2 = 's0'
    ptitle = "(%s , %s) fit" % (titlestring1, titlestring2)


    # show combo parameters
    P.figure(figsize=(11,11))
    P.subplot(221)
    P.subplots_adjust(hspace = 0.5, wspace = 0.5)

    # first the exponents
    P.pcolor(DTested[Key1], DTested[Key2], surf, vmax=N.max(surf[goods]))
    a = P.colorbar()
    a.set_label('Chisq')
    P.xlabel(A.SubsetKeysFixed[0])
    P.ylabel(A.SubsetKeysFixed[1])
    P.title(ptitle)

    # then show the fitted values
    P.subplot(222)
    surf = N.copy(DTested[A.SubsetKeysVary[0]])
    surf[bads] = N.max(surf[goods])
    P.pcolor(DTested[Key1], DTested[Key2], surf, vmax=N.max(surf[goods]))
    a = P.colorbar()
    a.set_label(titlestring1)
    P.xlabel(A.SubsetKeysFixed[0])
    P.ylabel(A.SubsetKeysFixed[1])
    P.title(ptitle)

    P.subplot(223)
    surf = N.copy(DTested[A.SubsetKeysVary[1]])
    surf[bads] = N.max(surf[goods])
    P.pcolor(DTested[Key1], DTested[Key2], surf)
    a = P.colorbar()
    a.set_label(titlestring2)
    P.xlabel(A.SubsetKeysFixed[0])
    P.ylabel(A.SubsetKeysFixed[1])
    P.title(ptitle)



    P.savefig('ShowTrialArray.png')

    # now that this is done, compute the quantities of interest over
    # the grid and interpolate to enforce uniform grid in the quantity
    # of interest.
    A.SubsetCalcMinimized()

def TestMinimizeStep(Key1='R0', Key2='Final', i0 = 0, i1 = 4, DoLoops = False, DoLog = False, DoSurf = False, CutBads = False):

    A = Plummer()
    A.GridReadFits('RunGridLoopsVec.fits')
    A.SubsetDoPlot = DoSurf
    A.SubsetPlotLog = DoLog
    A.CutUnphysical = CutBads
    A.SubsetTestMinimize(Key1, Key2, i0, i1)
    
    # do a plot to show the original data at this point
    try:
        P.savefig('ExampleSubspace.png')
    except:
        dum = 1

def TestMinGeneral(Key1='R0',i1=0,Key2='Final', i2 = 8, DoLog = False, DoCalc = True, CalcChoice='Rho0'):

    A = Plummer()
    A.GridReadFits('RunGridTrials.fits')

    # get subset to vary
    A.Subspace2DInitialise()
    A.Subspace2DGetSubset(Key1, Key2, i1, i2)
    A.Subspace2DMakeGrids()

    A.SubsetPlotLog = DoLog

    if DoCalc:
        A.Subspace2DCalcShow()
        A.Subspace2DInterpCalc('Rho0', NumInterp = 1000, CutClosest = False)
        A.Subspace2DInterpCalc('M3d',  NumInterp = 400)
        A.Subspace2DInterpWrite()

        return

    # call the minimizer
    xmin, ymin, zmin = A.FindMinimumTwo(A.SubspaceXGrid, A.SubspaceYGrid, \
                                            A.SubspaceZGrid, A.SubspaceUGrid)

    try:
        P.close()
    except:
        dum = 1

    # name and value of fixed parameters here
    Name1 = A.SubspaceKeysFixed[0]
    Inde1 = A.SubspaceIndsFixed[0]
    Val1  = A.TrialVects[Name1][Inde1]
    Name2 = A.SubspaceKeysFixed[1]
    Inde2 = A.SubspaceIndsFixed[1]
    Val2  = A.TrialVects[Name2][Inde2]
        
    P.figure()
    zcolor = A.SubspaceZGrid
    if DoLog:
        zcolor = N.log10(zcolor)
    P.pcolor(A.SubspaceXGrid,A.SubspaceYGrid,zcolor)
    P.colorbar()
    P.plot(xmin,ymin,'wx', markersize=5)
    P.plot([N.min(A.SubspaceXGrid), N.max(A.SubspaceXGrid)], [ymin, ymin],'y--')
    P.plot([xmin, xmin], [N.min(A.SubspaceYGrid), N.max(A.SubspaceYGrid)],'y--')
    P.xlabel(A.SubspaceKeysVary[0])
    P.ylabel(A.SubspaceKeysVary[1])
    P.title("%s = %.3f, %s = %.3f" % (Name1, Val1, Name2, Val2))

    print A.SubspaceValsFixed

# test minfinding given surface
def TestMinFromTrials(R0 = 0.3, MProj = 1.0e4, DoMProj = False):

    """Read in a trials-file, estimate the minimum chisq when a subset
    of params are fit at a given location for pars"""
    
    A = Plummer()
    
    # read in the trials
    A.GridReadFits('RunGridTrials.fits')

    # select one example radius
    MinDistRad = N.min(N.abs(A.TrialVects['R0'] - R0))*1.01
    MinDistMas = N.min(N.abs(A.TrialVects['Final'] - MProj))*1.01
    MinDistNr  = N.min(N.abs(A.TrialVects['Nr'] - 0.8))*1.01
    iThisRad   = N.where(N.abs(A.TrialVects['R0'] - R0) <= MinDistRad)[0][0]
    iThisMProj = N.where(N.abs(A.TrialVects['Final'] - MProj) <= MinDistMas)[0][0]
    iThisNr    = N.where(N.abs(A.TrialVects['Nr'] - 0.8) <= MinDistNr)[0][0]
    ChiTot = A.GridChisqRad + A.GridChisqTan

    # form x,y,z grid from this
    xgrid, ygrid = N.meshgrid(A.TrialVects['Nr'], A.TrialVects['Nt'])
    zgrid = ChiTot[iThisRad,:,:,iThisMProj]

    if DoMProj:
        xgrid, ygrid = N.meshgrid(A.TrialVects['Nt'], A.TrialVects['Final'])
        zgrid = ChiTot[iThisRad,iThisNr,:,:]

        # try this the function way
        print "DBG: attempting automated way"
        A.Subspace2DGetSubspace('R0', 'Nr', iThisRad, iThisNr)
    else:
        A.Subspace2DGetSubspace('R0', 'Final', iThisRad, iThisMProj)
        

    # try extracting vectors here
    print "Extraction DBG:", N.min(zgrid - A.SubspaceZOrig), N.max(zgrid - A.SubspaceZOrig)
    print "Extraction DBG:",A.SubspaceKeysFixed
    print "Extraction DBG:",A.SubspaceKeysVary

#    ztrial = ChiTot[A.SubspaceIndices]    
#    print "DBG:", len(A.SubspaceIndices)
#    print "DBG:" ,N.max(ztrial - zgrid), N.min(ztrial - zgrid)

    zgrid = N.transpose(zgrid)
    ugrid = N.ones(N.shape(zgrid))

    # do some selection
    badvals = N.where(zgrid > 1.0e8)
    ugrid[badvals] = 0
    goodvals = N.where(zgrid < 1.0e8)
    zgrid[badvals] = N.max(zgrid[goodvals])  # for shade-plot

    tzero = time.time()
    xmin, ymin, zmin = A.FindMinimumTwo(xgrid, ygrid, zgrid, ugrid)
    print "%e " % (time.time() - tzero)

    print xmin, ymin, zmin


    # show the results first
    try:
        P.close()
    except:
        dum = 1
        
    print N.shape(xgrid), N.shape(ygrid), N.shape(zgrid)


    P.figure()
    P.pcolor(xgrid,ygrid,zgrid)
    P.colorbar()
    P.plot(xmin,ymin,'wx', markersize=5)
    P.plot([N.min(xgrid), N.max(xgrid)], [ymin, ymin],'y--')
    P.plot([xmin, xmin], [N.min(ygrid), N.max(ygrid)],'y--')

# test surface minfinding
def TestFindMin():

    A = Plummer()
    x,y = N.mgrid[-1:1:20j,-1:1:10j]
    z = (x+y)*N.exp(-6.0*(x*x+y*y))

    # make x, y 1D arrays
    xr = N.ravel(x)
    yr = N.ravel(y)
    zr = N.ravel(z)

#    z[15,15] = 0.1

    # knock out an entire chunk of the image

    #$ add random noise
    z = z + N.random.normal(size=N.shape(z))*0.002
    z[15::,0:5] = 0.3

#    A.FindMinimumIn2D(x,y,z, nfine = 100)
    A.FindMinimumTwo(x,y,z, nfine = 1000)

def TestLoopTiming():

    bob = N.random.uniform(size=(16,6,6,50))

    bobshape = N.shape(bob)
    starttime = time.time()
    dum = 0.0

    print bobshape
    for i in range(bobshape[0]):
        for j in range(bobshape[1]):
            for k in range(bobshape[2]):
                for l in range(bobshape[3]):

                    dum = dum + bob[i][j][k][l]

    print time.time()-starttime

def TestCheckPhysicalBracket(Nr = 1.0, rmax = 2.0, npoints = 50):

    """Evaluate the square bracket in the physical-check relation for
    Nr = Nt"""

    A = Plummer()
    A.r3d = N.linspace(start = 0.01, stop = rmax, num = npoints)
    
    A.ModelR0 = 0.4
    A.ModelNr = Nr
    A.ModelNt = Nr
    
    squarebracket = A.ArrayCheckPhysicalSimplified()

    P.clf()
    P.plot(A.r3d, squarebracket)
    P.plot([0.0,rmax],[0.0,0.0])

# read in the interpolated values
def TestInterpCollapse():

    A = Plummer()
    A.Subspace2DInterpRead()
    A.SubspaceCollapseInterp()

# read in the noninterpolated values
def TestInterp(iPlot = 0, Smooth = 2):
    
    """Test interpolation"""

    A = Plummer()
    A.SubspaceReadUnbinned()
    A.PlotNum = iPlot
    A.Subspace2DInterpCalc('M3d', 1500, \
                               InSmooth = Smooth, CutClosest = False, UseClusters=True)
    A.Subspace2DInterpCalc('MProj', 1500, \
                               InSmooth = Smooth, CutClosest = False, UseClusters=True)
    A.Subspace2DInterpWrite(ThisKey = 'M3d')

# test surface density reading
def TestLoadSurfDens():

    """Load in and examine the surface density data of espinoza et al."""

    A = Plummer()
    A.LoadSurfaceDensity(DoPlot = True)
    A.ModelR0 = 0.4
    A.EvalChisqSurfDensArray(DoPlot = True)

# test a run-through on synthetic data
def TestRunThrough(UseData = True, quick=True, ChiLev = 9.72, SkipPop = True, \
                       FixedDens0 = None):

    TimeStarted = time.time()
    # will take a while

    NNr = 36
    NNt = 36
    NR  = 16
    if quick:
        NNr = 10
        NNt = 10
        NR  = 8


    if not SkipPop:
        TestSetupGrid(UseMProj = False, ReadInData = UseData, \
                          npoints=5, rmax = 0.3, alpha = 0.7, beta = 0.6, \
                          DoFig = True, ReducedSet = False, InError = 0.2, \
                          NumR = NR, NumNt = NNt, NumNr = NNr, r0 = 0.45, \
                          FixedDens0 = FixedDens0)

    print "TestRunThrough INFO - time to populate: %.2f sec" % (time.time() - TimeStarted)

    #R0, Nt appears to work well.

    # R0 and s0 are both well-constrained by the data for example
    # choice of Nr, Nt. Minimize over those variables.

    TestFindBubble(ChiLev = ChiLev)

    print "TestRunThrough INFO - time to evaluate region: %.2f sec" % (time.time() - TimeStarted)


    TestShowProfiles(ChiLev = ChiLev, NumRadii = 10, NumRows = 500)

    print "TestRunThrough INFO - time to do profiles: %.2f sec" % (time.time() - TimeStarted)

#    TestMinimizeArr('Nr','Nt',IndRep0 = 3,IndRep1 = 17, \
#                        InFits='RunGridLoopsVec.fits', CutBads = True, iplot=8)

    TimeEnded = time.time()
    print "TestRunThrough DBG - Time elapsed: %.2f seconds" % (TimeEnded - TimeStarted)

def TestFindBubble(ChiLev = 4.7, LoopFile = 'RunGridLoopsVec.fits'):

    """Given a grid of trials, find the chidsq bubble corresponding to
    desired significance"""

    A = Plummer()
    A.GridReadFits(LoopFile)
    A.EstimateChiSurface(ChiLev)

def RunThroughCorrect(self):

    """Rerun everything. Calculate Chisq in one loop through, then populate MProj in the next"""

    TestSetupGrid(UseMProj = False, ReadInData = UseData, \
                      npoints=5, rmax = 0.3, alpha = 0.7, beta = 0.6, \
                      DoFig = True, ReducedSet = False, InError = 0.2, \
                      NumR = 32, NumNt = 48, NumNr = 48, r0 = 0.45, \
                      LoopFile = 'RunGridLoopsVec_big.fits')

    TestFindBubble(ChiLev = 50.0, LoopFile = 'RunGridLoopsVec_big.fits')
    
def TestShowProfiles(ChiLev = 4.7, NumRows = 10, NumRadii = 10):

    A = Plummer()

    # pass options
    A.PlotRadiiN = NumRadii
    A.NumProfileSamples = NumRows

    # read in the trials
    print "TestShowProfiles INFO - reading in trials-file..."
    A.ReadTrialsWithMProj()
    print "TestShowProfiles INFO - ... done."

    # set up output dictionary and plot vector
    A.SetupRadiiForPlots()
    A.SetupDOutForPlot()
    
    # generate random indices within the chisq bubble
    A.DrawRandomsWithinChiLev(ChiLev)
    
    # loop through the evaluations
    print "Evaluating profiles..."
    A.Verbose=False
    A.EvalProfilesLoop()

    print "... Done."
    print N.size(N.where(A.DProfiles['Done'] > 0)[0])

    A.PlotEvalProfiles()

    # now write to disk for plotting
    A.WriteEvalProfiles()

# test plot
def TestReadProfiles():

    """Test profile-plotting"""

    A = Plummer()
    A.DoEPS = True
    A.ReadEvalProfiles()
    A.DataReadCaptured()
    A.PlotEvalProfiles()

# test plot for evaluations
def TestAgainstLRF92():

    """Compute and plot a test profile against Leonard, Fahlman &
    Richer (1992)"""

    A = Plummer()
    A.ProfilePlot()
    A.PlotOneProfile()

def TestAgainstIso():

    """Plot projected mass against its equivalent for plummer
    isotropic case"""

    A = Plummer()
    A.ProfilePlotIsotropic()

def TestShowKing():

    """Show surface density profile against king and plummer profiles"""

    A = Plummer()
    A.LoadSurfaceDensity(DoPlot = True, ShowKing=True)
