#!/usr/bin/env python
##############################################################################
#
# This PPXF_POPULATION_GAS_EXAMPLE_SDSS routine shows how to study stellar
# population with the procedure PPXF, which implements the
# Penalized Pixel-Fitting (pPXF) method by Cappellari M., & Emsellem E.,
# 2004, PASP, 116, 138.
#
# This example shows how to include gas emission lines as templates
# instead of masking them using the GOODPIXELS keyword.
#
# MODIFICATION HISTORY:
#   V1.0: Adapted from PPXF_KINEMATICS_EXAMPLE.
#       Michele Cappellari, Oxford, 12 October 2011
#   V1.1: Made a separate routine for the construction of the templates
#       spectral library. MC, Vicenza, 11 October 2012
#   V1.11: Includes regul_error definition. MC, Oxford, 15 November 2012
#   V2.0: Translated from IDL into Python. MC, Oxford, 6 December 2013
#   V2.01: Fit SDSS rather than SAURON spectrum. MC, Oxford, 11 December 2013
#   V2.1: Includes gas emission as templates instead of masking the spectrum.
#       MC, Oxford, 7 January 2014
#
##############################################################################

import pyfits
import pyspeckit
from scipy import ndimage
import numpy as np
import glob
import matplotlib.pyplot as plt
from time import clock
from os.path import basename, splitext
import csv
from astroquery.irsa_dust import IrsaDust  # needed for the ebv value

from ppxf import ppxf
import ppxf_util as util


def setup_spectral_library(velscale, FWHM_gal):


    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis et al. (2010, MNRAS, 404, 1639) http://miles.iac.es/.
    #
    # For this example I downloaded from the above website a set of
    # model spectra with default linear sampling of 0.9A/pix and default
    # spectral resolution of FWHM=2.51A. I selected a Salpeter IMF
    # (slope 1.30) and a range of population parameters:
    #
    #     [M/H] = [-1.71, -1.31, -0.71, -0.40, 0.00, 0.22]
    #     Age = range(1.0, 17.7828, 26, /LOG)
    #
    # This leads to a set of 156 model spectra with the file names like
    #
    #     Mun1.30Zm0.40T03.9811.fits
    #
    # IMPORTANT: the selected models form a rectangular grid in [M/H]
    # and Age: for each Age the spectra sample the same set of [M/H].
    #
    # We assume below that the model spectra have been placed in the
    # directory "miles_models" under the current directory.
    #
    vazdekis = glob.glob('miles_models/Mun1.30*.fits')
    FWHM_tem = 2.51  # Vazdekis+10 spectra have a resolution FWHM of 2.51A.

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = pyfits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange_temp = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1']-1)])
    sspNew, logLam_temp, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)

    # Create a three dimensional array to store the
    # two dimensional grid of model spectra
    #
    nAges = 26
    nMetal = 6
    templates = np.empty((sspNew.size, nAges, nMetal))

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels

    # Here we make sure the spectra are sorted in both [M/H]
    # and Age along the two axes of the rectangular grid of templates.
    # A simple alphabetical ordering of Vazdekis's naming convention
    # does not sort the files by [M/H], so we do it explicitly below
    #
    metal = ['m1.71', 'm1.31', 'm0.71', 'm0.40', 'p0.00', 'p0.22']
    for k in range(nMetal):
        new_list = [s for s in vazdekis if metal[k] in s]
        for j in range(nAges):
            hdu = pyfits.open(new_list[j])
            ssp = hdu[0].data
            ssp = ndimage.gaussian_filter1d(ssp, sigma)
            sspNew, logLam_temp, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)
            templates[:, j, k] = sspNew  # Templates are *not* normalized here

    return templates, lamRange_temp, logLam_temp

#------------------------------------------------------------------------------

def ppxf_population_gas_sdss(file, z, name):

    # Read SDSS DR8 galaxy spectrum taken from here http://www.sdss3.org/dr8/
    # The spectrum is *already* log rebinned by the SDSS DR8
    # pipeline and log_rebin should not be used in this case.

    hdulist = pyfits.open(file)
    VAC = (10**hdulist[1].data.loglam)
    wave = []
    for i in range(0, len(VAC)):
        wave.append(VAC[i] / (1.0 + 2.735182E-4 + 131.4182 / VAC[i]**2 + 2.76249E8 / VAC[i]**4) / (1+z))
    flux = hdulist[1].data.flux*10**-17
    err = hdulist[1].data.ivar*10**-17
    #bunit = hdulist[0].header['bunit']
    #c0 = hdulist[0].header['coeff0']
    #c1 = hdulist[0].header['coeff1']
    #units = 'erg/s/cm^2/Ang'

    xarr = pyspeckit.units.SpectroscopicAxis(wave, units='angstroms')
    spec = pyspeckit.OpticalSpectrum(header=hdulist[0].header, xarr=xarr, data=flux*1e17, error=err)
    #spec.units = 'erg s^{-1} cm^{-2} \\AA^{-1}'
    #spec.xarr.units='angstroms'

    #Galactic extinction correction
    #Take the ebv of the galaxy from IrsaDust
    table = IrsaDust.get_query_table(name, section='ebv')
    ebv = table['ext SFD mean'][0]

    spec.deredden(ebv=ebv)  # deredden in place
    t = hdulist[1].data
    #z = float(hdu[1].header["Z"]) # SDSS redshift estimate

    # Create the mask
    # Only use the wavelength range in common between galaxy and stellar library.
    mask = [True]*(len(wave))
    for i in range(0, len(wave)):
        #mask[i]=(wave[i] > 3540) & (wave[i] < 7409)
        mask[i] = (wave[i] > 3750) & (wave[i] < 7400)  # take a smaller the wavelength range

    #mask for the galaxy
    galaxy = t.field('flux')/np.median(t.field('flux'))  # Normalize spectrum to avoid numerical issues

    galaxymask = []
    for i in range(0, len(mask)):
        if mask[i]:
            galaxymask.append(galaxy[i])

    galaxy = np.array(galaxymask)

    #mask for the wavelength
    #create an array with only the allowed values of the wavelenght
    wavemask = []
    for i in range(0, len(mask)):
        if mask[i]:
            wavemask.append(wave[i])

    wave = np.array(wavemask)

    #create a mask for the emission lines
    NeIIIa = 3869.9
    NeIIIb = 3971.1
    Heps = 3890.2
    Hdelta = 4102.9
    Hgamma = 4341.7
    OIIIc = 4364.4
    HeIIa = 4687.0
    HeIIb = 5413.0
    SIII = 6313.8
    OIa = 5578.9
    OIb = 6365.5

    Hbeta = 4861.33
    OIIIa = 4958.92
    OIIIb = 5006.84
    OI = 6300.30
    NIIa = 6549.86
    NIIb = 6585.27
    Halpha = 6564.614
    SIIa = 6718.2
    SIIb = 6732.68
    ArIII = 7137.8

    delta = 10
    delta2 = 20
    maskHa = [True]*(len(wave))
    for i in range(0, len(wave)):
        maskHa[i] = (((wave[i] < (Halpha - delta2)) or (wave[i] > (Halpha + delta2))) &
                    ((wave[i] < (Hbeta - delta2)) or (wave[i] > (Hbeta + delta2))) &
                    ((wave[i] < (OIIIa - delta)) or (wave[i] > (OIIIa + delta))) &
                    ((wave[i] < (OIIIb - delta)) or (wave[i] > (OIIIb + delta))) &
                    ((wave[i] < (OI - delta)) or (wave[i] > (OI + delta))) &
                    ((wave[i] < (NIIa - delta)) or (wave[i] > (NIIa + delta))) &
                    ((wave[i] < (NIIb - delta)) or (wave[i] > (NIIb + delta))) &
                    ((wave[i] < (SIIa - delta)) or (wave[i] > (SIIa + delta))) &
                    ((wave[i] < (SIIb - delta)) or (wave[i] > (SIIb + delta))) &
                    ((wave[i] < (NeIIIa - delta)) or (wave[i] > (NeIIIa + delta))) &
                    ((wave[i] < (NeIIIb - delta)) or (wave[i] > (NeIIIb + delta))) &
                    ((wave[i] < (Heps - delta)) or (wave[i] > (Heps + delta))) &
                    ((wave[i] < (Hdelta - delta)) or (wave[i] > (Hdelta + delta))) &
                    ((wave[i] < (Hgamma - delta)) or (wave[i] > (Hgamma + delta))) &
                    ((wave[i] < (OIIIc - delta)) or (wave[i] > (OIIIc + delta))) &
                    ((wave[i] < (HeIIa - delta)) or (wave[i] > (HeIIa + delta))) &
                    ((wave[i] < (HeIIb - delta)) or (wave[i] > (HeIIb + delta))) &
                    ((wave[i] < (SIII - delta)) or (wave[i] > (SIII + delta))) &
                    ((wave[i] < (OIa - delta)) or (wave[i] > (OIa + delta))) &
                    ((wave[i] < (OIb - delta)) or (wave[i] > (OIb + delta))) &
                    ((wave[i] < (ArIII - delta)) or (wave[i] > (ArIII + delta))))

    # mask for the wavelength for the emission lines
    # create an array with only the allowed values of the wavelenght
    wavemask = []
    for i in range(0, len(maskHa)):
        if maskHa[i]:
            wavemask.append(wave[i])

    wave = np.array(wavemask)

    #Use this mask for the galaxy
    galaxymask = []
    for i in range(0, len(maskHa)):
        if maskHa[i]:
            galaxymask.append(galaxy[i])
        
    galaxy = np.array(galaxymask)
    
    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0)
    #
    #
    noise = galaxy*0 + 0.01528           # Assume constant noise per pixel here

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458  # speed of light in km/s
    velscale = np.log(wave[1]/wave[0])*c
    FWHM_gal = 2.76  # SDSS has an instrumental resolution FWHM of 2.76A.

    stars_templates, lamRange_temp, logLam_temp = \
        setup_spectral_library(velscale, FWHM_gal)

    # The stellar templates are reshaped into a 2-dim array with each spectrum
    # as a column, however we save the original array dimensions, which are
    # needed to specify the regularization dimensions
    #
    reg_dim = stars_templates.shape[1:]
    stars_templates = stars_templates.reshape(stars_templates.shape[0], -1)

    # See the pPXF documentation for the keyword REGUL,
    # for an explanation of the following two lines
    #
    stars_templates /= np.median(stars_templates)  # Normalizes stellar templates by a scalar
    regul_err = 0.004  # Desired regularization error

    # Construct a set of Gaussian emission line templates
    #
    gas_templates = util.emission_lines(logLam_temp, FWHM_gal)

    # Combines the stellar and gaseous templates into a single array
    # during the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    #
    templates = np.hstack([stars_templates, gas_templates])

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_KINEMATICS_EXAMPLE_SAURON.
    #
    z = 0  # redshift already corrected
    c = 299792.458
    dv = (np.log(lamRange_temp[0])-np.log(wave[0]))*c  # km/s
    vel = c*z  # Initial estimate of the galaxy velocity in km/s
    
    # Here the actual fit starts. The best fit is plotted on the screen.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    start = [vel, 180.]  # (km/s), starting guess for [V,sigma]

    t = clock()

    plt.clf()
    plt.subplot(211)

    # Assign component=0 to the stellar templates and
    # component=1 to the gas emission lines templates.
    # One can easily assign different components to different gas species
    # e.g. component=1 for the Balmer series, component=2 for the [OIII] doublet, ...)
    # Input a negative MOMENTS value to keep fixed the LOSVD of a component.
    #
    component = [0]*stars_templates.shape[1] + [1]*gas_templates.shape[1]
    moments = [4, 4]  # fit (V,sig,h3,h4) for both the stars and the gas
    start = [start, start]  # adopt the same starting value for both gas and stars

    pp = ppxf(file, templates, wave, galaxy, noise, velscale, start,
              plot=True, moments=moments, degree=-1, mdegree=10,
              vsyst=dv, clean=False, regul=1./regul_err,
              reg_dim=reg_dim, component=component)

    # When the two numbers below are the same, the solution is the smoothest
    # consistent with the observed spectrum.
    #
    print 'Desired Delta Chi^2:', np.sqrt(2*galaxy.size)
    print 'Current Delta Chi^2:', (pp.chi2 - 1)*galaxy.size
    print 'elapsed time in PPXF (s):', clock() - t

    plt.subplot(212)
    #plt.set_cmap('gist_heat') # = IDL's loadct, 3
    plt.imshow(np.rot90(pp.weights[:np.prod(reg_dim)].reshape(reg_dim)/pp.weights.sum()),
               interpolation='nearest', aspect='auto', extent=(np.log(1.0),
               np.log(17.7828), -1.9, 0.45))
    plt.set_cmap('gist_heat')  # = IDL's loadct, 3
    plt.colorbar()
    plt.title("Mass Fraction")
    plt.xlabel("log Age (Gyr)")
    plt.ylabel("[M/H]")
    plt.tight_layout()

    # Save the figure
    name = splitext(basename(file))[0]
    plt.savefig(name)

    return

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#Write the value of sigma on a file (no log yet)
def writesigma(file, pp):
    file_sigma = open("sigma", "w")  # Create a file sigma
    #file_sigma.write('OI/ha erOI/ha NII/ha erNII/ha SIIha erSII/ha \n')
    file_sigma.close()
    file_sigma = open("sigma", "a")
    file_sigma.write(str(file))
    file_sigma.write(str(pp))
    file_sigma.close()
    return

# Create a file sigma
file_sigma = open("sigma", "w")  # Create a file sigma
file_sigma.close()

#read the data file (name, redshift and name for extinction)
names = []
redshift = []
extinctionname = []
with open('data', 'r') as infile:
    csv_reader = csv.reader(infile, delimiter='\t')
    for line in csv_reader:
        names.append(line[0])
        redshift.append(line[1])
        extinctionname.append(line[2])
infile.close()

#read the data file with the information if there is the broad component or not
broad = []
with open('flux3', 'r') as infile:
    csv_reader = csv.reader(infile, delimiter='\t')
    for line in csv_reader:
        broad.append(line[2])
infile.close()

for i in range(0, len(broad)):
    if str(broad[i]) == 'TRUE':
        broad[i] = True
    else:
        broad[i] = False

redshift=map(float, redshift)
nbr = len(redshift)  # nbr of spectrum

for i in range(0, nbr):
    if broad[i] == False:
        print i
        print names[i]
        ppxf_population_gas_sdss(names[i]+'.fits', redshift[i], extinctionname[i])
