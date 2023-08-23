# -*- coding: utf-8 -*-
'''
@author: Elizabeth Watkins

This script "exampling_testing_pipeline.py" tests if the pca pipeline works,
and provides a working example of the pipeline being set-up and used in full,
with some additional options not discussed here being used (these optional
mainly help with the fake data aspect). It also generates some plots to show 
what is going on. 
Preferably, real sky data and science data should be used. For now, this script
uses fake data that is made in the function `make_fake_data` imported from
fakedatalvm.py to test the pipeline.
!!!WARNING if the fake data doesn't have correlated features, gappyrecon.py 
(the pca reconstruction that fills missing data with a reconstructed guess), 
fails with a error with a tensordot. I believe this occurs since random noise 
does not have correlated features, so it couldn't reconstruct and find a 
solution.

'''

from scipy.ndimage import median_filter
import numpy as np
import numpy.random as random

import pcapipeline as ppline #robitgaPCA.pipe as PCApipe
import configpca as config
import fakedatalvm as fake

VARIABLE_NAMES = config.variable_names

#THESE FUNCTIONS ARE EXAMPLES TO TEST THE PIPELINE AND TO PROVIDE AN EXMAPLE OF
#HOW TO SET UP THE PREPROCESSING FUNCTIONS. THEY ARE ROUGHLY IN LINE WITH 
#HOW THE PREPROCESSING FUNCTIONS COULD LOOK LIKE, BUT SPECIFIC VARIABLES FOR
#LVM WILL NEED FIGURING OUT IF THEY ARE USED, 
#SOME MIGHT BE GOOD ENOUGH FOR INTIAL THE PREPROCESSING (EXCEPT FOR THE 
#SCIENCE LINE IDENTIFICATION, WHICH IS JUST A DUMMY FUNCTION)

def skycorr_continuum_removal_median_filter(spectra, filter_size=51, **kwargs):
    """
    Takes in spectra, calculates the continuum using a median filter. Median
    filter was used in previous SDSS continuum removal. 
    !!!If LVM has their own method for characterising continuum that is more
    sophisticated, please use that instead        

    Parameters
    ----------
    spectra : numpy.ndarray
        DESCRIPTION.
    filter_size : int, optional
        The size of the filter. Must be an odd number. The default is 51.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    kwargs_from_pcaPrep = {}
    for var in VARIABLE_NAMES:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
        
    if filter_size % 2 == 0:
        raise ValueError('`filter_size` must be an odd number')
    
    contiuum = median_filter(spectra, filter_size, axes=1)
    
    contiuum_subtracted = spectra - contiuum
    
    return contiuum, contiuum_subtracted


def skycorr_normalise(spectra, noise_gen_data, noise_is_poisson=False, **kwargs):
    """
    Might need observation numbers to assign the right models to the data?

    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.
    noise_gen_data : TYPE
        DESCRIPTION.

    Returns
    -------
    normalised : TYPE
        DESCRIPTION.
    poisson_estimate : TYPE
        DESCRIPTION.
    bool
        DESCRIPTION.

    """
    kwargs_from_pcaPrep = {}
    for var in VARIABLE_NAMES:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
        
    poisson_estimate = np.sqrt(noise_gen_data)
    normalised = spectra/poisson_estimate
    
    return normalised, poisson_estimate, noise_is_poisson

def skycorr_skylines(skyline_mask, **kwargs):
    """
    If each sky model has its own skymask, this function produces
    a sky mask from the union of those masks

    Parameters
    ----------
    skyline_mask : numpy.1darray
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    mask : numpy.1darray
        DESCRIPTION.

    """
    
    kwargs_from_pcaPrep = {}
    for var in VARIABLE_NAMES:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
    
    mask = np.sum(skyline_mask, axis=0)
    mask[mask>0]=1
        
    return mask
    
def skycorr_sciencelines_DUMMY(spectra, skyline_mask=None, error_spectra=None, science_library=None, **kwargs):
    kwargs_from_pcaPrep = {}
    for var in VARIABLE_NAMES:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
    #Some maths to identify skylines while ignoring skylines, or use
    #some science line library
    return np.ones_like(spectra)

def skycorr_outlier_spectra(spectra, skyline_mask=None, science_line_mask=None, error_spectra=None, mean_limit=0.2, std_limit=0.8, **kwargs):
    """
    Outlier identification from Wild & Hewett 2005 - 
    "For the wavelength range used in the analysis (6700--9180 A) the spectra 
    must satisfy the following criteria, where the numerical values are in the
    units of the SDSS spectra (10^17erg s^−1 cm^−2A^−1):
        (i) −0.2 < mean flux < 0.2, ensuring the spectra have a mean
            close to zero.
        (ii) variance of the flux < 0.8, ensuring that the amplitude of
             pixel-to-pixel fluctuations is not unusually high
        (iii) "spectral colours" have values |a − b| < 0.1, |a − c| < 0.3
              and |b−c| < 0.3 ensuring that the spectra do not exhibit 
              significant large scale gradients. a, b and c are calculated by 
              averaging the flux in three wavelength regions largely free of OH
              sky emission (7000:7200 \AA [a], 8100:8250 \AA [b] and 9100:9180 \AA [c]). ˚
    We further only accept spectra with a minimum of 3800 good pixels ("NGOOD" 
    parameter) over the entire wavelength range of the spectrum, to eliminate
    spectra with substantial numbers of missing pixels."
    
    I've implemented in this dummy function using criteria i, and ii

    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.
    skyline_mask : TYPE, optional
        DESCRIPTION. The default is None.
    science_line_mask : TYPE, optional
        DESCRIPTION. The default is None.
    errors : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    kwargs_from_pcaPrep = {}
    for var in VARIABLE_NAMES:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
        
    spectra_to_use = np.ones(np.shape(spectra)[0])
    
    #Should just be ones and zeros. For skylines, 1 indicates there
    #is a skyline there. For sciencelines, 1 indicates there is NO scienceline
    #(i.e., the data can be used as there is no science line there)
    
    spectra_sky_science_ignore = np.copy(spectra)
    
    spectra_sky_science_ignore[:, skyline_mask==0] = np.nan
    spectra_sky_science_ignore[science_line_mask==0] = np.nan
    
    spectra_mean = np.nanmean(spectra_sky_science_ignore, axis=1)
    spectra_std = np.nanstd(spectra_sky_science_ignore, axis=1)
    spectra_to_use[np.abs(spectra_mean)>mean_limit] = 0
    spectra_to_use[spectra_std>std_limit] = 0
    
    return spectra_to_use

PLOTTING=True

random.seed = 1
### Making fake data to test the pipeline runs as expected ====================
num_spec = 6000
num_pixels = 820
sky_spectra, skyline_mask_1d, fake_poisson_noise, error_spectra = fake.make_fake_data(num_spec, num_pixels, noise_std=0.18, subtract=True)
wavelength_lower = 6700 # AA
wavelength_upper = 9180 # AA
wavelength = np.linspace(wavelength_lower, wavelength_upper, num_pixels)
wavelength1 = np.linspace(wavelength_lower, wavelength_upper, num_pixels)
### ===========================================================================
if PLOTTING:
    import matplotlib.pyplot as plt
    from plotslvm import DiagnosticPlots as plots
    plt.close('all')
    pl = plots(sky_spectra, wavelength1, skyline_mask_1d)#, spectrum_number=0, fake=True, perc=1) #<defaults
    
sky_model_for_skys = fake_poisson_noise 
                  
preprocessing_methods = {
    'continuum'     : skycorr_continuum_removal_median_filter,
    'normalise'     : skycorr_normalise,
    'skylines'      : skycorr_skylines,
    'science_lines' : skycorr_sciencelines_DUMMY,
    'outliers'      : skycorr_outlier_spectra
}

cont_kwargs = {
    'filter_size':51 # filter size will need adjusting. 
}

outliers_kwargs = {
    'mean_limit' : 0.3,
    'std_limit'  : 0.9
    }

#For this example, only want kwargs for the continuum subtraction.
method_kwargs = {
    'continuum'     : cont_kwargs,
    # 'normalise'     : norm_kwargs,
    # 'skylines'      : skyline_kwargs,
    # 'science_lines' : scienceline_kwargs,
    # 'outliers'      : outliers_kwargs
}

#Generating the pca library, The preprocessing needs to be done first, so
#we are initialising the class using a class method that performs the pcaPrep
# for us
lib = ppline.pcaLibrary.run_pcaPrep(
    sky_spectra, 
    noise_gen_data=sky_model_for_skys, 
    subtraction_methods=preprocessing_methods, 
    method_kwargs=method_kwargs,
    wavelength=None, 
    wavelength_range=None,
    skyline_mask=skyline_mask_1d,
    error_spectra=error_spectra,
    save_extra_param=True, #when running the PCA this outputs the tracked eigen system. You do not need to include it unless checking stuff
    number_of_iterations=2 #default is 5. Set to 2 for speed. 3 minimum is recommended (repeats the pca process for a different random stream of spectra to confirm the pca has converged)
) 
#this autoruns and the pca library is output as the attribute `output` 

if PLOTTING:
    processed_sky_spectra = lib.spectra
    pl.view_spectrum(processed_sky_spectra, 'before', processed=True)
    pl.view_all_spectra(processed_sky_spectra, 'before', processed=True)
    pl.view_Poisson(fake_poisson_noise)

        
# If we didnt include `save_extra_param=True`, the output would just be:
# `pca_library = lib.output`
pca_library, tracked_pca_param = lib.output # can save (np.save) and load it (np.load) later


##
#If you run out of memory, either go to a beefier PC or you can reduce the
# number of science spectra handed to it in one go. 
#-
#If the the number of components needed for most are <10, Ive found it can run
# over 4000 vectorised at the same time, if not more. at least
##

#Fake science data
num_spec_sci = 500
science_spectra, __, fake_poisson_noise, error_spectra_for_science = fake.make_fake_data(num_spec_sci, num_pixels, noise_std=0.2, subtract=True)
#To check stuff is working, I've replaced the first 'science' spectra with
# a 'sky' spectra
science_spectra[0] = sky_spectra[0]
#
# sky_model_for_science = fake_poisson_noise

sub = ppline.pcaSubtraction.run_pcaPrep(
    science_spectra,
    noise_gen_data=sky_model_for_skys, #sky_model_for_science < using `sky_model_for_skys` since more fake spectra were generated so the Poisson noise estimate will be more accurate for this test
    subtraction_methods=preprocessing_methods, 
    method_kwargs=method_kwargs,
    wavelength=None, 
    wavelength_range=None,
    skyline_mask=skyline_mask_1d,
    error_spectra=error_spectra_for_science,
    pca_library = pca_library, 
)
""" #========================================================================
Because this is fake data, the pca method is not optimal and so we should
not need too many, and also the rms calculation for the fake data 
probably doesn't do a good job for the stopping criteria. Use of too many
components makes the PCA act as a low pass (high-frequency) filter, and I've 
seen this occur when testing with fake data. When you run on real data, you
should use more than 100 (previous sdss used up to 300) since there is more 
information and variances within the data!!!
Since rms is not going to be a good check of the stopping criteria 
`replace_max_solution` has been set to False since rms in many fake spectra
does not improve and results in returning a lot of the original spectra.
""" #========================================================================
max_recon_number = 100
residual_removed, rms_changes, n_comp_used = sub.run(projection_method='best', #default is 'best'
                                                     return_extra_info=True, #default is False
                                                     undo_preprocessing=False, #default is False
                                                     max_comp=max_recon_number, #default=300
                                                     replace_max_solution=True, #default True
                                                     rms_leeway=0.02 # default = 0.00 # Helps since fake data
                                                     )

res_remo_pre_proc_remo = sub.undo_preprocessing(residual_removed)
wavelength1 = np.linspace(wavelength_lower, wavelength_upper, num_pixels)

if PLOTTING:
    pl.view_number_comps_hist(n_comp_used)
    pl.view_spectrum(residual_removed, which='after', processed=True)
    pl.view_spectrum(res_remo_pre_proc_remo, which='after', processed=False)
    
    pl.view_all_spectra(residual_removed, which='after', processed=True)
    pl.view_all_spectra(res_remo_pre_proc_remo, which='after', processed=False)
    
    #diagnostic plots =====================
    # pl.plot_comp_plots(sub, pca_library, tracked_pca_param, max_comp=5, skyline_mask_1d=skyline_mask_1d)
    
