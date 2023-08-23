# -*- coding: utf-8 -*-
"""
@author: Elizabeth Watkins

This script is a pipeline to perform PCA using a given PCA method on subtracted 
background data to form a PCA library and is also a wrapper to apply the PCA
library to similar data to remove subtraction residuals. 

Since it is expected to be run on optical spectra to remove sky subtraction
residuals from science data, the naming conventions of variables used in
this pipeline reflect this.

Code requires that numpy, scipy have been installed.
Requires the robust pca and gappy reconstruction scripts.

Copyright (c) 2023, Elizabeth Jayne Watkins

Permission to use, copy, modify, and/or distribute this software for any purpose 
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies. 

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR 
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE. 
"""

#Need a function/set of functions that stacks all the spectra into 2d array. This will involve loading in spectra and then stacking them together. Will need path management. This should then be saved as a file

#Need a function that takes in skycorr results and outputs a total skyline mask


#dictionary for skycorr functions.

import warnings
import robustpca as pca
import gappyrecon as gappy
import numpy as np
import copy
import utilitymasking as util

import configpca as config

def _pop_pca_kwargs(kwargs):
    """Separates out kwargs for the method and kwargs for the PCA

    Parameters
    ----------
    kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    kwargs : TYPE
        DESCRIPTION.
    separate_kwargs : TYPE
        DESCRIPTION.

    """

    separate_kwargs = {}
    for key in config.pca_kwargs.keys():
        separate_kwargs[key] = kwargs.pop(key, config.pca_kwargs[key])

    return kwargs, separate_kwargs
# (377,)
# (377, 100)
    # print(mean_prev.shape)
    # print(eigen_vectors.shape)
# ================= Default functions ======================
def project_spectra(spectra, errors, eigen_vectors, mean_spectra, 
                    namplitudes=5, run_mask=False, return_extra=False):
    """Default function to reconstruct a spectra using the `gappy` method.
    This routine nicely deals with 'missing data' (such as if there is 
    a science line present at a skyline location, and so part of the skyline
    has been masked out where the science line is, or if some pixels read
    out badly). It does this by filling the gap with a reconstruction of the
    data at this location by solving the data normalisation. This has the
    advantage of keeping the pca euclidean space exact, which minimises
    some artefacts, but comes at the cost that the reconstructions are not
    fully orthogonal any more. It is recommended to ALWAYS reconstruct using
    this function, unless an alternative reconstruction method is used
    that also implements smart handling of data gaps.

    Parameters
    ----------
    spectra : numpy.ndarray
        The spectra that will be used to make the reconstruction
    errors : numpy.ndarray
        Errors. Just needs to be 0's where to ignore the data and 1's 
        everywhere else.
    eigen_vectors : numpy.ndarray
        The sky library eigen vectors.
    mean_spectra : numpy.1darray
        The mean spectra.
    namplitudes : int, optional
        The number of components used to reconstruct. The default is 5.
    get_run_mask : bool, optional
        Whether to apply the mask where reconstruction performed on. The 
        default is False.
    return_extra : bool, optional
        If `return_extra` is True extra parameter `used_spectra` is returned
        or the extra parameter is None if `get_run_mask` is None. The default
        is False.        

    Returns
    -------
    reconstructed_spectra : numpy.ndarray
        The reconstructed spectra.

    """
    if namplitudes == 0:
        return spectra

    e1_to_construct = eigen_vectors[:,:namplitudes]

    spectra, errors = [util.check_make_2d(ar) for ar in [spectra, errors]]
    
    if run_mask:
        final_spectra = np.zeros_like(spectra)
        # gappy.run_normgappy is vectorised. array[None], just adds a dummy dimension
        # so that we can run the code (array.shape = [N] array[None].shape = [1,N])
        scores, norm, used_spectra = gappy.run_normgappy(errors, spectra, mean_spectra, e1_to_construct, return_run_mask=run_mask)
        reconstructed_spectra  = (scores @  e1_to_construct.T) + mean_spectra
        final_spectra[used_spectra] = reconstructed_spectra
        final_spectra[~used_spectra] = spectra[~used_spectra]    
        
    else:
        scores, norm = gappy.run_normgappy(errors, spectra, mean_spectra, e1_to_construct)
        final_spectra  = (scores @  e1_to_construct.T) + mean_spectra
        
    if return_extra:
        if run_mask:
            return final_spectra, used_spectra 
        else:
            return final_spectra, None 
    else:
        return final_spectra

def get_rms(spectra, rms_mask, where_data_good_and_errors, rms_percentile=90):
    """Default function to find the rms of the data robustly by using a 
    percentile value rather than the raw mean (i.e., if `rms_percentile`=50
    it would be the root median squared). Default value is 90th percentile
    of the rms values to minimise outliers, as described in Wild et al.

    Parameters
    ----------
    spectra : numpy.ndarray
        The spectra.
    rms_mask : numpy.ndarray
        Which pixels from the spectra are being used for the the rms. Not the 
        same as where the data is good or not. Only part of the data 
        for the rms (i.e., on or sky) is needed
    where_data_good_and_errors : numpy.ndarray
        0's where the data is bad. The rest contain the error values.
    rms_percentile : float or int, optional
        The percentile above which the rms is calculated from. The default is
        90.

    Returns
    -------
    rms_per_specta : numpy.ndarray
        The rms value for each spectra.

    """
    #Ignore outliers indicated in the error spectra that might not
    #have been indicated yet
    # where_data_good[error_spectra==0] = 0

    data_over_error = (spectra / where_data_good_and_errors)

    where_data_good = copy.deepcopy(where_data_good_and_errors)
    where_data_good[where_data_good !=0] = 1
    where_data_good = where_data_good.astype(bool)
    rms_mask = rms_mask.astype(bool)

    data_over_error[~where_data_good] = np.nan
    data_over_error[~rms_mask] = np.nan
    #check
    rms_per_specta = np.nanpercentile(data_over_error, rms_percentile, axis=1)

    return rms_per_specta


#!!!NOTES for doc
#we used the median Poisson noise of the skypixels in older sdss because the
#sky was next to the object, and represented the Poisson noise that the sky
#gave independent of the science. We do not have this is LVM. We have
#a sky that is far away that is used to model the sky at the science

#The issue is that the Poisson noise from the sky telescopes are going to be very different
#to the science.
#With skycorr, and the fact it gives us a model sky in the science
# it has separated out the sky noise from the object noise. And so
#Vivian used the skycorr model as a proxy for the Poisson noise.
#So when making the sky library I can normalise by the error arrays but this
#would not keep things consistent, so I'll also need to normalise by the final
#skycorr model of the sky spectra
#Need to ask what is kept in the skymodel. It should be all the
#sky emission that needs to be subtracted

class pcaPrep():
    """Description: This class is a wrapper for the preparation needed to build
    a PCA library and/or before applying a PCA library to identify and remove
    correlated noise/outlier features present throughout different instances
    of an observation set.
    
    ...
        
    Attributes
    ----------
    
    prep_functions : dict
        Dictionary containing the preprocessing functions with the following
        keys:
            `'continuum'`
            `'normalise'`
            `'skylines'`
            `'science_lines'`
            `'outliers'`
    sky_spectra : numpy.ndarray
        The final calibrated, sky subtracted spectra that contain sky
        subtraction residuals.
    noise_gen_data : numpy.ndarray
        Data/spectra that will be used to (estimate/)get the Poisson
        noise and normalise the spectra with.
    observation_numbers : numpy.1darray
        Iterable ID numbers of the spectra to preserve knowledge of
        where they came from. Some analysis/prep requires keeping track
        of the observation set etc.
    skyline_mask : numpy.1darray, optional
        If known, the position of skylines to perform the PCA analysis
        upon. This 1d array applies to all spectra The default is None.
    error_spectra : numpy.ndarray, optional
        If the phyiscal error arrays of the spectra are different to the 
        array needed when normalising the spectra by its Poisson noise, add
        the noise array using this variable (needed for skycorr). The 
        default is None.
    noise_gen_is_error : bool, optional
        If the error error is the same as the data needed to normalise
        the spectra using the Poisson noise, set this to True, else 
        leave as default or as False. The default is False.
    verbose : bool, optional
        If True, prints out updates as the class runs its methods. 
        The default is True.
        TODO: ADD print statements logs, and diagnostic plots
    processed_spectra : dict
        Dictionary stores the result of each preprocessing step
    method_kwargs : dict or None, optional
        If additional variables are needed when running the preprocessing,
        they can be provided as a dictionary containing these
        variables as a dictionary. For example, if `get_science_lines` and
        'get_outliers' needs additional variables (e.g. an array containing
        all known optical lines to help with science line identification,
        for get_science_lines, or to use non-default option in the
        outlier method provided, such as sigma=5, MAD=True)
        provide it as:
            method_kwargs = {
                'science_lines' : {
                    'all_optical':`YOUR_ARRAY`
                },
                'outliers' : {
                    'sigma' : 5
                    'MAD'   : True
                }
            }
        The default is None.
    _start : str
        The default name of the default starting method for the preprocessing.
        The default is 'continuum'
    _end : str
        The default name of the default final method of the preprocessing.
        The default is 'outliers'
        
    Methods
    -------
    get_continuum(full=False, **kwargs)
        Performs continuum subtraction on the spectra. If full is `True`,
        the continuum found before the subtraction is also returned.
    get_normalise(self, full=False, **kwargs)
        Normalises the spectra. If full is `True`, the normalisation used is
        also returned.
    get_skylines(, **kwargs)
        Gets the skyline mask as a 1d array. This is a mask for the data that
        one wants to USE, so 1's/True are set at skyline locations, and 0's/
        Falses at positions that do not contain skylines
    get_science_lines(mask=None, **kwargs)
        Method for finding and masking science lines as a 2d array. To ensure
        that skylines, or other known common features throughout the spectra
        are not accidentally identified as science lines, either a mask at these 
        positions needs to be provided or `get_skylines` needs to be run
        beforehand and be saved within the attribute `processed_spectra`. Since
        this mask identifies pixels we DO NOT want the PCA to see, 
        science line locations are set to 0 (False) and the remaining usable
        positions are set to 1 (True).
    get_outliers(mask=None, science_line_masks=None, **kwargs)
        Identifies where an entire spectrum is statistically an outlier
        and masks the entire spectrum. This can be either a 1d array
        at the positions to ignore, or can be a 2d array where the pixels
        for the bad spectrum are all masked. To ensure that skylines, or other
        known common features throughout the spectra do not impact the spectra 
        rejection criteria, these locations need to be provided, or need to 
        have already been generated by the object before running this method.
        !!!WARNING:  Individual outlier pixels are not meant to be masked since these
        are part of the noise statistics that the PCA needs to make the eigen
        spectra. What we need to mask are spectra that overall look bad and need
        to be excluded from further analysis.
    run(start=None, end=None, method_kwargs=None)
        Convenience function to run the default class methods while
        defining the start and stop of the methods to run. These methods 
        are (in order):
            `'continuum'`
            `'normalise'`
            `'skylines'`
            `'science_lines'`
            `'outliers'`
    autorun(autorun):
        If autorun has been set at class initialisation, class
        will automatically run the analysis. If the start and/or end
        have been provided when initialising, these will be used, else
        it automatically runs all the prep methods in order. These are:
            `continuum`
            `normalise`
            `skylines`
            `science_lines`
            `outliers`
                 
    Purpose of preprocessing
    ------------------------
    To generate or apply a PCA library, any source/science features, including
    emission, absorption, continuum occurring within the data have to be ignored
    otherwise they will be identified by the PCA as a feature and will result 
    in the loss of science flux during the reconstruction and subtraction
    process. If the noise is proportional to the signal, this also has to be
    accounted for by normalising this property away. Otherwise, places with more
    flux, and therefore higher noise will be more strongly weighted by the PCA,
    resulting in a biased reconstruction that priorities these features.
    Therefore in this case, the data needs to be normalised by the noise to
    negate this affect. For optical IFU data, the Poisson noise needs to be 
    removed before the PCA library, and the PCA reconstruction is run.
    
    For LVM the preprocessing steps are therefore (in order):

        1. Continuum removal              
        2. Normalisation by the Poisson noise  
        3. Skyline identification          
        4. Science line identification      
        5. Outlier identification
    
    
    There are many ways to do the preprocessing steps, therefore this class
    only provides the functionality to run these steps in a predefined way and 
    check which prerequisite steps have been performed. The actual functions to
    perform this preprocessing need to be provided by the user for their
    specific data. This allows the pipeline to be used for different 
    techniques, including previous SDSS implementations of the PCA routine and
    easily allows different methods to be tested and used. 
    
    
    Information for LVM
    ---------------------
    For LVM, untested functions for some of the preprocessing using skycorr 
    have been written and are set as the default if no other preprocessing
    functions are provided. Skycorr is very different to what was done before, 
    and causes there to be no way to calculate the Poisson noise since a MODEL
    is subtracted from the data, NOT an average sky spectrum. Since Poisson
    noise is proportional to the square root of the photon counts, we can try 
    and estimate the Poisson noise added to the spectra by the sky flux by 
    assuming the skycorr model is a good representation of the flux; therefore 
    we can use the square root of the skycorr model that has been subtracted as
    an estimate of the Poisson noise.
    
    Final regards
    -------------
    Typically, this class will only be run by itself for testing. The analysis
    classes run this class themselves so the user should not need to touch this
    class. If the user does want to run it separately and then input it into the
    analysis class, please limit the wavelength range to the part of the 
    spectrum that needs sky residuals removed. This can be done using the
    convenience decorator function `mask_arrays_columns` as so:
        #pp is the pcaPrep object
    pp = mask_arrays_columns(wavelength_mask)(pcaPrep)(
        sky_spectra=sky_spectra, noise_gen_data=noise_gen_data, 
        observation_numbers=observation_numbers, 
        subtraction_methods=subtraction_methods, **kwargs
        )

        
    USAGE SCRAP NOTES
    -----------------
    `observation_numbers` as long as it is iterable, should be
    what allows you to keep track of which observation the spectra
    came from. Some reductions use the median Poisson noise per
    observation to normalise the sky AND and science. Use a convention
    that allows them to preserve this information for both
    the sky and science reductions.
    Needs to be iterable since later reduction steps remove bad spectra
    using their indices.
    TODO: CHECK NAN HANDLING WORKS
    """
    def __init__(self, sky_spectra, noise_gen_data, subtraction_methods, \
                 observation_numbers=None, skyline_mask=None, error_spectra=None, \
                 autorun=False, noise_gen_is_error=False, verbose=True, \
                 method_kwargs=None):
        """     
        Parameters
        ----------
        sky_spectra : numpy.ndarray
            The final calibrated, sky subtracted spectra that contain sky
            subtraction residuals.
        noise_gen_data : numpy.ndarray
            Data/spectra that will be used to (estimate/)get the Poisson
            noise and normalise the spectra with.
        observation_numbers : numpy.1darray
            Iterable ID numbers of the spectra to preserve knowledge of
            where they came from. Some analysis/prep requires keeping track
            of the observation set etc.
        subtraction_methods : dict
            What sky subtraction methods to use, which are wrapped into
            the pipeline.           
        skyline_mask : numpy.ndarray, optional
            If known, the position of skylines to perform the PCA analysis
            upon. The default is None.
        error_spectra : numpy.ndarray, optional
            If the physical error arrays of the spectra are different to the 
            array needed when normalising the spectra by its Poisson noise, add
            the noise array using this variable (needed for skycorr). The 
            default is None.
        autorun : bool or 2-list of strings, optional
            Whether to automatically run the nominal version of this pipeline.
            If True, assumes the data needs all the prep performed (continuum
            removal, Poisson noise normalisation, skyline identification, 
            science line masking and outlier removal in that order). If the
            start or end point of the prep pipeline can be skipped, the user
            defined starting and end point can be provided and the pipeline
            will run from these steps in order. The order and method names are:
                `'continuum'`
                `'normalise'`
                `'skylines'`
                `'science_lines'`
                `'outliers'`
            To run for example, from normalise to the end, you can provide 
            either just provide the string `'normalise'`, or a list
            like: `['normalise', None]` or `['normalise', 'outliers']`
            or `['normalise', '']` or `['normalise', True]`
            To limit the just end point to science line identification
            (`'science_lines'`),provide the list: 
            `[None, 'science_lines']` or `['', 'science_lines']` or 
            `[True, 'science_lines']`. If a different start and end want to
            be automatically  run, provide their names are strings in a 2-list,
            for example `['normalise', science_lines]`. If `False` is provided 
            within the 2-list, and the other option is a method, only that
            single method is run. This can also be achieved by inputting the 
            method name twice such as ['outliers', 'outliers'] to run just the
            outlier method. This functionality is best access using the
            `get_'method' methods (i.e., to run just the skyline 
            identification) initialised the class with no autorun and call the 
            method `this_object.get_skylines()`. The default is False.
        noise_gen_is_error : bool, optional
            if the error error is the same as the data needed to normalise
            the spectra using the Poisson noise, set this to True, else 
            leave as default or as False. The default is False.
        verbose : bool, optional
            If True, prints out updates as the class runs its methods. 
            The default is True.
            TODO: Currently few print statements or logs, or diagnostic
            plot have been coded.
        method_kwargs : dict or None, optional
            If additional variables are needed when running the preprocessing,
            they can be provided as a dictionary containing these
            variables as a dictionary. For example, if `get_science_lines` and
            'get_outliers' needs additional variables (e.g. an array containing
            all known optical lines to help with science line identification,
            for get_science_lines, or to use non-default option in the
            outlier method provided, such as sigma=5, MAD=True)
            provide it as:
                method_kwargs = {
                    'science_lines' : {
                        'all_optical':`YOUR_ARRAY`
                    },
                    'outliers' : {
                        'sigma' : 5
                        'MAD'   : True
                    }
                }
            The default is None.

        Returns
        -------
        None.

        """
        #noise_gen_data usually just error_spectra, but some reductions (i.e.,
        #skycorr uses a different data set for generating the noise so 
        #they are defined as two different variables.
        sky_spectra, error_spectra = util.nan_inf_masking(sky_spectra, error_spectra)
        self.prep_functions = subtraction_methods
        self.sky_spectra = sky_spectra
        self.noise_gen_data = noise_gen_data
        self.observation_numbers = observation_numbers
        self.skyline_mask = skyline_mask
        self.error_spectra = error_spectra
        self.noise_gen_is_error = noise_gen_is_error
        self.verbose = verbose
        self.method_kwargs = method_kwargs
        
        

        self.processed_spectra = {}
        
        self._start = 'continuum'
        self._end = 'outliers'

        self.autorun(autorun)

    def get_continuum(self, full=False, **kwargs):
        """Method runs the continuum function that has been provided. Inputs
        multiple arrays to accommodate any potential continuum removal
        methods. Median filter just requires the data and filter size

        Parameters
        ----------
        full : bool, optional
            If False, only the continuum subtracted data is returned. If True,
            both the continuum subtracted and the found continuum are returned.
            The default is False.    
        kwargs :
            Any additional variables needed for the provided continuum 
            subtraction function.
            
        Returns
        -------
        contiunuum_subtracted_spectra : numpy.ndarray
            Data that has been continuum subtracted.
            
        Optional
        --------
        continuum :  numpy.ndarray
            The continuum found is returned if `full` is True

        """
        contiunuum_func = self.prep_functions['continuum']
        continuum, contiunuum_subtracted_spectra = contiunuum_func(
            self.sky_spectra,
            noise_gen_data=self.noise_gen_data, 
            error_spectra=self.error_spectra, 
            observation_numbers=self.observation_numbers,
            **kwargs
        )

        if full:
            return contiunuum_subtracted_spectra, continuum
        else:
            return contiunuum_subtracted_spectra

    def get_normalise(self, full=False, **kwargs):
        """Method runs the normalisation if need using the function that has
        been provided.

        Parameters
        ----------
        full : bool, optional
            If False, only the normalised data is returned. If True,
            both the normalised data, and the data used to normalise are 
            returned. The default is False.
        kwargs :
            Any additional variables needed for the provided normalisation 
            function.

        Returns
        -------
        normed_sky_spectra : numpy.ndarray
            Data that has been normalised.
            
        Optional
        --------
        poisson_noise : numpy.ndarray
            The normalisation used is returned if `full` is True

        """
        norm_func = self.prep_functions['normalise']
        #uses the most recent data product (so if data needed to be continuum
        #subtracted, that data is pulled into this method)
        spectra = self._get_previous_reduced_data('normalise')

        output = norm_func(
            spectra=spectra, 
            noise_gen_data=self.noise_gen_data, 
            error_spectra=self.error_spectra, 
            observation_numbers=self.observation_numbers,
            **kwargs)

        #want the option to output if the data used to generate to the norm
        #is the errors or not. This is a messy way of checking for this later
        #since this allows us to reuse this class to prep the science data
        #but has the prerequisite of needing the error spectra whereas
        #the library generation does not after the sky has been normalised
        #here
        if isinstance(output, tuple):
            if len(output)==3:
                normed_sky_spectra, poisson_noise, self.noise_gen_is_error = output
            else:
                normed_sky_spectra, poisson_noise = output

        if full:
            return normed_sky_spectra, poisson_noise
        else:
            return normed_sky_spectra

    def get_skylines(self, **kwargs):
        """Gets the skyline mask using the method provided. 
        
        NOTE: Skycorr automatically creates a skyline mask that we can use.
        If this is the case, basically just provide a dummy function or the
        skyline mask when initialising the class under the attribute
        `skyline_mask`.
        If master skys are being used, rather than using skycorr, then skylines 
        need to be found using an algorithm/function.
        
        Parameters
        ----------
        kwargs :
            Any additional variables needed for the provided skyline 
            identification function.

        Returns
        -------
        mask : numpy.1darray
            1d bitmap mask (functionality also allows bool maps I believe)  
            where 1's (True) identify where there is a skyline, 0's (False)
            are not a skyline. Since we only run PCA where we expect skylines
            the masks is 1's/True at these locations

        """
        if self.skyline_mask is not None:
            return self.skyline_mask

        else:
            skyline_func = self.prep_functions['skylines']
            spectra = self._get_previous_reduced_data('skylines')

            mask = skyline_func(spectra=spectra,
                                noise_gen_data=self.noise_gen_data, 
                                error_spectra=self.error_spectra, 
                                observation_numbers=self.observation_numbers,
                                skyline_mask=self.skyline_mask, 
                                **kwargs)

            return mask # we want skylines so the mask here is 1's/True for skyline, 0/False otherwise

    def get_science_lines(self, mask=None, **kwargs):
        """Method for finding and masking science lines 

        Parameters
        ----------
        mask : numpy.1darray, optional
            Mask of data that we know already are not science lines (such as
            skylines) that we want to ignore when finding science lines. 1's
            (True) indicate there is a skyline there, 0's (False) indicate
            it does not contain a skyline
            The default is None.
        kwargs :
            Any additional variables needed for the provided science line 
            identification function.

        Returns
        -------
        science_line_masks : numpy.ndarray
            Bitmap mask (functionality also allows bool maps I believe) 
            where 1's (True) identify data we want to use, 0's (False) are
            science lines we need to ignore.

        """
        if mask is None:
            # if 'skylines' not in self.processed_spectra:
            #     mask = 
            mask = self._run_prerequisite_method('skylines')

        spectra = self._get_spectra(config.prep_names['normalise'])#self._get_spectra(prep_names['continuum'])

        science_lines_func = self.prep_functions['science_lines']
        science_line_masks = science_lines_func(spectra=spectra, 
                                                skyline_mask=mask, 
                                                error_spectra=self.error_spectra,
                                                observation_numbers=self.observation_numbers,
                                                **kwargs)

        # zeros/False indicate we don't want to use this data, so we exclude zeros/false and
        #keep ones/True 
        return science_line_masks

    def get_outliers(self, mask=None, science_line_masks=None, **kwargs):
        """Identifies where an entire spectrum is statistically an outlier
        and masks the entire spectrum. 
        NOTE: We do not mask individual outliers since these are part
        of the noise statistics that the PCA needs to make the Eigen-spectra.
        What we need to mask are spectra that overall look bad and need to
        be excluded from further analysis.

        Parameters
        ----------
        mask : numpy.ndarray, optional
            Mask of data that we know already to ignore (such as
            skylines) when finding outlier spectra. 1's (True) are skyline
            locations. The default is None.
        science_line_masks : numpy.ndarray, optional
            Mask of data that we know already are science lines (such as
            skylines) that we want to ignore when finding outlier spectra. 
            The default is None.
        kwargs :
            Any additional variables needed for the provided outlier 
            identification function.

        Returns
        -------
        outlier_spectra_mask : numpy.ndarray
            1d (can be 2d if the entire spectra has been masked)
            indicating entire spectra to ignore (0's/False)
            and which spectra to use (1's/True). 

        """
        if mask is None:
            # if 'skylines' not in self.processed_spectra:
            mask = self._run_prerequisite_method('skylines')

        if science_line_masks is None:
            # if 'science_lines' not in self.processed_spectra:
            science_line_masks = self._run_prerequisite_method('science_lines')
        
        spectra = self._get_spectra(config.prep_names['normalise'])#self._get_spectra(prep_names['continuum'])

        outlier_func = self.prep_functions['outliers']

        # zeros/False indicate we don't want to use this data, so we exclude
        # zeros/false an keep ones/True
        outlier_spectra_mask = outlier_func(spectra=spectra, 
                                            skyline_mask=mask, 
                                            science_line_mask=science_line_masks,
                                            error_spectra=self.error_spectra, 
                                            observation_numbers=self.observation_numbers,
                                            **kwargs)

        # This is a 1d array for the rows we want to exclude from analysis
        # (zeros/False) and include (ones). But with how I've set the
        #future functions up, this can be a 2darray?
        return outlier_spectra_mask

    def run(self, start=None, end=None, method_kwargs=None):
        """Convenience function to run the default class methods while
        defining the start and stop of the methods to run. These methods 
        are (in order):
            `'continuum'`
            `'normalise'`
            `'skylines'`
            `'science_lines'`
            `'outliers'`

        Parameters
        ----------
        start : string, optional
            Method to begin running. The default is None, which means
            self._start will be called which is currently set to `'continuum'`.
        end : string, optional
            Method to run until. The default is None, which means
            self._end will be called which is currently set to `'outliers'`.
        method_kwargs : dict or None
            If additional variables are needed when running the preprocessing,
            they can be provided as a dictionary containing these
            variables as a dictionary. For example, if `get_science_lines` and
            'get_outliers' needs additional variables (e.g. an array containing
            all known optical lines to help with science line identification,
            for get_science_lines, or to use non-default option in the
            outlier method provided, such as sigma=5, MAD=True)
            provide it as:
                method_kwargs = {
                    'science_lines' : {
                        'all_optical':`YOUR_ARRAY`
                    },
                    'outliers' : {
                        'sigma' : 5
                        'MAD'   : True
                    }
                }

        Returns
        -------
        None.

        """
        if method_kwargs is None:
            if self.method_kwargs is None:
                method_kwargs = {}
            else:
                method_kwargs = self.method_kwargs
        #adds flexibility if slightly different formats are entered
        start, end = self._run(start, end)
        
        #This makes sure that the method strings entered exist. If they do not
        #a key error is thrown here and is easy to identify
        start_int = config.prep_names[start]
        end_int   = config.prep_names[end]
        
        for i in range(start_int, end_int+1):
            method = config.prep_steps[i] #dict
            kwargs = method_kwargs.pop(method, {})
            # kwargs = {} if kwargs is None else kwargs
            self.processed_spectra[method] = getattr(self, 'get_' + method)(**kwargs) #(*(label, pointings))
    
    def _run(self, start, end):
        """Takes in various inputs of which method to run and outputs their
        string equivalent. `None`, `''`, `True` all  result in the default
        string being used. For `start`, this is `self._start`, which should be
        `'continuum'`, for `end`, this should be `'outliers'`
        If `False` is provided for start or end
        within the 2-list, and the other option is a method, only that
        single method is run. This can also be achieved by inputting the 
        method name twice such as ['outliers', 'outliers'] to run just the
        outlier method. This functionality is best access using the
        `get_'method' methods (i.e., to run just the skyline 
        identification) initialised the class with no autorun and call the 
        method `this_object.get_skylines()`.

        Parameters
        ----------
        start : string 
            Method to begin running.
        end : string
            Method to run until.
            
        Returns
        -------
        start : string 
            Method to begin running.
        end : string
            Method to run until.

        """
        start_str = end if start is False else start
        end_str   = start if end is False else end
        
        choose_default_method_options = ['', None, True]
        start = start_str if start_str not in choose_default_method_options else self._start
        end   = end_str if end_str not in choose_default_method_options else self._end
        return start, end
    
    def autorun(self, autorun):
        """If autorun has been set at class initialisation, class
        will automatically run the analysis. If the start and/or end
        have been provided when initialising, these will be used, else
        it automatically runs all the prep methods in order. These are:
            `continuum`
            `normalise`
            `skylines`
            `science_lines`
            `outliers`

        Parameters
        ----------
        autorun : None or 2-list
            If provided `autorun` contains a list where to start and stop
            the analysis. If `''`, `None` or `True` provided in the 2-list,
            the default start/end are ran. If `False` is provided within the
            2-list, and the other option is a method, only that single method
            is run, though this functionality is best access using the
            `get_'method' methods (i.e., to run just the skyline identification)
            initialised the class with no autorun and call the method
            `this_object.get_skylines()`

        Returns
        -------
        None.

        """
        if autorun == True:
            self.run()
            # self.run(self._start, self._end) 
        elif autorun is None or autorun == '' or autorun == False:
            pass
        elif all([not x for x in autorun]):
            pass        
        else:
            start, end = self._autorun(autorun)
            self.run(start, end)  
            
    def _autorun(self, autorun):
        """For more varied inputs that have obvious meanings, this method
        provides a decision tree to interpret various inputs and
        provide outputs than can be recognised later on when running
        the prep method.

        Parameters
        ----------
        autorun : None or 2-list
            If provided `autorun` contains a list where to start and stop
            the analysis. If `''`, `None` or `True` provided in the 2-list,
            the default start/end are ran. If `False` is provided within the
            2-list, and the other option is a method, only that single method
            is run, though this functionality is best access using the
            `get_'method' methods (i.e., to run just the skyline identification)
            initialised the class with no autorun and call the method
            `this_object.get_skylines()`

        Returns
        -------
        start : str, bool or None
            start method
        end : str, bool or None
            end method

        """
        if autorun == True:
            # if isinstance(autorun, bool):
            self.run()                
        else:
            if len(autorun) > 2:
                raise IndexError(
                    'More than two methods were given. Provide either' \
                    ' a single method to indicate the start, or two to' \
                    ' indicate the start and stop.')
            elif len(autorun) == 1: 
                # if run through autorun, this never triggers
                if autorun[0] == False: 
                    return None
                elif isinstance(autorun[0], str) or autorun[0] == True:
                    start = autorun[0]
                    end = 'outliers'
            else:  
                start = autorun[0]
                end = autorun[1] 
                
            return start, end
                 
    def _get_previous_reduced_data(self, current_process):
        """Wrapper that finds the last run process and returns that data if there.

        Parameters
        ----------
        current_process : string
            String indicating the current method being run

        Returns
        -------
        data : numpy.ndarray
            The data preprocessing that occurred chronologically before
            the method entered as a string as the variable `current_process`            

        """
        current_step = config.prep_names[current_process]
        previous_step = current_step-1

        data = self._get_spectra(previous_step)

        return data

    def _get_spectra(self, processing_step):
        """Method outputs the prepped data provided, checking if potential.
        prerequisite data has been run, and if not, provides a warning

        Parameters
        ----------
        processing_step : string
            String indicating the data that has been called to be retrieved.

        Returns
        -------
        data_for_current_step : numpy.ndarray
            Data for the method requested.
            
        Warnings
        --------
        Raises a warning to the user if a preprocessing step is being run
        but the data in the previous step does not exist. In some cases,
        the relevant preprocessing steps have already been done to the data
        outside of this pipeline, and so if this is the case, ignore this 
        warning

        """
        processing_method = config.prep_steps[processing_step]

        if processing_method in self.processed_spectra:
            data_for_current_step = self.processed_spectra[processing_method]
        else:
            if self.verbose:
                steps = ''
                for n in range(processing_step-1, 0, -1):
                    steps += config.prep_steps[n] + ','

                warnings.warn(processing_method + ' and the earlier reduction ' \
                              'steps, ' + steps + ' have not been run. If '\
                              'entered spectra have not had these steps '\
                              'performed already, PCA will give incorrect '\
                              'results.')
            data_for_current_step = self.sky_spectra

        return data_for_current_step

    def _run_prerequisite_method(self, method, run_if_missing=False):
        """Will try and run a prerequisite method 

        Parameters
        ----------
        method : str
            The name of the method to run.
        run_if_missing : bool, optional
            Whether to run the method is the prerequisite is missing.
            Currently this variable is not used. The default is False.

        Returns
        -------
        numpy.ndarray
            The data for provided method

        """
        if method not in self.processed_spectra:
            self.processed_spectra[method] = getattr(self, 'get_' + method)()

        return self.processed_spectra[method]


#apply wavelength cut to skyline mask
#Apply skyline mask to spectra and error spectra

class pcaSkyBase():
    """This class contains the base functionality to initialise PCA
    analysis (for the sky library and the science) and can run the pca
    prep in one step. 
        
    Attributes
    ----------
    spectra : numpy.ndarray
        The final calibrated, sky subtracted spectra that contain sky
        subtraction residuals.
    skylines : numpy.1darray
        The position of skylines to perform the PCA analysis upon. 1's are
        positions of the skylines (the data to USE) and 0's are where no
        skylines are expected.
    pca_use_mask_and_errors : numpy.ndarray
        Array where all bad/unusable pixels are set to zero so that they
        can be ignored later on, and the remaining pixels are set to the noise 
        error array values. It will be the same shape as `error_spectra`.
    pp_obj : pythonName.pcaPrep object or None, optional
        The pcaPrep object. The default is None.
    verbose : bool, optional
        Whether to print additional information. The default is True.
    pca_kwargs :
        kwargs.
    
    Class methods
    -------------
    from_pcaPrep(pp_obj, **pca_kwargs)
        Takes in the pcaPrep class that has its wavelength range chosen,
        grabs out the data needed, condenses where the pca analysis will
        be run and then inputs them into the class.
    run_pcaPrep(sky_spectra, noise_gen_data, observation_numbers, 
                subtraction_methods, wavelength=None, wavelength_range=None,
                **kwargs)
        With the calibrated, sky subtracted data, this function initialises 
        the pcaPrep class and limits the wavelength range to the limits
        provided. If no limits are provided, then the entire spectrum will
        be run through the PCA method.
    
    Methods
    -------
    None
    
    Purpose of class
    ----------------
    It essentially acts as a wrapper for pcaPrep if required, but can be run 
    normally if the data provided has already been prepped. One of the 
    additional functionalities this base class provides is to mask the
    wavelength range if that is required (doing this here massively simplifies
    the code later on). In previous SDSS, only the red part of the spectrum 
    underwent PCA subtraction as it was the red part that was bad (i.e., 
    skyline forests). Skyline subtraction residuals in the blue were typically
    treated using normal methods. 
    
    Since this is just a base class that is inherited to provide
    this functionality, instructions on how to run the pipeline are
    provided in the child classes, but in summary, it is expected that
    the class and its children are initialised using the class methods.
    The data needed to initialise the pcaPrep method can be inputted here and
    the class will initialise and run the prep object given, and then limit
    the wavelength range to the limits provided.
    
    TODO: I am not sure how well the PCA sky residual removal will work
    on the blue end and it needs to be tested.
    
    TODO:  CHECK IF NANS AND INFS ARE HANDLED CORRECTLY. 
    """
    def __init__(self, spectra, skylines=None, pca_use_mask_and_errors=None, pp_obj=None, verbose=True, **pca_kwargs): #amount_of_eigen=100, save_extra_param=False, c_sq=0.787**2, number_of_iterations=5):
        """
        Parameters
        ----------
        spectra : numpy.ndarray
            The final calibrated, sky subtracted spectra that contain sky
            subtraction residuals.
        skylines : numpy.1darray
            The position of skylines to perform the PCA analysis upon. 1's are
            positions of the skylines (the data to USE) and 0's are where no
            skylines are expected.
        pca_use_mask_and_errors : numpy.ndarray
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the noise 
            error array values. It will be the same shape as `error_spectra`.
        pp_obj : pythonName.pcaPrep object or None, optional
            The pcaPrep object. The default is None.
        verbose : bool, optional
            Whether to print additional information. The default is True.
        pca_kwargs :
            kwargs.
                
        """
        #applying wavelength limits before pca
        spectra, pca_use_mask_and_errors = util.nan_inf_masking(spectra, pca_use_mask_and_errors)
        self.spectra = spectra
        if skylines is None:
            skylines = np.ones(np.shape(spectra)[1])
        self.skylines = skylines
        self.pca_use_mask_and_errors = self._set_pca_use_mask_and_errors(pca_use_mask_and_errors)
        self.pp_obj = pp_obj
        self.verbose = verbose
        self.pca_kwargs = pca_kwargs
        # self.output = self.run(**self.pca_kwargs)

    @classmethod
    def from_pcaPrep(cls, pp_obj, **pca_kwargs):
        """Takes in the pcaPrep class that has its wavelength range chosen,
        grabs out the data needed, condenses where the pca analysis will
        be run and then inputs them into the class.

        Parameters
        ----------
        pp_obj : fileName.pcaPrep
            pcaPrep object.
        **pca_kwargs
            The variables for the pcaPrep object.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        spectra = pp_obj.get_normalise(full=False)
        skylines = pp_obj.get_skylines()
        science_lines = pp_obj.get_science_lines()
        error_spectra = pp_obj.error_spectra

        if error_spectra is None:
            if pp_obj.noise_gen_is_error:
                error_spectra = pp_obj.noise_gen_data

        spectra_to_remove = pp_obj.get_outliers()

        pca_use_mask_and_errors = util.make_pca_use_mask_and_errors(science_lines, error_spectra, spectra_to_remove)

        return cls(spectra, skylines, pca_use_mask_and_errors, pp_obj, verbose=pp_obj.verbose, **pca_kwargs)

    @classmethod
    def run_pcaPrep(cls, sky_spectra, noise_gen_data, subtraction_methods,
                    observation_numbers=None, method_kwargs=None, wavelength=None,\
                    wavelength_range=None, verbose=True, **kwargs):
        """With the calibrated, sky subtracted data, this function initialises 
        the pcaPrep class and limits the wavelength range to the limits
        provided. If no limits are provided, then the entire spectrum will
        be run through the PCA method.
        !!!IMPORTANT: When removing residuals from the science data, the 
        wavelength range provided MUST match the wavelength range of the 
        pca library.

        Parameters
        ----------
        sky_spectra : numpy.ndarray
            The final calibrated, sky subtracted spectra that contain sky
            subtraction residuals.
        noise_gen_data : numpy.ndarray
            The data that contains the noise statistics.
        observation_numbers : array-like
            Some iterable array to track of which spectra come from which
            observation set. Some processing needs to keep track of this
            so that the right model can be found and used.
        subtraction_methods : dict
            What sky subtraction methods to use, which are wrapped into
            the pipeline. 
        wavelength : numpy.1darray, optional
            The wavelength values of the spectra. The default is None.
        wavelength_range : 2-list like array, optional
            The range of the data to useThe default is None.
        verbose : bool, optional
            Whether to print additional information. The default is True.
        **kwargs :
            kwargs

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        kwargs, pca_kwargs = _pop_pca_kwargs(kwargs)
        # pca_library = kwargs.pop('pca_library')
        wavelength_mask = util.get_wavelength_mask_for_spectra(sky_spectra, wavelength, wavelength_range)

        # kwargs = mask_2darr_col_1darr_inputs(wavelength_mask, sky_spectra=sky_spectra, noise_gen_data=noise_gen_data, observation_numbers=observation_numbers, subtraction_type=subtraction_type, **kwargs)
        
        # pcaPrep object decorated so that all the wavelenght range is masked
        # using the wavelength range requested
        pp = util.mask_arrays_columns(wavelength_mask)(pcaPrep)(
            sky_spectra, 
            noise_gen_data=noise_gen_data, 
            subtraction_methods=subtraction_methods, 
            method_kwargs=method_kwargs, 
            autorun=True,
            verbose=verbose,
            **kwargs)
        # pp = pcaPrep(
        #     sky_spectra, 
        #     noise_gen_data=noise_gen_data, 
        #     observation_numbers=observation_numbers, 
        #     subtraction_methods=subtraction_methods, 
        #     method_kwargs=method_kwargs, 
        #     autorun=True,
        #     **kwargs
        # )

        # pp = pcaPrep(**kwargs)
        # pp = pcaPrep(sky_spectra, noise_gen_data, observation_numbers, subtraction_methods, **kwargs)

        return cls.from_pcaPrep(pp, **pca_kwargs)

    def _set_pca_use_mask_and_errors(self, pca_use_mask_and_errors):
        """Sets the mask (i.e where there are skylines and the pixels are good)
        that PCA will run on, where zeros are ignored and the remaining data
        values are the errors. If running a simple reconstruction (no rms
        optimised analysis), and there are no bad pixels or spectra,
        this array is but set to ones.

        Parameters
        ----------
        pca_use_mask_and_errors : numpy.ndarray or None
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.

        Returns
        -------
        pca_use_mask_and_errors : numpy.ndarray or None
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.

        """
        if pca_use_mask_and_errors is None:
            pca_use_mask_and_errors = np.ones_like(self.spectra, dtype=int)

        return pca_use_mask_and_errors


    # def run(self,wavelength_mask, **kwargs):
    #     pass

#function for applying pca to science data
class pcaLibrary(pcaSkyBase):
    """Class creates the eigenspectra (pca library) data that has had the
    prerequisite preprocessing done to it.
    
    Attributes
    ----------
    output : dict
        The generated PCA library. Keys and notation has been set to
        match previous SDSS versions of the PCA method. The dictionary
        contains the keys:
            'U'    : These are the eigen vectors;
            'm'    : This is the mean array spectrum;
            'W'    : These are the eigen values; 
            'vqu'  : These are the iterative values that are constrained
                     to fine the PCA solution;
            'sig2' : The robust scale (i.e., standard deviation) values.
    verbose : bool, optional
        Whether to print additional information. The default is True.
            
    Class methods
    -------------
    run_pcaPrep(sky_spectra, noise_gen_data, observation_numbers, 
                subtraction_methods, wavelength=None, wavelength_range=None,
                **kwargs)
        With the calibrated, sky subtracted data, this function initialises 
        the pcaPrep class and limits the wavelength range to the limits
        provided. If no limits are provided, then the entire spectrum will
        be run through the PCA method.
    from_pcaPrep(pp_obj, **pca_kwargs)
        Takes in the pcaPrep class that has its wavelength range chosen,
        grabs out the data needed, condenses where the pca analysis will
        be run and then inputs them into the class.
    
    Methods
    -------
    run(**pca_kwargs)
        Runs the iterative PCA decomposition on sky subtracted sky data and
    generates the pca library
    
    """

    def __init__(self, spectra, skylines=None, pca_use_mask_and_errors=None, pp_obj=None, verbose=True, **pca_kwargs):
        """Wrapper that runs the robust, iterative PCA decomposition on sky
        subtracted sky data ONLY in the positions where we expect sky to
        be (using a 1d) skyline mask. This object runs automatically when
        initialised.

        Parameters
        ----------
        spectra : numpy.array
            The spectra that will be used to make the sky library from.
        skylines : numpy.1darray, optional
            The position of where the skylines are. If only skyline data is
            being provided, this can be None. The default is None.
        pca_use_mask_and_errors : numpy.ndarray, optional
            The errors of the data, and positions where the data is bad
            by setting those pixels to zero. The default is None.
        pp_obj : fileName.pcaPrep, optional
            pcaPrep object. The default is None.
        verbose : bool, optional
            Whether to print additional information. The default is True.
        **pca_kwargs 
            Kwargs.
            
        """
        super(pcaLibrary, self).__init__(spectra, skylines, pca_use_mask_and_errors, pp_obj, verbose, **pca_kwargs)

        self.output = self.run(**self.pca_kwargs)

    def run(self, **pca_kwargs):
        """Runs the iterative PCA decomposition on sky subtracted sky data and
        generates the pca library

        Parameters
        ----------
        **pca_kwargs :
            Kwargs.

        Returns
        -------
        output : dict
            The generated PCA library. Keys and notation has been set to
            match previous SDSS versions of the PCA method. The dictionary
            contains the keys:
                'U'    : These are the eigen vectors;
                'm'    : This is the mean array spectrum;
                'W'    : These are the eigen values; 
                'vqu'  : These are the iterative values that are constrained
                         to fine the PCA solution;
                'sig2' : The robust scale (i.e., standard deviation) values.

        """
        if not bool(pca_kwargs):
            pca_kwargs = self.pca_kwargs

        # sky_pixels = (wavelength_mask.astype(bool)) & (skylines.astype(bool))

        #RUN ONLY ON SKY PIXELS
        
        spectra, pca_use_mask_and_errors = util.apply_1d_mask_to_2dcol(self.skylines.astype(bool), self.spectra, self.pca_use_mask_and_errors)
        #make this interchangeable with first output being a dictionary of the pca library
        pca_library = pca_kwargs.pop('pca_library', None)
        output = pca.run_robust_pca(spectra, errors=pca_use_mask_and_errors, **pca_kwargs)

        return output
        #add wavelength mask


# recon_kwargs = {
#     'rms_percentile' : 90,
#     'max_comp' : 300,
#     'ncomp' : 5
# }





# class rmsSkyCalculations():
    # skylines : numpy.1darray
    #     The position of skylines to perform the PCA analysis upon. 1's are
    #     positions of the skylines (the data to USE) and 0's are where no
    #     skylines are expected.
    # pca_use_mask_and_errors : numpy.ndarray
    #     Array where all bad/unusable pixels are set to zero so that they
    #     can be ignored later on, and the remaining pixels are set to the noise 
    #     error array values. It will be the same shape as `error_spectra`.
    # pp_obj : pythonName.pcaPrep object
    #     The pcaPrep object.
    # pca_kwargs :
    #     kwargs.
class pcaSubtraction(pcaSkyBase):
    """Performs the sky residual removal on the normalised sky subtracted
    science spectra ONLY on the parts that have skylines spectra 
    using the provided sky library containing the results of the eigen
    system.  
    
    Class methods
    -------------
    from_pcaPrep(pp_obj, **pca_kwargs)
        Takes in the pcaPrep class that has its wavelength range chosen,
        grabs out the data needed, condenses where the pca analysis will
        be run and then inputs them into the class.
    run_pcaPrep(sky_spectra, noise_gen_data, observation_numbers, 
                subtraction_methods, wavelength=None, wavelength_range=None,
                **kwargs)
        With the calibrated, sky subtracted data, this function initialises 
        the pcaPrep class and limits the wavelength range to the limits
        provided. If no limits are provided, then the entire spectrum will
        be run through the PCA method.
    
    Attributes
    ----------
    spectra : numpy.ndarray
        Spectra that have correlated residuals to be removed using a pca
        library.
    pca_library : dict
        The generated PCA library. Keys and notation has been set to
        match previous SDSS versions of the PCA method. The dictionary
        contains the keys:
            'U'    : These are the eigen vectors;
            'm'    : This is the mean array spectrum;
            'W'    : These are the eigen values; 
            'vqu'  : These are the iterative values that are constrained
                     to fine the PCA solution;
            'sig2' : The robust scale (i.e., standard deviation) values.
    poisson_noise : numpy.ndarray
        The Poisson noise to use for each spectra.
    continuum : numpy, optional
        The continuum for each spectra. The default is None.
    skylines : numpy.1darray, optional
        If the spectra given contain sky (to be corrected) and nonsky
        pixels, indicate the sky using a 1d mask with values one,
        and zero for non sky pixels. If all pixels are sky pixels and
        just want a simple reconstruction of the spectra with spectra 
        being reconstructed using the same number of components,
        no skyline mask is needed. The default is None.
    error_spectra : numpy.ndarray, optional
        If the array used to generate the Poisson noise is different to
        the science data error arrays (which IS the case if using skycorr)
        and the better reconstruction is being performed using the optimal
        number of components per spectra, `error_spectra` is needed
        not needed if no rms checks (i.e., simple 
        reconstruction using  wanted
        #If rms checks not needed but spectra have additional bad pixels
        # to mask out, set `error_spectra` to 0 where there are bad pixels
        # and to 1 everywhere else. The default is None.
    pca_use_mask_and_errors : numpy.ndarray or None, optional
        To ignore pixels entirely in the pca reconstruction since they are
        outliers, but want pca to fill in that pixel using the
        eigenvectors, enter an array that has zeros at those locations.
        If you have errors, the remaining positions are the error values.
        If errors not needed, set the remaining to 1. The default is None.
    pp_obj : fileName.pcaPrep, optional
        The pcaPrep object. The default is None.
    rms_function : func
        The function used to find the rms of each spectrum. The default is
        `get_rms`
    pca_projection_function : func
        The function used to reconstruct the spectra using pca projection.
        It is not recommended to change this as the current project method 
        (`project_spectra`, a wrapper for the gappy reconstruction routine) 
        uses a method that deals with data gaps (i.e., bad pixels) in a 
        consistent way. This is needed to predict why the sky residual looks 
        like if we had to mask out that part of the spectrum due to it 
        containing a science line, or some corrupted data. This method predicts
        how the skyline residual that might be there impacts the science line 
        using the remaining skyline residuals and subtracts it. If a different 
        method for dealing and predicting data has been masked out in the 
        reconstruction wants to be used instead, this is less of an issue.
        The default is `project_spectra`.
    verbose : bool, optional
        Whether to print additional information. The default is True.
    **pca_kwargs :
        kwargs
    
    NOTES
    -----
    Noted here is that this method ONLY should be run on the part
    of the spectra that we expect skylines (and therefore where residuals)
    should be. 
    
    Running information
    -------------------
    There are two ways that this class provides ways to remove the 
    sky subtraction residuals. This first way (not recommended for 
    actual scientific analysis, but for simple tests), is to reconstruct
    each spectra using the exact same number of pca components. This
    will provide a quick, but a non-optimal solution for the sky residual
    subtraction. If doing this method, you can either input spectra that
    have already been masked and so only contain the chopped up parts
    of the spectra with skylines, but its probably simpler to provide
    the range to be analysed and provide a skyline mask.
    
    The reason this 1st method is not recommended is that for each 
    individual spectrum, use of too few components in the reconstruction 
    (and therefore the in the reconstructed spectrum to be subtracted from 
    the sky subtracted science) results in not enough noise being removed, 
    and will often result in making the spectrum WORSE. Using too many 
    components causes the rms noise to become lower than the rest of the 
    data (i.e., lower than the rms noise in science spectra that has no 
    skylines and so is untouched by the analysis.) and so acts like a
    high-frequency filter. Since this is unphysical it should not be
    done nor used for science.
    
    Therefore the better way (but results in a more complex method) is
    to iteratively reconstruct each spectra, subtract this reconstruction
    from the skyline part of the spectra, and then compare the rms of
    the subtracted part to the untouched part of the spectra that contains
    no skylines (i.e., represents the true rms). If the rms of the 'sky' 
    pixels after the subtraction are higher than the 'non-sky' pixels, add 
    another pca component and repeat. If the rms of the 'sky' pixels after 
    the subtraction are now lower than the 'non-sky' pixels, stop, and use 
    this as the final spectrum.
    
    So the 1st method should just be used to provide a quick look
    at the PCA library and that everything is working. It also means
    you can fix the number of components that are used and provides a
    good basis for comparisons and test. 
    
    For the 2nd method, you have to provide all the data needed before
    running the final subtraction (i.e, the final version of the data)
    and different PCA library will result in a different number of
    components being used, which can make comparisons and testing harder.
    """
    #error is the noise error. For bad pixels, set them to 0 (NaN's should be converted to 0)
    #For skycorr, the array used to generate the Poisson noise is different to the observation
    #errors/noise, so they need to be defined separately
    def __init__(self, spectra, pca_library, poisson_noise, continuum=None, skylines=None, error_spectra=None, pca_use_mask_and_errors=None, pp_obj=None, verbose=True, **pca_kwargs):
        """
        Parameters
        ----------
        spectra : numpy.ndarray
            Spectra that have correlated residuals to be removed using a pca
            library.
        pca_library : dict
            The generated PCA library. Keys and notation has been set to
            match previous SDSS versions of the PCA method. The dictionary
            contains the keys:
                'U'    : These are the eigen vectors;
                'm'    : This is the mean array spectrum;
                'W'    : These are the eigen values; 
                'vqu'  : These are the iterative values that are constrained
                         to fine the PCA solution;
                'sig2' : The robust scale (i.e., standard deviation) values.
        poisson_noise : numpy.ndarray
            The Poisson noise to use for each spectra.
        continuum : numpy, optional
            The continuum for each spectra. The default is None.
        skylines : numpy.1darray or None, optional
            If the spectra given contain sky (to be corrected) and nonsky
            pixels, indicate the sky using a 1d mask with values one,
            and zero for non sky pixels. If all pixels are sky pixels and
            just want a simple reconstruction of the spectra with spectra 
            being reconstructed using the same number of components,
            no skyline mask is needed. The default is None.
        error_spectra : numpy.ndarray, optional
            If the array used to generate the Poisson noise is different to
            the science data error arrays (which IS the case if using skycorr)
            and the better reconstruction is being performed using the optimal
            number of components per spectra, `error_spectra` is needed
            not needed if no rms checks (i.e., simple 
            reconstruction using  wanted
            #If rms checks not needed but spectra have additional bad pixels
            # to mask out, set `error_spectra` to 0 where there are bad pixels
            # and to 1 everywhere else. The default is None.
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            To ignore pixels entirely in the pca reconstruction since they are
            outliers, but want pca to fill in that pixel using the
            eigenvectors, enter an array that has zeros at those locations.
            If you have errors, the remaining positions are the error values.
            If errors not needed, set the remaining to 1. The default is None.
        pp_obj : fileName.pcaPrep, optional
            The pcaPrep object. The default is None.
        verbose : bool, optional
            Whether to print additional information. The default is True.            
        **pca_kwargs :
            kwargs

        Returns
        -------
        None.

        """
        super(pcaSubtraction, self).__init__(spectra, skylines, pca_use_mask_and_errors, pp_obj, verbose, **pca_kwargs)
                
        #I always need to apply the wavelength cut if there so I should apply
        # it here. I will raise an error if the mean pca array
        # is a different length after wavelength cut

        ###
        #There are many masks which are needed or can supply:
        #`pca_use_mask_and_errors`
        #  To ignore pixels entirely in the pca reconstruction since they are
        #    outliers, but want pca to fill in that pixel using the
        #    eigenvectors, enter an array that has zeros at those locations.
        #    If you have errors, the remaining positions are the error values.
        #    If errors not needed, set the remaining to 1
        #'skylines'
        #  If the spectra given contain sky (to be corrected) and nonsky
        #    pixels, indicate the sky using a 1d mask with values one,
        #    and zero for non sky pixels. If all are sky pixels and
        #    just want a constant reconstruction no skylines are needed


        #The pca library
        self.pca_library = pca_library

        ####
        #`error_spectra` not needed if no rms checks wanted
        #If rms checks not needed but spectra have additional bad pixels
        # to mask out, set `error_spectra` to 0 where there are bad pixels
        # and to 1 everywhere else
        self.error_spectra = error_spectra
        if self.error_spectra is not None:
            self.pca_use_mask_and_errors[self.error_spectra==0] = 0
            replace_with_errors = self.pca_use_mask_and_errors !=0
            self.pca_use_mask_and_errors[replace_with_errors] = self.error_spectra[replace_with_errors]

        #Need to reverse all the preprocessing
        self.continuum = continuum
        #Needed to reverse preprocessing and if rms checks wanted (must undo
        # normalisation to measure rms)
        self.poisson_noise = poisson_noise
        
        #the rms function takes in the un-normalised spectra, the pixel
        #of a spectra to calculate the rms from, and the errors (which 
        #include bad pixels with nan values)
        self.rms_function = get_rms
        self.pca_projection_function = project_spectra

    @classmethod
    def from_pcaPrep(cls, pp_obj, **pca_kwargs):
        """With the calibrated, sky subtracted data, this function initialises 
        the pcaPrep class and limits the wavelength range to the limits
        provided. If no limits are provided, then the entire spectrum will
        be run through the PCA method
        !!!IMPORTANT: When removing residuals from the science data, the 
        wavelength range provided MUST match the wavelength range of the 
        pca library.

        Parameters
        ----------
        pp_obj : fileName.pcaPrep
            The pcaPrep object.
        **pca_kwargs : 
            Kwargs

        Raises
        ------
        ValueError
            Will not run if no pca library is provided.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        continuum = pp_obj.get_continuum(full=True)[1]
        spectra, poisson_noise = pp_obj.get_normalise(full=True)
        skylines = pp_obj.get_skylines() #make dummy
        science_lines = pp_obj.get_science_lines()
        error_spectra = pp_obj.error_spectra

        if error_spectra is None:
            if pp_obj.noise_gen_is_error:
                error_spectra = pp_obj.noise_gen_data

        spectra_to_remove = pp_obj.get_outliers()

        pca_use_mask_and_errors = util.make_pca_use_mask_and_errors(science_lines, error_spectra, spectra_to_remove)

        pca_library = pca_kwargs.pop('pca_library',None)
        if pca_library is None:
            raise ValueError('Cannot run sky subtraction unless the pca '\
                             'library is provided using the variable, '\
                             '`pca_library`.')

        return cls(spectra, pca_library, poisson_noise, continuum, skylines, error_spectra, pca_use_mask_and_errors, pp_obj, pp_obj.verbose, **pca_kwargs)

    def run(self, projection_method='best', **pca_kwargs):
        """Runs the sky residual removal on the sky subtracted science data.
        For a quick check of the removal using a set number of components
        in the reconstructed spectra (which can be changed from the default
        using the optional kwarg variable `max_comp`), set the variable
        `projection_method` to 'simple'. For the optimised number of components
        per spectrum in the reconstruction, set `projection_method` to 'best'.

        Parameters
        ----------
        projection_method : str, optional
            The method for reconstructing the spectra. For a quick 
            reconstruction and non-optimal subtraction that does not use the
            rms, chose `'simple'`. For a optimal construction that adds 
            components to the reconstruction for each spectra until the rms
            of the pca altered sky pixels matches the unaltered, non-sky pixels.
            Requires more arrays (i.e., noise array) to work. The default is
            'best'.
        **pca_kwargs : 
            Kwargs.

        Returns
        -------
        output : numpy.ndarray
            The reconstructed spectra.

        """
        #==`'simple'` is reprojection using a constant number of components
        #==`'best'` is reprojection using the number of components per spectra
        #   that results in an improved rms compared to non-projected
        #   pixels. Requires the Poisson noise spectra and the error spectra

        if not bool(pca_kwargs):
            pca_kwargs = self.pca_kwargs

        output = getattr(self, 'get_' + projection_method + '_projection')(**pca_kwargs)

        return output
    
    def reset_default_rms_function(self):
        self.rms_function = get_rms
        
    def set_rms_function(self, rms_function):
        if rms_function is None:
            self.reset_default_rms_function()
        else:
            self.rms_function = rms_function
            
    def reset_default_pca_projection_function(self):
        self.pca_projection_function = project_spectra
        
    def set_pca_projection_function(self, pca_projection_function):
        """Allows a uses to change the reprojection method, but this is not
        recommended as the current project method (`project_spectra`, a wrapper
        for the gappy reconstruction routine) uses a method that deals with
        data gaps (i.e., bad pixels) in a consistent way. This is needed to
        predict why the sky residual looks like if we had to mask out that part
        of the spectrum due to it containing a science line, or some corrupted
        data. This method predicts how the skyline residual that might be
        there impacts the science line using the remaining skyline residuals
        and subtracts it. If a different method for dealing and predicting
        data has been masked out in the reconstruction wants to be used 
        instead, this is less of an issue.

        Parameters
        ----------
        pca_projection_function : func
            The function to reconstruct spectra. The default is 
            `pca_projection_function`.

        Returns
        -------
        None.

        """
        if pca_projection_function is None:
            self.reset_default_pca_projection_function()
        else:
            self.pca_projection_function = pca_projection_function           

    def _check_prerequisite_data_exists(self):
        #check poisson noise array exists since it is needed to undo the
        #normalisation

        #With skycorr, the poisson noise and the error spectra are not
        #the same so need to have them distinct.
        pass

    def get_simple_projection(self, max_comp=5, pca_use_mask_and_errors=None, 
                              pca_projection_function=None, 
                              undo_preprocessing=False):
        """Method runs a reconstruction of all spectra using the same
        number of components, using the reconstruction method. 
        !!!Do not change or provide a pca_projection_function function
        unless it is for tests, or a better method for dealing with data
        gaps in reconstruction is being provided     

        Parameters
        ----------
        max_comp : TYPE, optional
            DESCRIPTION. The default is 5.
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values. The default is None.
        undo_preprocessing : bool, optional
            Whether to undo the pre-processing on the spectra before returning
            is. The default is False

        Returns
        -------
        reconstructed_spectra : numpy.ndarray
            The reconstructed spectra. If spectra is True the pca subtracted
            spectra, with the pre-processing removed is returned instead

        """
        self.set_pca_projection_function(pca_projection_function)
        spectra, pca_use_mask_and_errors = self._projection_prep(pca_use_mask_and_errors)
        mean_spectra, eigen_vectors = self._projection_eigen_sys()
        
        reconstructed_spectra = copy.deepcopy(spectra)

        reconstructed_spectra[:,self.skylines==1] = self.pca_projection_function(spectra[:,self.skylines==1],
                                                                          pca_use_mask_and_errors[:,self.skylines==1], 
                                                                          eigen_vectors, 
                                                                          mean_spectra, 
                                                                          namplitudes=max_comp,
                                                                          run_mask=True,
                                                                      )

        
        if undo_preprocessing:
            reconstructed_spectra = self.undo_preprocessing(spectra-reconstructed_spectra)

        return reconstructed_spectra

    def _skip_if_too_many_bad_pixels(self, spectra_mask_high_rms, pca_use_mask_and_errors, fraction_to_skip=0.2):
        """Method finds if any spectra have too many missing pixels and masks 
        them out. If too many pixels are missing, the reconstruction quality
        can be affected resulting in worse spectra. This test is carried out
        automatically and if such spectra are found, the mask that indicates
        which spectra need reconstructing (spectra_mask_high_rms) is altered
        so that it thinks no reconstruction is needed.

        Parameters
        ----------
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra
        pca_use_mask_and_errors : numpy.ndarray
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.
        fraction_to_skip : int between 0 to 1, optional
            The fraction of bad pixels in a spectrum that when surpassed,
            the spectrum is skipped as it can result in biasing the data. The
            default is 0.2.

        Returns
        -------
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra

        """
        pca_use_mask_binary = copy.deepcopy(pca_use_mask_and_errors)
        #wher data, 1's. Where bad pixels, 0
        pca_use_mask_binary[pca_use_mask_binary != 0] = 1
        
        ##
        #If onsky, all other pixels are set to 0. This now leaves a bit
        #map where the off sky and the bad pixels are 0
        # onsky_mask = self.where_on_or_off_sky(pca_use_mask_binary, True)
        # num_pixels = np.sum(onsky_mask, axis=1)
        
        #Getting the total number of actual on sky pixels
        num_pixels = np.sum(self.skylines)
        
        ##
        #number of good pixels on skyline locations / total number of pixels at
        #at sky locations
        good_pixels_percentage = np.sum(pca_use_mask_binary, axis=1)/num_pixels
        
        #True when the fraction of usable pixels at skyline locations is
        #less than `fraction_to_skip` (default: 20%). Therefore spectra where
        #more than 20% of the pixels at sky locations are usable, we ignore
        where_skip = (1 - good_pixels_percentage) >= fraction_to_skip
        
        
        spectra_mask_high_rms[where_skip] = False

        return spectra_mask_high_rms

    def get_best_projection(self, pca_use_mask_and_errors=None, max_comp=300,
                            frac_pix_to_skip=0.2, rms_leeway=0.00, return_extra_info=False, 
                            rms_function=None, pca_projection_function=None,
                            undo_preprocessing=False, replace_max_solution=True):
        """Finds the best number of components to use for each spectra by
        comparing the rms of the data-reconstructed data, where data is the
        science spectra that can contain skylines, vs the rms of data that
        has no skylines (and so therefore no reconstruction is subtracted).
        Once the rms of the two match, the spectra is finalised. If the rms 
        of the data with skylines never drops below the data without skylines 
        before the maximum number of components to reconstruct with has been
        reached (`max_comp`), and `replace_max_solution` is True the 
        minimum rms value found throughout the reconstruction process for a 
        given spectrum is used so long as the rms is lower than than it started
        at. If the rms never improves, the original spectrum is returned
        instead. This only occurs if
                
        Parameters
        ----------
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values. The default is None.
        max_comp : int, optional
            Th maximum number of components to use in the reconstruction. The
            default is 300.
        frac_pix_to_skip : float between 0 to 1, optional
            The fraction of bad pixels in a spectrum that is surpassed, no
            reconstruction/subtraction is performed. The default is 0.2.
        rms_leeway : float, optional
            Allows the rms of sky pixels to be a little bit higher than the 
            non-sky pixels by `rms_leeway` amount. This makes the stopping
            critiria less strict by a small amount. This exists so that if the 
            rms is really close, like when using fake data to test but doesn't 
            quite make the rms to beat, it helps stop the routine from
            adding a too many of extra components.
            !!!WARNING do not set this high or the reconstruction will not
            be optimal. The default is 0.00. 
        return_extra_info : bool, optional
            Whether to return additional information about how the 
            reconstruction went, such as how the rms changed for each spectrum
            when adding components, and the final number of components used
            for each spectrum. The default is False.
        rms_function : func, optional
            The rms function to determine when the rms of each spectra is good
            enough. If `None`, the default function defined (recommended) is
            used. The default is None.
        undo_preprocessing : bool, optional
            Whether to undo the pre-processing on the spectra before returning
            is. The default is False
        replace_max_solution : bool, optional
            Whether to replace the reconstruction with the lowest rms found
            during reconstruction and to replace reconstructions where the
            rms never improved with the origonal spectra.
            TODO: behaviour of this needs to be tested better. 
            The default is True

        Returns
        -------
        pca_subtracted_spectra : numpy.ndarray
            Spectra that have the reconstructed spectra that have been
            optimally found based of the improvement in rms subtracted from
            the original spectra
            
        Optional
        --------
        rms_changes : numpy.ndarray
            The change in rms as a new component was added until either the rms
            was good enough, or the maximum number of components to used, 
            indicated by variable `max_comp` had been reached. Only returned if
            `return_extra_info` is True. default of `return_extra_info` is 
            False.
        n_comp_used : numpy.1darray
            The number of components eventually used in the reconstruction. Only
            returned if `return_extra_info` is True. default of 
            `return_extra_info` is False.
            
        Warnings
        --------
        Raises a warning to the user if `rms_leeway` is set above or equal to
        0.05. Needs testing but even 0.04 might be too high. This exists
        so that if the rms is really close, like when using fake data to test
        but doesn't quite make the rms to beat, it helps stop the routine from
        adding a load of extra components.
            
        Methodology:
        -----------
        - The inputted spectra get inputted into a prep function to find the
        starting rms of the science spectra containing skylines, and the
        science spectra without skylines using `pca_use_mask_and_errors`.
        The mask part of this variable means that where the data is bad and
        should not be used to calculate the rms. The errors part contains
        the noise map of each spectrum needed for find the rms;
        - If the rms at skyline locations are already better than the 
        non-skyline locations, no reconstruction is performed;
        - If too my pixels in a given spectrum (default at 20%: 
          `frac_pix_to_skip`) are bad, no reconstruction is performed as the
          resulting reconstruction is likely to be bad due to the number of
          missing pixels. Now loop the remaining up to the set maximum
          number of components (`max_comp`) to use in reconstruction. The use
          of too many components can result in reduction of the quality of the
          data, with the PCA acting like a high-frequency filter:
        1. Reconstruct with nth number of components (vectorised action);
        2. Subtract the reconstruction from the positions containing skylines;
        3. Undo the Poisson normalisation (this step is done for all rms 
           checks);
        4. Calculate the new rms of the spectra that contained skylines;
        5. If the rms is better than the non-skyline data, stop reconstructing
           that spectrum. For spectra that still have higher rms's go back to 
           step 1.; 
        - End loop. Loop is performed in method `loop_rms_projection`
        - If the maximum amount of components to reconstruct with has been 
           reached and the rms of skyline pixels is still too high, find the
           minimum rms that was reached during the iterative reconstruction
           process;
        - If this minimum rms is lower than the starting rms, use this
          reconstruction, otherwise, return the original spectrum with no
          alterations.
          
        During this process, all the rms's for each included component, and
        the final number of components needed to improve the rms are tracked.
        This information is not returned on default, but to output them, set
        the method variable `return_extra_info` to True.

        """       
        #This allows the user to provide a different rms function and set
        # it as the attribute `rms_function`. To reset to the default function
        # call the method `reset_default_rms_function()`.
        self.set_rms_function(rms_function)
        
        #TODO: THIS IS AN EMPTY FUNCTION, FIX
        self._check_prerequisite_data_exists() #need skylines, need errors
        # sky_pixels = (wavelength_mask.astype(bool)) & (skylines.astype(bool))
        
        ######
        #just applying the masks and getting the baseline rms for
        # pixels not used in the pca
        ##
        #if no `pca_use_mask_and_errors` is given, the default attribute is used.
        # (this just allows for some flexibility but I don't expect the user to
        # input a different `pca_use_mask_and_errors` than the calculated one
        # in reality)
        ####
        if rms_leeway >=0.05 and self.verbose:
            warnings.warn('Setting `rms_leeway` above 0.05, could impact the'\
                          'quality of the final spectra in a negative away.')
            
        spectra, pca_use_mask_and_errors, rms_to_beat = self._projection_best_prep(pca_use_mask_and_errors)
        rms_to_beat += rms_leeway

        #performing the loop to iteratively find the number of component needed to reproject and get a better rms
        pca_subtracted_spectra, spectra_mask_high_rms, rms_changes, n_comp_used = \
            self.loop_rms_projection(spectra, pca_use_mask_and_errors, rms_to_beat, 1, max_comp, frac_pix_to_skip, pca_projection_function)
        
        #replacing where max_comp was reached with origonal spectra, which
        #will only be updated if rms lowered
        if replace_max_solution:
            
            pca_subtracted_spectra, rms_changes, n_comp_used = \
                self._replace_remaining_spectra(pca_subtracted_spectra, spectra_mask_high_rms, rms_changes, n_comp_used, pca_use_mask_and_errors, frac_pix_to_skip, pca_projection_function)
        
        rms_changes[0] = rms_to_beat - rms_leeway
        
        if undo_preprocessing:
            pca_subtracted_spectra = self.undo_preprocessing(pca_subtracted_spectra)
        
        if return_extra_info:
            return pca_subtracted_spectra, rms_changes, n_comp_used
        else:
            return pca_subtracted_spectra
        
    def _replace_remaining_spectra(self, pca_subtracted_spectra, spectra_mask_high_rms, 
                                  rms_changes, n_comp_used, pca_use_mask_and_errors=None,
                                  frac_pix_to_skip=0.2, pca_projection_function=None):
        
        spectra, pca_use_mask_and_errors, rms_to_beat = self._projection_best_prep(pca_use_mask_and_errors)
        
        pca_subtracted_spectra[spectra_mask_high_rms] = spectra[spectra_mask_high_rms]
        
        #some spectra might not have beat the rms set. If their rms got
        # better, reconstruct using the number of components that had the
        # minimum rms. If rms never decreased, no recon is performed
        where_to_recon, ncomp_at_min_rms = self._remaining_spectra_projection(spectra_mask_high_rms, rms_changes[1:])
        
        #Changed n_comp to zero and if `where_to_recon` has values, it will
        #be updated
        n_comp_used[spectra_mask_high_rms] = 0
        
        #has to be looped since using a specific number of components
        #to project each spectra that didn't beat the rms of the reference
        try: # if `max_comp` was not reached, and reconstuction complete, pass
            for ind in range(where_to_recon):
    
                i = ncomp_at_min_rms[ind]
                pca_subtracted_spectra[ind], spectra_mask_high_rms[ind], rms_changes[i:,ind], n_comp_used[i] = \
                    self.loop_rms_projection(spectra[ind], pca_use_mask_and_errors[ind], rms_changes[0], i, i, frac_pix_to_skip, pca_projection_function)
        except TypeError:
            pass
        
        return pca_subtracted_spectra, rms_changes, n_comp_used
        

    def _remaining_spectra_projection(self, spectra_mask_high_rms, rms_changes):
        """If the rms never fell below the rms to beat before the maximum number
        of components to use in the reconstruction was met, this method looks
        for the lowest rms achieved during each reconstruction with n+1
        components, and uses that spectrum. If the rms is never lower that 
        the original spectrum, the original is returned.

        Parameters
        ----------
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra
        rms_changes : numpy.ndarray
            The change in rms as a new component was added until either the rms
            was good enough, or the maximum number of components to used, 
            indicated by variable `max_comp` had been reached.
            
        Returns
        -------
        where_to_recon : numpy.ndarray
            Indicies of where to reconstruct to.
        ncomp_at_min_rms : numpy.1darray
            The number of components to use in the reconstruction.

        """
        min_rms = np.min(rms_changes, axis=0)
        ncomp_at_min_rms = np.argmin(rms_changes, axis=0)

        to_recon = ((min_rms - rms_changes[0]) <= 0) & (spectra_mask_high_rms)
        # no_recon = ((min_rms - rms_changes[0]) >= 0) & (spectra_mask_high_rms)

        # ncomp_at_min_rms = ncomp_at_min_rms[to_recon]
        where_to_recon = np.where(to_recon)[0]

        # return where_to_recon, no_recon, ncomp_at_min_rms
        return where_to_recon, ncomp_at_min_rms

    def loop_rms_projection(self, spectra, pca_use_mask_and_errors, rms_to_beat, n_start, n_end, frac_pix_to_skip=0.2, pca_projection_function=None):
        """This function performs the iterative loop to reconstruct the spectra
        with n number of components, and to stop when the rms where the spectra
        contain skylines falls below the rms of non-skyline data. The loop
        steps are:
            1. Reconstruct with nth number of components (vectorised action);
            2. Subtract the reconstruction from the positions containing skylines;
            3. Undo the Poisson normalisation (this step is done for all rms 
               checks);
            4. Calculate the new rms of the spectra that contained skylines;
            5. If the rms is better than the non-skyline data, stop reconstructing
               that spectrum. For spectra that still have higher rms's go back to 
               step 1.

        Parameters
        ----------
        spectra : numpy.ndarray
            Spectra that are to have their error residuals reconstructed.
        pca_use_mask_and_errors : numpy.ndarray 
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.
        rms_to_beat : numpy.ndarray
            The rms per spectra that needs to be beat for reconstruction to 
            stop.
        n_start : int
            The number of components to start the reconstruction with.
        n_end : int
            The max number of components to reconstruction with.
        frac_pix_to_skip : float between 0 to 1, optional
            The fraction of bad pixels in a spectrum that is surpassed, no
            reconstruction/subtraction is performed. The default is 0.2.

        Returns
        -------
        pca_subtracted_spectra : numpy.ndarray
            Spectra that have the reconstructed spectra that have been
            optimally found based of the improvement in rms subtracted from
            the original spectra
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra
        rms_changes : numpy.ndarray
            The change in rms as a new component was added until either the rms
            was good enough, or the maximum number of components to used, 
            indicated by variable `max_comp` had been reached.
        n_comp_used : numpy.1darray
            The number of components eventually used in the final
            reconstruction.

        """
        #Allows a uses to change the reprojection method, but this is not
        # recommended as the current project method uses a method that
        # deals with data gaps (i.e., bad pixels) in a consistent way. This
        # is needed to predict why the sky residual looks like if we had to
        # mask out that part of the spectrum due to it containing a science
        # line. This method predicts how the skyline residual that might be
        # there impacts the science line using the remaining skyline residuals
        # and subtracts it. If a different method for dealing and predicting
        # data has been masked out in the reconstruction wants to be used 
        # instead, this is less of an issue.
        self.set_pca_projection_function(pca_projection_function)
    
        ###
        #`spectra_mask_high_rms` is a 1d bool array indicating where the rms
        # of skyline pixels is higher than the non-skyline pixels (True 
        # indicates continue reconstructing as the rms is too high. False means
        # rms has been met so skip this spectrum)
        mean_spectra, eigen_vectors = self._projection_eigen_sys()
        spectra_mask_high_rms, pca_subtracted_spectra, n_comp_used, rms_changes = self._projection_best_array_init(spectra, n_end)

        #pca reconstruct might be bad if more than 20% of pixels are masked
        # so no reconstruct is done on these
        spectra_mask_high_rms = self._skip_if_too_many_bad_pixels(spectra_mask_high_rms, pca_use_mask_and_errors, fraction_to_skip=frac_pix_to_skip)
        pca_subtracted_spectra[~spectra_mask_high_rms] = spectra[~spectra_mask_high_rms]
        #reconstruct has been vectorised and will reconstruct all spectra
        # for the given number of components. Loop increases the number of
        # components incrementally, finds which have beat the rms limit,
        # masks them and then repeats on the unmasked spectra with +1
        # components until all have had their rms improved, or the limit
        # for the number of components to project with has been hit.
        for i in range(n_start, n_end+1):
            if self.verbose:
                print('Using %d components for %d remaining spectra' % (i, len(spectra_mask_high_rms[spectra_mask_high_rms])))

            #tracking how many components have been used to reconstruct
            n_comp_used[spectra_mask_high_rms] = i

            #masking out spectra with good enough rms. This isn't a 1,0's mask
            # but it removes the spectrum from `spectra` to make the array
            # spectra_r which only contains spectra to reconstruct and saves
            # the reconstructed with good enough rms to the array
            # `pca_subtracted_spectra`
            spectra_r, pca_use_mask_and_errors_r = util.apply_1d_mask_to_2drow(
                spectra_mask_high_rms, 
                spectra, 
                pca_use_mask_and_errors
            )
            reconstructed = copy.deepcopy(spectra_r)
            #make a user settable function
            #reconstructing with i number of components
            reconstructed[:,self.skylines==1] = self.pca_projection_function(
                spectra_r[:,self.skylines==1], 
                pca_use_mask_and_errors_r[:,self.skylines==1], 
                eigen_vectors,
                mean_spectra, 
                i
            )
            #getting the new rms of the reconstructed
            recon_rms = self.get_rms_from_reconstruct(
                spectra_r, 
                reconstructed, 
                pca_use_mask_and_errors_r
            )
            #tracking the rms changes as number of components (i) increases
            rms_changes[i:, spectra_mask_high_rms] = recon_rms

            # reconstructed spectra with good enough rms are saved and removed
            # from being reconstructed further.
            subtracted_spectra = copy.deepcopy(spectra_r)
            subtracted_spectra[:,self.skylines==1] = spectra_r[:,self.skylines==1] - reconstructed[:,self.skylines==1]
            
            #for final values
            spectra_mask_high_rms_before = copy.deepcopy(spectra_mask_high_rms)

            pca_subtracted_spectra, spectra_mask_high_rms = self.set_spectra(
                subtracted_spectra,
                pca_subtracted_spectra, 
                rms_to_beat,
                recon_rms, 
                spectra_mask_high_rms
            )


            #stop if all spectra have had their rms improved
            if np.all(~spectra_mask_high_rms) and i<n_end:
                rms_changes[i+1:] = 0
                return pca_subtracted_spectra, spectra_mask_high_rms, rms_changes, n_comp_used
                # break
        #Anywhere still bad is set with the final reconstuction
        pca_subtracted_spectra[spectra_mask_high_rms_before] = subtracted_spectra#[spectra_mask_high_rms]

        return pca_subtracted_spectra, spectra_mask_high_rms, rms_changes, n_comp_used

    def get_rms_from_reconstruct(self, current_spectra, reconstructed_spectra, pca_use_mask_and_errors=None):
        """Finds the rms after subtracting the residual reconstruction from
        the spectra remaining.

        Parameters
        ----------
        current_spectra : numpy.ndarray
            The original spectra.
        reconstructed_spectra : numpy.ndarray
            The reconstructed spectra to subtract from the original spectra.
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values. The default is None.

        Returns
        -------
        rms_per_specta : numpy.1darray
            The rms for each spectra.

        """
        spectra_residual_removed = current_spectra - reconstructed_spectra

        rms_per_specta = self._get_rms(spectra_residual_removed, pca_use_mask_and_errors, on_sky=True)

        return rms_per_specta

    def where_on_or_off_sky(self, current_spectra_mask, on_sky):
        """Masks the relevant part of the spectrum to analyse either the pixels 
        containing skylines by setting the variable `on_sky` to True and to
        False for `non-sky` pixels instead.

        Parameters
        ----------
        current_spectra_mask : numpy.ndarray
            Mask to use when calculating the rms.
        on_sky : bool
            Whether to get the rms on the pixels (`on_sky=True`) that need 
            reconstructing or the pixels that don't need reconstructing
            (`on_sky=False`).

        Returns
        -------
        mask : numpy.ndarray
            The mask to use for the rms. 0 are not used.

        """

        # on_sky = np.int32(on_sky)
        skylines = self.skylines.astype(bool)
        
        if on_sky:
            current_spectra_mask[:, ~skylines] = 0
        else:
            current_spectra_mask[:, skylines] = 0

        return current_spectra_mask

    def _get_rms(self, spectra, pca_use_mask_and_errors, on_sky=True):
        """Wrapper to determine which pixels are being used to get the rms, and
        then calculates the rms. To do this first the Poisson error
        normalisation needs to be reversed to get the true noise on the
        data then rms can be estimated using the rms function provided in the 
        attribute `rms_function`. Default `rms_function` is a robust estimator 
        of the rms using the 90th percentile of the rms. Changing this to a 
        less robust rms estimator can result in less components being used 
        since rms is a mean based statistic which is NOT robust against 
        outliers.

        Parameters
        ----------
        spectra : numpy.ndarray
            The spectra to calculate the rms from
        pca_use_mask_and_errors : numpy.ndarray
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.
        on_sky : bool, optional
            Whether the rms is calculated using sky pixels (pixels to be
            reconstructed) or using non-sky pixels. The default is True.

        Returns
        -------
        rms_per_specta : numpy.ndarray
            The rms for each spectra.

        """

        rms_mask = self.where_on_or_off_sky(np.ones_like(spectra), on_sky)

        #if we have determined rms has been met, no need to recheck so will
        #mask it out
        # if spectra_to_check is None:
        #     spectra_to_check = np.ones(np.shape(spectra)[0])
        unnormed = self.undo_normalisation(spectra, self.poisson_noise)

        #(self.skylines == on_sky) & (self.wavelength_mask == 1)

        rms_per_specta = self.rms_function(unnormed, rms_mask, pca_use_mask_and_errors)

        #figure out what to do with rms=np.nan for where spectra
        #are all bad/not included
        return rms_per_specta

    def rms_check(self, rms_to_beat, current_rms, spectra_mask_high_rms):
        """Finds if the specta at skyline positions minus the reconstructed sky 
        subtraction residuals has an rms lower than the spectra away from
        skyline positions. If it does have a better rms, that position
        is set to False to discontinue reconstruction and stop in 
        `spectra_mask_high_rms`.

        Parameters
        ----------
        rms_to_beat : numpy.1darray
            The rms per spectra that needs to be beat for reconstruction to 
            stop.
        current_rms : numpy.1darray
            The current rms of the reconstructed pixels of the spectra.
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra

        Returns
        -------
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra
        rms_still_worse : numpy.ndarray of bool
            True if the current rms (`current_rms`). is still not good enough,
            False if the rms has beaten the rms it needs to (`rms_to_beat`).

        """
        #True are the positions/spectra that still need components.
        #False are positions that are good enough
        rms_to_beat = rms_to_beat[spectra_mask_high_rms]
        change_in_rms = rms_to_beat - current_rms
        rms_still_worse = change_in_rms<=0 #True when worse, False when better
        #replacing the Trues and Falses. True means continue adding components
        # spectra_mask_high_rms[spectra_mask_high_rms] = rms_not_better   

        return rms_still_worse

    def set_spectra(self, spectra, pca_subtracted_spectra, rms_to_beat, current_rms, spectra_mask_high_rms):
        """Where spectra now no longer need an additional component (since the 
        rms is low enough), method adds the data-recon_nth to the final
        output array `pca_subtracted_spectra`, and updates the boolean mask that is
        keeping track of which spectra still have too large rms's (too high
        rms mean we keep trying so appear as True in `spectra_mask_high_rms`).

        Parameters
        ----------
        spectra : numpy.ndarray
            The spectra that have had the 
        pca_subtracted_spectra : numpy.ndarray
            Spectra that have the reconstructed spectra that have been
            optimally found based of the improvement in rms subtracted from
            the origonal spectra
        rms_to_beat : numpy.ndarray
            The rms per spectra that needs to be beat for reconstruction to 
            stop.
        current_rms : : numpy.1darray
            The current rms of the reconstructed pixels of the spectra.
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra

        Returns
        -------
        pca_subtracted_spectra : numpy.ndarray
            Spectra that have the reconstructed spectra that have been
            optimally found based of the improvement in rms subtracted from
            the origonal spectra
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra

        """

        rms_still_worse = self.rms_check(rms_to_beat, current_rms, spectra_mask_high_rms)
        
        #pca_subtracted_spectra is full sized, spectra only has the spectra that need
        #reconing. Therefore there are two masks since the arrays are
        #different lengths
        
        # if np.all(~rms_still_worse): # all the rms is better
        #     #so replace all the True locations (which are not longer worse)
        #     pca_subtracted_spectra[spectra_mask_high_rms] = spectra
        #     #making them all False (do not continue)
        #     spectra_mask_high_rms[spectra_mask_high_rms] = rms_still_worse 
        # else:

        #replace all the previous True (worse) where now spectra are better 
        inds1 = np.where(spectra_mask_high_rms)[0]

        pca_subtracted_spectra[inds1[~rms_still_worse]] = spectra[~rms_still_worse]
        # pca_subtracted_spectra[spectra_mask_high_rms][~rms_still_worse] = spectra[~rms_still_worse] 
        
        #Now all the places where we still needed to check are replaced with the new values
        spectra_mask_high_rms[spectra_mask_high_rms] = rms_still_worse 
        
        return pca_subtracted_spectra, spectra_mask_high_rms

    def undo_preprocessing(self, spectra, poisson_noise=None, continuum=None):
        """Wrapper to undo all the preprocessing needed before running the
        sky residual subtraction. In order, this is removing the Poisson noise
        normalisation, the re-adding the continuum to the data. 

        Parameters
        ----------
        spectra : numpy.ndarray
            The specra with preprocessing steps applied that need to be 
            removed.
        poisson_noise : numpy.ndarray, optional
            The Poisson noise used to normalise the spectra. The default is 
            None.
        continuum : numpy.ndarray, optional
            The continuum that was removed from the spectra. The default is 
            None.

        Returns
        -------
        usuable_spectra : numpy.ndarray
            Spectra that has had the preprocessing removed.

        """
        if poisson_noise is None:
            poisson_noise = self.poisson_noise

        unnormed_spectra = self.undo_normalisation(spectra, poisson_noise)

        if continuum is None:
            continuum = self.continuum

        usuable_spectra = self.undo_continuum(unnormed_spectra, continuum)

        return usuable_spectra

    def undo_continuum(self, unnormed_spectra, continuum=None):
        """Adds the continuum back to the spectra

        Parameters
        ----------
        unnormed_spectra : numpy.ndarray
            Spectra that has no normalisation applied and needs its continuum
            added back.
        continuum : numpy.ndarray, optional
            The continuum to add back to the spectra. The default is None.

        Raises
        ------
        TypeError
            If no continuum is provided, (i.e, `continuum=None`), an error is
            raised.

        Returns
        -------
        continuum_added : numpy.ndarray
            Spectra with the continuum added back.

        """

        if continuum is None:
            raise TypeError('No continuum has been provided. If no'\
                            'continuum is present, use an array containing'\
                            'zeros.')

        continuum_added = unnormed_spectra + continuum

        return continuum_added

    def undo_normalisation(self, spectra, poisson_noise=None):
        """Reverses the normalisation by multiplying by the Poisson noise

        Parameters
        ----------
        spectra :  numpy.ndarray
            Spectra that has normalisation applied and needs this to be undone.
        poisson_noise : numpy.ndarray, optional
            The Poisson noise used to normalise the data. The default is None.

        Raises
        ------
        TypeError
            If no noise array is provided, (i.e, `poisson_noise=None`), an 
            error is raised.

        Returns
        -------
        unnorm : numpy.ndarray
            The spectra.

        """
        if poisson_noise is None:
            raise TypeError('No Poisson noise has been provided')

        unnorm = spectra * poisson_noise

        return unnorm

    def get_origonal_rms_non_sky(self, pca_use_mask_and_errors=None):
        """Gets the baseline rms that the spectra have to beat to stop
        the reconstruction.

        Parameters
        ----------
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values. The default is None.

        Returns
        -------
        rms_per_specta : numpy.1darray
            The rms value for each spectra calculated using the part of the
            spectra that isn't used.

        """
        rms_per_specta = self._get_rms(self.spectra, pca_use_mask_and_errors, on_sky=False)

        return rms_per_specta

    # def get_origonal_rms_sky(self, rms_percentile, pca_use_mask_and_errors=None):
    #     # poisson_noise = self.poisson_noise if poisson_noise is None else poisson_noise

    #     rms_per_specta = self._get_rms(self.spectra, pca_use_mask_and_errors, on_sky=True)

    #     return rms_per_specta


    def _projection_eigen_sys(self):
        """Convenience function to grab out the eigen system. 
        !!!NOTE: ASSUMES PREVIOUS SDSS CONVENTIONS ARE BEING USED, SO THE PCA 
        LIBRARY IS ASSUMEDTO BE A DICTIONARY WITH THE FOLLOWING KEYS AND
        MEANINGS:
            'U'    : These are the eigen vectors;
            'm'    : This is the mean array spectrum;
            'W'    : These are the eigen values; 
            'vqu'  : These are the iterative values that are constrained
                     to fine the PCA solution;
            'sig2' : The robust scale (i.e., standard deviation) values.


        Returns
        -------
        mean_spectra : numpy.1darray
            The mean spectra.
        eigen_vectors : numpy.ndarray
            The eigen vectors from a sky library of eigen spectra.

        """

        mean_spectra = self.pca_library['m']
        eigen_vectors = self.pca_library['U']

        return mean_spectra, eigen_vectors


    def _projection_best_prep(self, pca_use_mask_and_errors=None):
        """Initialises the arrays and starting rms values for the iterative
        method for removing the sky subtraction residuals

        Parameters
        ----------
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values. The default is None.

        Returns
        -------
        spectra : numpy.ndarray
            The spectra that needs pca detectable residuals removed.
        pca_use_mask_and_errors : numpy.ndarray
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.
        rms_to_beat : numpy.ndarray
            The rms per spectra that needs to be beat for reconstruction to 
            stop.

        """
        spectra, pca_use_mask_and_errors = self._projection_prep(pca_use_mask_and_errors)

        # if rms_perc is None:
        #     #default kwarg for `rms_percentile` is 90
        #     rms_perc = self.pca_kwargs.pop('rms_percentile', recon_kwargs['rms_percentile'])

        # rms_sky_start = self.get_origonal_rms_sky(rms_perc, error_spectra)
        rms_to_beat = self.get_origonal_rms_non_sky(pca_use_mask_and_errors)

        # return spectra, pca_use_mask_and_errors, rms_sky_start, rms_to_beat
        return spectra, pca_use_mask_and_errors, rms_to_beat

    def _projection_prep(self, pca_use_mask_and_errors=None):
        """Initialises the base arrays to perform the sky subtraction residual
        reconstruction. Basically, if no user inputted array containing 
        pixels to mask (as 0's), and the noise error map everywhere else
        is provided, uses the attribute where these values show exist.
        This allows some flexibility in the pipeline, but most, if not all
        users will not need nor want to used this flexibility.

        Parameters
        ----------
        pca_use_mask_and_errors : numpy.ndarray or None, optional
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values. The default is None.

        Returns
        -------
        spectra : numpy.ndarray
            The spectra that needs pca detectable residuals removed.
        pca_use_mask_and_errors : numpy.ndarray
            Array where all bad/unusable pixels are set to zero so that they
            can be ignored later on, and the remaining pixels are set to the 
            noise error array values.

        """

        if pca_use_mask_and_errors is None:
            pca_use_mask_and_errors = self.pca_use_mask_and_errors

        spectra = self.spectra

        #Ignore outliers indicated in the error spectra
        # pca_use_mask_and_errors[error_spectra==0] = 0
        # pca_use_mask_and_errors[pca_use_mask_and_errors !=0] = error_spectra[pca_use_mask_and_errors !=0]

        return spectra, pca_use_mask_and_errors


    def _projection_best_array_init(self, spectra, max_comp):
        """Initalised the empty arrays for the analysis results to go into.

        Parameters
        ----------
        spectra : numpy.ndarray
            The spectra that needs pca detectable residuals removed.
        max_comp : int
            The maximum number of components to use when reconstructing spectra.

        Returns
        -------
        spectra_mask_high_rms : numpy.1darray of bool
            True if using the spectra. False if skipping the spectra
        reconstructed_spectra : numpy.ndarray
            Array for spectra with the reconstructed spectra removed from it.
        n_comp_used : numpy.1darray
            Empty array where the number of components needed for the
            reconstruction for each spectra will be inputted.
        rms_changes : numpy.ndarray
            Empty array for the rms per nth component will be stored.

        """
        n_spec, no_pixel = np.shape(spectra)
        
        #True when high, False when good enough
        spectra_mask_high_rms = np.ones(n_spec, dtype=bool)
        reconstructed_spectra = np.zeros((n_spec, no_pixel), dtype=float)
        n_comp_used = np.zeros(n_spec, dtype=int)

        rms_changes = np.ones([max_comp+1, n_spec])

        return spectra_mask_high_rms, reconstructed_spectra, n_comp_used, rms_changes

def main():
    pass

if __name__ == "__main__":
    main()
