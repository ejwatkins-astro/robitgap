# -*- coding: utf-8 -*-
"""
@author: Elizabeth Watkins

Basically contains masking functions
"""

import numpy as np
import copy
import warnings

def make_pca_use_mask_and_errors(science_lines, error_spectra, bad_specta_mask):
    """Creates an array where all bad/unusable pixels are set to zero so that they
    can be ignored later on, and the remaining pixels are set to the noise 
    error array values. It will be the same shape as `error_spectra`.

    Parameters
    ----------
    science_lines : numpy.ndarray
        Mask of where there are science lines to ignore. These should be
        0 for ignore (since they are a science line) and 1 to use (i.e., not
        a science line.
    error_spectra : numpy.ndarray
        Array containing the noise error array of the data.
    bad_specta_mask : numpy.1darray
        1d array for the positions of entire spectra that are bad (i.e., 
        statistically determined to be an outlier).

    Returns
    -------
    pca_use_mask_and_errors : numpy.ndarray
        Just contains the error array with all bad or unusable pixels
        masked out as 0's.

    """
    #zeros are where we ignore and are dealt with using the gappy PCA

    #zeros mean we skip for pca generation and RMS checks
    pca_use_mask_and_errors = np.ones_like(science_lines)

    to_use_mask = (np.isfinite(science_lines)) & (science_lines != 0)
    if error_spectra is not None:
        pca_use_mask_and_errors = error_spectra

    # to_use_mask = (to_use_mask) & (np.isfinite(error_spectra)) & (error_spectra != 0)
    #
    #     pca_use_mask_and_errors[to_use_mask] = error_spectra[to_use_mask]
    # else:
    #     pca_use_mask_and_errors[to_use_mask] = 1 #ones mean we use these values
    pca_use_mask_and_errors[~to_use_mask] = 0

    #`bad_specta_mask` can be 1d or 2d?
    pca_use_mask_and_errors[bad_specta_mask==0] = 0
    return pca_use_mask_and_errors

def get_wavelength_mask_for_spectra(spectra, wavelength, wavelength_range):
    """Takes in a wavelength array and the upper and lower wavelength limits
    and provides a same-length boolean mask True where the wavelength
    values are within the upper and lower limits given

    Parameters
    ----------
    spectra : numpy.ndarray
        The data that the extent will be masked.
    wavelength : numpy.1darray
        The values along the x axis (rows) of `spectra`.
    wavelength_range : 2-list like array
        The range of the data to use

    Returns
    -------
    wavelength_mask : numpy.ndarray
        The mask that limits the range of the data shown.

    """

    if wavelength is None or wavelength_range is None:
        wavelength_mask = np.ones(np.shape(spectra)[1], dtype=bool)
    else:
        l, u = wavelength_range
        wavelength_mask = (wavelength >= l) & (wavelength <= u)

    return wavelength_mask

def mask_arrays_columns(mask):
    """Decorator for functions that have multiple data of the same size that
    all need to have the same mask applied to them

    Parameters
    ----------
    mask : numpy.ndarray
        Mask that will be applied to the data given.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if mask is None:
                return func(*args, **kwargs)
            args, kwargs = apply_mask_to_column(mask, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def apply_mask_to_column(mask, *args, **kwargs):
    """Masks along the column axis.
    Assumes column is the last index position.
    Takes in the variables to be masked as args and kwargs

    Parameters
    ----------
    mask : numpy.1darray
        mask, where 1's are to keep and use, and zeros are to be ignored.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    args : TYPE
        DESCRIPTION.
    kwargs : TYPE
        DESCRIPTION.

    """
    mask = mask.astype(bool)

    for key in kwargs.keys():
        kwarg = kwargs[key]
        kwargs[key] = apply_1d_mask_to_2dcol(mask, kwarg)[0]

    args = tuple(apply_1d_mask_to_2dcol(mask, *args))
    # args = apply_1d_mask_to_2dcol(mask, *args)

    return args, kwargs

def apply_1d_mask_to_2dcol(mask, *args):
    """Masks entire columns in a 2d array assuming the provided 1d mask of
    booleans corresponds the the columns to keep

    Parameters
    ----------
    mask : TYPE
        DESCRIPTION.
    arrays_to_be_masked : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if np.all((mask==1) | (mask==True)):
        return args

    masked = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            sh = np.shape(arg)
            if len(sh) > 0 and sh[-1] == np.shape(mask)[0]:
                mask = mask.astype(bool)
                masked.append(arg.T[mask].T)
            else:
                masked.append(arg)
        else:
            masked.append(arg)

    # masked = []
    # if arg in args:
    #     # if arg is None:
    #     #     masked.append(arg)

    #     masked.append(arg[:, mask])

    return masked

def check_make_2d(array_to_check):
    """Checks if the array is 2d and converts it to 2d if it is 1d

    Parameters
    ----------
    array_to_check : numpy.ndarray
        Array that needs to be made 2d if not already.

    Returns
    -------
    array_to_check : numpy.array
        Array that is 2d.

    """
    if len(np.shape(array_to_check)) == 1:
        array_to_check = array_to_check[None]
    return array_to_check

def apply_1d_mask_to_2drow(mask, *args):
    """Function takes in 2d arrays as args and applies `mask` to them. 

    Parameters
    ----------
    mask : numpy.1darray
        1d mask to apply to the rows of 2d arrays to remove them.
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if np.all((mask==1) | (mask)):
        return args

    masked = []
    for arg in args: #was if
        mask_bool = (mask == False) | (mask == 0)
        masked.append(copy.deepcopy(arg[~mask_bool]))

    return masked

def nan_inf_masking(spectra, error_array):
    
    is_bad = ~np.isfinite(spectra)  
    spectra[is_bad] = 0 
    if error_array is None:
        error_array = np.ones_like(spectra)
        
    error_array[is_bad] = 0
    
    return spectra, error_array

def check_and_make_data_finite(data, errors, verbose=True):
    """
    Checks if the data has any non finite values (nans and infs).
    If any non finite values exist, user is warned and PCA is still run
    by assuming the non finite values are bad data values. Bad data
    values should be indicated and propagated by setting the `errors` array
    to `0` at those locations.  `errors` is updated at these locations
    and the average values replaces the non finite values in the `data`
    array as a dummy value. The replacement values might impact the
    initial convergence of the iterative PCA so it is recommended the
    original data value be used if possible.

    Parameters
    ----------
    data : array_like
        The data
        Dimensions: [number of spectra, length of spectra]
    errors : None type or array_like
        Array containing the data errors, where zero indicates where the
        data is bad and needs replacing. If None, assumes all the data is
        valid.

    Returns
    -------
    data : array_like
        The data with non finite values replaced with the median pixel
        Dimensions: [number of spectra, length of spectra]
    errors : None type or array_like
        Array containing the data errors, where zero indicates where the
        data is bad and will be replaced. If None, assumes all the data is
        valid.

    """
    where_finite = np.isfinite(data)
    if not np.all(where_finite):
        if verbose:
            warnings.warn('Data has NaNs or infs. These values have been '\
                          'replaced with a dummy average value and their '\
                          'location tracked using the `errors` array. Bad '\
                          'values in `data` should always be indicated by '\
                          'using the `errors` array by setting these bad values '\
                          'to `0` in `errors`. Bad values are dealt with using '\
                          'the gappy routines automatically. It is recommended '\
                          'you do this step yourself beforehand to make sure '\
                          'this is what you intend and to have absolute '\
                          'knowledge which pixels have had the gappy routine '\
                          'applied to.')

        data, errors = replace_nonfinite_with_median(data, errors)

    return data, errors

def replace_nonfinite_with_median(data, errors=None):
    """
    Finds and replaces non finite values (infs, nans), with their median
    value and indicates these locations in the `error` array by setting
    those locations to zero.

    Parameters
    ----------
    data : array_like
        The data
        Dimensions: [number of spectra, length of spectra]
    errors : None type or array_like
        Array containing the data errors, where zero indicates where the
        data is bad and needs replacing. If None, assumes all the data is
        valid.

    Returns
    -------
    data : array_like
        The data with non finite values replaced with the median pixel
        Dimensions: [number of spectra, length of spectra]
    errors : None type or array_like
        Array containing the data errors, where zero indicates where the
        data is bad and will be replaced. If None, assumes all the data is
        valid.

    """
    where_finite = np.isfinite(data)
    if np.all(where_finite):
        return data, errors

    data_with_median_replacment = np.nanmedian(data, axis=0) * np.ones_like(data)
    data_with_median_replacment[where_finite] = data[where_finite]

    if errors is None:
        errors = np.zeros_like(data)

    errors[where_finite] = 1

    return data_with_median_replacment, errors

def main():
    pass

if __name__ == "__main__":
    main()

    