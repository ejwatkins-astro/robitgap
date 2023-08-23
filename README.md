# Introduction

## Why PCA

When sky is subtracted from spectra, any slight misalignment of the flux in the pixels results in correlated subtraction residuals, especially in the red part of the optical spectrum, where OH forest lines are present. These look like spikes both in the positive and negative direction. Therefore to improve the quality of the spectra, methods that can identify correlated features and reconstruct them to subtract the residuals away are good. PCA is perfect for this.

## Core Algorithms

### Robust PCA

roitgap stands for RObust ITerative GAPppy PCA. It is a pca routine developed by Tamas Budavari and Vivienne Wild that uses robust statistics to reduce the impact of outliers on the principle components. This is because the normative pca uses the mean, standard deviation and the covariance matrix, which are easily skewed with one bad outlier, so robust estimators of these statistics are used instead. It does this by building up the pca solution iteratively by streaming in 1 spectrum at a time, where the spectra are given to the method in a random order. For the maths behind this technique, see Budavari et al. 2007.

### Data gaps: gappy reconstruction

PCA is a Euclidean method. Therefore the "distance" between the variables matter. If data is missing due to it containing corrupt values, or science lines that we need to ignore, (which will influence the reconstruction), PCA perceives this as a larger distance. and it can impact the reconstruction. To avoid this, a routine was developed that fills the gaps using the rest of the spectrum and the eigen system to reconstruct/predict the data there first. The routine is based off of Connolly & Szalay (1999, AJ, 117, 2052) but the exact maths behind the routine here are not published (the paper, Lemson \[in prep\] where the technique is outlined, has been in prep since 2008). The python script containing the gappy method has outlined the mathematical description explaining the routine if needed.)  The disadvantage of doing this is that the components are no longer fully orthogonal, but this is a minor issue with little impact of the pca.

### Differences between IDL and python implementation here

The robust pca and gappy routines have been coded into discrete functions and have been corrected for any slight bugs that were found, and have been written to run in a vectorised way, speeding up the run time.

## Implementation

The core algorithms have been written into the three python scripts "*robustpca.py*" (a wrapper around the a core robust pca method) "*core.py*" (the core pca method for a single iteration) "*gappyrecon.py*" (data reconstruction that deals with data gaps)

The actual application of these three core scripts requires a pipeline to pre-process the data, and methods for determining how many components to construct with. This has been created in the program "*pcapipeline.py*"

## Necessary pre-pre-processing: "*pcapipeline.py*"

### Summary of requirements

1.  Continuum removal
2.  Poisson noise normalisation
3.  Skyline identification
4.  Science line identification and masking
5.  Outlier spectra identification and masking

Before PCA can be run on spectra to generate the pca sky library, and before the pca library can be used to remove sky subtraction residuals, it needs to be pre-processed so that the only correlated/repeating signals in the data are sky subtraction residuals. Therefore any features present in the data, such as object continuum, science lines and correlated noise features that are not sky subtraction residuals need to be remove and masked respectively (otherwise it will be picked up as a component by the PCA). In addition, the pca should only be run on the parts of the spectrum that need correcting, namely where there are skylines. Therefore a skyline mask indicating where these are needs to be generated and the PCA ran on only those pixels. Finally, while the pca routine is robust, any obviously identifiable bad sky spectra need to be ignored which will help the robust PCA converge to the correct principle components.

### 1\. Continuum removal

Any object continuum in either the sky subtracted sky spectra, or the sky subtracted science spectra need to be removed and flat spectra flat about zero. Any continuum features remaining will be identified by the PCA and will result in poor reconstructions and poor residuals removal.

### 2\. Poisson noise normalisation

The spectra need to be normalised by their Poisson noise. This is because Poisson noise is proportional to the incoming photon counts (i.e the flux). PCA will detect this correlation (i.e more flux equals more bigger noise) and so it will weight these large flux values higher in priority when generating the principle components, resulting in a biased reconstruction since strong skylines will have more flux from the Poisson noise. To eliminate this, the sky subtracted spectra need to be normalised by their Poisson noise. For previous SDSS, this was done using the median noise array from sky spectra for each plate (this is because the noise here is a good representation of the Poisson noise induced by the skylines ONLY. In science spectra noise arrays, there will be more Poisson noise in pixels containing science flux, which we do not want to correct for. But with Skycorr, a model of the sky is subtracted and there is NO sky spectra and NO associated noise exist that can be used to estimate the Poisson noise induced by just the skylines (the noise array of the science spectra will contain Poisson noise also from the science objects). Therefore, we are going to estimate the Poisson noise using the sky model(s) generated for the sky spectra, and the Poisson noise using sky model(s) generated for the science spectra. Since Poisson noise is proportional to the square root of the flux, we estimate the Poisson noise caused by the skylines is sqrt(sky model).

### 3\. Skyline identification

PCA is only run and generated on the pixels that contain skylines. Since PCA is a Euclidean process, one mask where all the skylines may be needs to be created or supplied. Where skylines may be are called "sky" pixels. IMPORTANT. While PCA is only run on positions masked to cover skylines, "non-sky" pixels are needed in the pca subtraction to find the optimal reconstruction, so while the pca library generation can be done on a cut part of the spectrum containing just sky lines, the mask used will be needed for the application to the science data.

### 4\. Science line identification and masking

Science line are correlated and repeated features found within spectra. If any are found in either the sky or the science spectra, they need to be masked to ensure they are not identified as a component. If any persist, they can cause actual science flux to be lost. An algorithm needs to be written for the identification of science lines. These positions are analysed using the gappy reconstruction, so if a science line overlaps with a skyline, the skyline residual there will still be reconstructed and removed.

### 5\. Outlier spectra identification and masking

While the PCA method is robust, obviously identifiable outlier spectra should be ignored before streaming them into the PCA library which will help the routine converge and minimise the chance of a bad representation of the system occurring. Bad spectra includes spectra who's average flux significantly deviates from zero, have large standard deviations, and who's "spectral colour" does not match expected values.

NOTE: Individual outliers are not masked as a lot of these will be sky subtraction residuals, which we need to keep in for the pca routine. So outlier features are never masked (except if they have known bad values from corruption, or science lines)

## What are the inputs?

To generate a pca library using skycorr, the fully calibrated sky subtract sky spectra are needed (i.e. the finalised usable data with just sky subtraction residuals remaining) and the sky model(s) used to produce the sky subtract sky spectra.

When applying the pca library, the sky subtracted science data, the sky model used to subtract the sky, and the noise arrays are needed.

### Optional inputs depending

TODO

# Pipeline

A pipeline has been written to wrap the robust pca. The pre-processing steps, and the application of robust pca and gappy reconstruction into one package.

# How to use and requirements

## PCA library generation

To run the pca library generation part of the pipeline, you need to provide a mixture of datasets and pre-processing routines, which are:

1.  The sky subtracted sky data;
2.  1D Mask of where skylines maybe located;
3.  A way to subtract continuum from the spectra (This will be a function that you'll provide along with its required kwargs);
4.  Data for estimating and normalising away the Poisson noise (For skycorr this is the square root of the skymodel that has been generated and subtracted from the sky spectra, with the sky model generated for the sky spectra will be the sky model. Turning the model into a Poisson noise estimate should be performed using a function you provide for the pipeline);
5.  A way of identifying object/science lines (This will be a function that you'll provide along with its required kwargs);
6.  A way of identifying spectra that are statically bad to makes them out (This will be a function that you'll provide along with its required kwargs).
7.  The wavelength range to perform the PCA on. Optional, but basically only the red part of the spectrum should have PCA applied. The blue part should be fine.

These data are entered in the variables, which are:

```python
    sky_spectra=,        #the sky subtracted sky spectra; 
    noise_gen_data=,     #the data that will normalise away Poisson noise;
    skyline_mask=None,   #the 1d mask containing 1's where skyines can be. OPTIONAL;
    error_spectra=None,      #a mask of equal size where bad pixels are indicated using 0's. The rest of the values are 1's, or can just be the noise array values from the sky data. !!!WARNING: NEVER use NaNs. Please set NaNs in the data as zero, and indicate the data is bad using this error array. Also if there are any infs present, do the same. Matrix operations fail if NaNs or infs are present. OPTIONAL;
    noise_gen_is_error=False, #Boolean. If the error_spectra contains the bad pixel locations and is identical to the data needed to estimate the Poisson noisem (False for LVM-skycorr) set this to True. OPTIONAL; 
    wavelength=None,      #The wavelength values of the x axis;
    wavelength_range=None,#A 2-list of the wavelengths to run between, for example [6700, 9180] # \AA
```

## Required Setup: pre-processing functions - function creation rules and example

* * *

The pre-processing pipeline was written as a wrapper around analysis functions that a user provides since this allows it to be used outside of LVM functionality, and also, the exact functions for the pre-processing haven't been discussed, so I didn't know what to use. Therefore it will use whatever functions you give providing that it follows these rules.

## Pre-processing functions

### How to provide the functions

===============================
The required functions for the pre-processing steps are entered as a dictionary with attribute name `subtraction_methods` under the labels:
'continuum' : `your_cont_func`
'normalise' : `your_norm_func`
'skylines' : `your_skyline_func` #if skylines are known, make this a dummy but I believe the function is not run if skylines are provided, so should be able to use None, else do: def your\_skyline\_func(skyline\_mask): return skyline\_mask
'science_lines' : `your_scienceline_identification_func`
'outliers' : `your_outlier_spectrum_identification_func`
...

### Variables

============
The first variable should always be a required variable using the variable name `spectra`. Even if spectra is never used, defined it anyway. When making these functions, if they use any variables that the class object already uses/has as attributes, use those names in the function. These are:

```python
'noise_gen_data',
'error_spectra',
'observation_numbers',
'skyline_mask',
'science_line_mask'
```

and they should be defined at at the top of your script as:

```python
VARIABLE_NAMES = [
    'noise_gen_data', 
    'error_spectra', 
    'observation_numbers', 
    'skyline_mask', 
    'science_line_mask'
]
```

These can be imported from the pca config file however:

```python
import config_robitgapPCA as config

VARIABLE_NAMES = config.variable_names
```

Some if not all of these variables are automatically entered into the function by the pipeline (less variables are entered for earlier pre-processing methods since those variables do not exist yet), and if not defined in the function, are shoved into a kwarg. To ignore automatically unneeded variables, always add these lines to the top of the function:

```python
kwargs_from_pcaPrep = {}
for var in VARIABLE_NAMES:
    kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
```

...

### Returned variables required

==========
Each method needs to return a specific set of processed data products. These are:

***The continuum function***

Return in this order:

1.  the continuum,
2.  the continuum subtracted spectra

***The Poisson normalisation function***

Return in this order:

1.  Normalised data,
2.  the Poisson estimate.
3.  Optional. Can also return a third variable which is if the noise/error array is the same as the Poisson noise as a bool (True if same, False if different. For LVM and skycorr, this is False. The default of the pipeline is also set to False)

***The skyline identification function***

- Return a 1d mask of where there are skylines. !!!WARNING: The skyline mask needs to show where all possible skylines might be and needs to be the exact same mask used when removing skyline subtraction residuals from the science spectra

***The science line identification function***

- Return a 2d mask where science features are marked using 0's, and everything is 1's
- **The outlier identification function**
    Return a 1d mask of length equal to the number of sky spectra provided where outlier spectra are marked with 0's, and good spectra are marked using 1'.
    ...

### Additional arguments for the function

Finally, if any function you define requires additional arguments, they are provided when initialising the prep class using the attribute `method_kwargs` They are entered in a dictionary, as a dictionary. So if for example outlier finding needs tuning parameters, they are listed as a dictionary:

```python
    outliers_kwargs = {
        'mean_limit' : 0.3,
        'std_limit'  : 0.9
        }
```

And are then placed inside another dictionary with the keys corresponding to the pre-processing method name:

```python
    method_kwargs = {
        #'continuum'     : cont_kwargs,
        # 'normalise'     : norm_kwargs,
        # 'skylines'      : skyline_kwargs,
        # 'science_lines' : scienceline_kwargs,
         'outliers'      : outliers_kwargs
    }
```

...

# Calling the pipeline

The pre-processing pipeline can be run by itself, but future parts of the pipeline automatically run the pre-processing and then run the pca library generation in one step. I recommend starting at the library generation stage and automatically run the pre-processing using the class methods. Therefore, run
`lib = PCApipe.pcaLibrary.run_pcaPrep("Variables here")`

This above line will run the pre-processing, enter into another class method automatically (`from_pcaPrep`), which then initialises the pca libray generation class `pcaLibrary`.
From here, the pca library is automatically generated in the attribute `output`
`pca_library = lib.output`

## Script example from pre-processing to skylibrary generation

=================
Altogether an example might look like (only showing two example functions):

```python
import numpy as np
import pcapipeline as ppline
import configpca as config

VARIABLE_NAMES = config.variable_names
    
def normalise_lvm(spectra, noise_gen_data,  noise_is_poisson=False, **kwargs):

    kwargs_from_pcaPrep = {}
    for var in variable_names:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
        
    poisson_estimate = np.sqrt(noise_gen_data)
    normalised = spectra/poisson_estimate
    
    return normalised, poisson_estimate, noise_is_poisson

def skycorr_outlier_spectra(spectra, skyline_mask=None, science_line_mask=None, mean_limit=0.2, std_limit=0.8, **kwargs):

    kwargs_from_pcaPrep = {}
    for var in VARIABLE_NAMES:
        kwargs_from_pcaPrep[var] = kwargs.pop(var, None)
        
    spectra_to_use = np.ones(np.shape(spectra)[0])
    
    #Should just be ones and zeros. For skylines, 1 indicates there
    #is a skyline there. For sciencelines, 1 indicates there is NO scienceline
    #(i.e., the data can be used as there is no sciene line there)
    
    spectra_sky_science_ignore = np.copy(spectra)
    
    spectra_sky_science_ignore[:, skyline_mask==0] = np.nan
    spectra_sky_science_ignore[science_line_mask==0] = np.nan
    
    spectra_mean = np.nanmean(spectra_sky_science_ignore, axis=1)
    spectra_std = np.nanstd(spectra_sky_science_ignore, axis=1)
    spectra_to_use[np.abs(spectra_mean)>mean_limit] = 0
    spectra_to_use[spectra_std>std_limit] = 0
    
    return spectra_to_use

### Add other required functions

preprocessing_methods = {
    'continuum'     : your_cont_func, 
    'normalise'     : normalise_lvm,
    'skylines'      : your_skyline_func,
    'science_lines' : your_scienceline_identification_func,
    'outliers'      : skycorr_outlier_spectra
}

outliers_kwargs = {
    'mean_limit'  : 0.3,
    'std_limit'   : 0.9
    }

method_kwargs = {
    'outliers' : outliers_kwargs
}

#Load in data, skycorr
sky_spectra = #For bad values, do not use NaN, just set to zero and indicate where bad in `sky_noise_arrays_and_bad_pixels`
skyline_positions =
skycorr_models = 
sky_noise_arrays_and_bad_pixels = # marks where there are bad pixels using a bitmap. (0's bad, 1 good)

#runs the pre-processing and then the pca generation 
lib = ppline.pcaLibrary.run_pcaPrep(
    sky_spectra, 
    noise_gen_data=skycorr_models, 
    subtraction_methods=preprocessing_methods, 
    method_kwargs=method_kwargs,
    skyline_mask=skyline_positions, #1d mask of 1's where sky, 0 where not sky
    error_spectra=sky_noise_arrays_and_bad_pixels # 0 for bad pixels. Do not need error spectra, so remaining values can be set to 1
    wavelength=None, 
    wavelength_range=None,
) 
pca_library = lib.output # can save this and load in for future
```

# Performing pca reconstruction and sky subtraction residual removal

## Summary

This is simple to run once you have a pca library, the pre-processing functions and some sky subtracted science spectra using the class `pcaSubtraction`. It can be initialised using a class method that automatically runs the pre-processing and enters the result into the class (via a different class method that takes a pcaPrep object and pulls out pcaPrep attributes to initialise the class). The format is nearly identical to the sky library generation, and the only difference is that we need to provide the pca library.

## What are the inputs?

It needs science spectra, the pre-processing functions, the Poisson noise, the noise arrays (errors of each spectra), where the skylines are and the pca library. Initialised therefore looks like this:

```python
sub = ppline.pcaSubtraction.run_pcaPrep(
    science_spectra,
    noise_gen_data=sky_model_for_science, 
    subtraction_methods=preprocessing_methods, 
    method_kwargs=method_kwargs,
    skyline_mask=skyline_positions,
    error_spectra=error_spectra_for_science,
    pca_library = pca_library,
    wavelength=None, 
    wavelength_range=None
)
```

## Running the subtraction

This does not run automatically, since we can get the reconstructed spectra using two different methods, though one is only really used for testing and a quick look. So to perform the skyline residual removal, finally run,

`residual_removed_science_spectra = sub.run(undo_preprocessing=True)` 

and the residuals will be removed. The optional variable, `undo_preprocessing`, un-normalised the data and then adds back on the continuum. If you only want to undo the normalisation, run with `undo_preprocessing` set to False, and after pass `residual_removed_science_spectra` into the method `undo_normalisation` :

```python
residual_removed_science_spectra = sub.run()
final_spectra = sub.undo_normalisation(residual_removed_science_spectra)
```

This reconstructs using the number of spectra needed to lower the rms in the affected, and being corrected sky pixels to below the rms of the non sky pixels, which should have a lower rms since there are no residuals there. For each iteration, a new component is added, then the rms compared after subtracting the reconstructed spectra away from the sky pixels in the science spectra. Once the rms is equal to the non-sky pixels, the reconstruction halts and no more components are added.

NOTE: `sub.run()` is performed in a vectorised way (every single spectra is reconstructed with ***n*** number of components in one matrix operation, which can use a lot of memory at once when reconstruction with a lot of components (n=500 for example). You can either use a slightly better pc, or just provide a smaller number of science spectra at a time and loop. 

## Script example from skylibrary generation to the final science spectra with subtraction residuals removed

=================
Altogether an example might look like (only showing two example functions):

```python
# can save and load in
pca_library = lib.output

#can use the same `preprocessing_methods` ones as before
#preprocessing_methods = {...

cont_kwargs = {
    'filter_size' : 31
    }

outliers_kwargs = {
    'mean_limit'  : 0.3,
    'std_limit'   : 0.9
    }

method_kwargs = {
    'continuum' : cont_kwargs,
    'outliers' : outliers_kwargs
}

#Load in data...
science_spectra = #For bad values, do not use NaN, just set to zero and indicate where bad in `sky_noise_arrays_and_bad_pixels`
skyline_positions = # NEEDS to be identical to mask used for pca library generation
sky_model_for_science = 
error_spectra_for_science= # NEED the error values  marks where there are bad pixels using a bitmap. (0's bad, 1 good)

sub = ppline.pcaSubtraction.run_pcaPrep(
    science_spectra,
    noise_gen_data=sky_model_for_science, 
    subtraction_methods=preprocessing_methods, 
    method_kwargs=method_kwargs,
    skyline_mask=skyline_positions,
    error_spectra=error_spectra_for_science,
    pca_library = pca_library,
    wavelength=None, 
    wavelength_range=None
)

final_science_spectra, n_comp_needed, rms_changes = sub.run(undo_preprocessing=True, return_extra_info=True)
```

# Provided script

This script "exampling\_testing\_pipeline.py" tests if the pca pipeline works, and provides a working example of the pipeline being set-up and used in full, with some additional options not discussed here being used (these optional mainly help with the fake data aspect). It also generates some plots to show what is going on.

Preferably, real sky data and science data should be used. For now, this script uses fake data that is made in the function `make_fake_data` imported from fakedatalvm.py to test the pipeline.

!!!WARNING if the fake data doesn't have correlated features, gappyrecon (the pca reconstruction that fills missing data with a reconstructed guess), fails with a error with a tensordot. I believe this occurs since random noise does not have correlated features, so it couldn't reconstruct and find a solution.