# -*- coding: utf-8 -*-
"""

@author: Elizabeth Watkins


Functions for making fake data to test that the PCA pipelines runs all the way
through and is working.
"""

import numpy as np
import numpy.random as random

def gaussian(x, mu, sig, A):
    return A * np.exp(-np.power((x - mu)/sig, 2.)/2)

def vectorised_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

    # Always use a negative shift, so that column_indices are valid.
    # Alternative: r %= A.shape[1]
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]
    
    result = A[rows, column_indices]
    
    return result

def make_fake_data(num_spec, num_pixels, cont_pow=-0.1, frac_errors_per_spec=0.05, fraction_shift=0.3, shift_amount=3, noise_std=0.2, subtract=False):
    
    sky_spectra= random.normal(0, noise_std, (num_spec, num_pixels))
    
    As = random.random(num_spec) * 0.5
    intensity = random.random([9, num_spec])
    sky_spectra[:,20:40] += gaussian(np.linspace(-5,5, 20), 0, 3, As[:,None])  * intensity[0, None].T
    
    sky_spectra[:,85:90] += gaussian(np.linspace(-2,2, 5), 0, 1, 5*As[:,None]) * intensity[7, None].T
    
    sky_spectra[:,100:105] += gaussian(np.linspace(-3,3, 5), 0, 2, 6*As[:,None]) * intensity[1, None].T
    
    sky_spectra[:,200:206] += gaussian(np.linspace(-2,2, 6), 0, 2, As[:,None]) * intensity[2, None].T
    sky_spectra[:,208:214] += gaussian(np.linspace(-3,3, 6), 0, 2, 1.3*As[:,None]) * intensity[2, None].T
    sky_spectra[:,216:222] += gaussian(np.linspace(-3,2, 6), 0, 2, As[:,None]) * intensity[2, None].T
    
    sky_spectra[:,300:305] += gaussian(np.linspace(-2,2, 5), 0, 1, -4*As[:,None]) * intensity[7, None].T
    
    sky_spectra[:,333:353] += gaussian(np.linspace(-2,2, 20), 0, 2, 1.1*As[:,None]) * intensity[3, None].T
    
    sky_spectra[:,400:420] += gaussian(np.linspace(-5,5, 20), 0, 3, -2*As[:,None]) * intensity[4, None].T
    sky_spectra[:,421:431] += gaussian(np.linspace(-3,3, 10), 0, 3, -3*As[:,None]) * intensity[4, None].T
    
    sky_spectra[:,453:463] += gaussian(np.linspace(-3,3, 10), 0, 2, 3.3*As[:,None]) * intensity[5, None].T

    sky_spectra[:,500:510] += gaussian(np.linspace(-2,-2, 10), 0, 1, 2.05*As[:,None]) * intensity[6, None].T
    sky_spectra[:,515:525] += gaussian(np.linspace(-2,-2, 10), 0, 1, 3.1*As[:,None]) * intensity[6, None].T
    sky_spectra[:,530:540] += gaussian(np.linspace(-2,-2, 10), 0, 1, 5*As[:,None]) * intensity[6, None].T
    sky_spectra[:,545:555] += gaussian(np.linspace(-2,-2, 10), 0, 1, 3.3*As[:,None]) * intensity[6, None].T
    sky_spectra[:,560:570] += gaussian(np.linspace(-2,-2, 10), 0, 1, 2*As[:,None]) * intensity[6, None].T
    
    sky_spectra[:,620:625] += gaussian(np.linspace(-2,2, 5), 0, 1, -6*As[:,None]) * intensity[7, None].T
    
    sky_spectra[:,670:674] += gaussian(np.linspace(-2,-2, 4), 0, 1, 2*1.1*As[:,None]) * intensity[8, None].T
    sky_spectra[:,678:682] += gaussian(np.linspace(-2,-2, 4), 0, 1, 2*1.5*As[:,None]) * intensity[8, None].T
    sky_spectra[:,686:691] += gaussian(np.linspace(-2,-2, 5), 0, 1, 2*2*As[:,None]) * intensity[8, None].T
    sky_spectra[:,695:700] += gaussian(np.linspace(-2,-2, 5), 0, 1, 2*3.2*As[:,None]) * intensity[8, None].T
    sky_spectra[:,703:709] += gaussian(np.linspace(-2,-2, 6), 0, 1, 2*4.5*As[:,None]) * intensity[8, None].T
    sky_spectra[:,711:718] += gaussian(np.linspace(-2,-2, 7), 0, 1, 2*6.2*As[:,None]) * intensity[8, None].T
    sky_spectra[:,721:727] += gaussian(np.linspace(-2,-2, 6), 0, 1, 2*4.7*As[:,None]) * intensity[8, None].T
    sky_spectra[:,731:736] += gaussian(np.linspace(-2,-2, 5), 0, 1, 2*3.5*As[:,None]) * intensity[8, None].T
    sky_spectra[:,740:745] += gaussian(np.linspace(-2,-2, 5), 0, 1, 2*2*As[:,None]) * intensity[8, None].T
    sky_spectra[:,750:754] += gaussian(np.linspace(-2,-2, 4), 0, 1, 2*1.6*As[:,None]) * intensity[8, None].T
    sky_spectra[:,758:762] += gaussian(np.linspace(-2,-2, 4), 0, 1, 2*1*As[:,None]) * intensity[8, None].T
    
    random_shift_numbers = random.random(num_spec)
    where_shift = np.zeros_like(random_shift_numbers)
    
    shift_frac = np.linspace(0,fraction_shift/2,shift_amount)
    for i,n in enumerate(shift_frac[::-1]):
        where_shift[random_shift_numbers<=n] = -(i+1)
        
    for i,n in enumerate(shift_frac):
        where_shift[random_shift_numbers>=(1-n)] = (i+1)
        

    bad_values = ~np.isfinite(sky_spectra)     
    sky_spectra[bad_values] = 0  
    sky_spectra = vectorised_roll(sky_spectra, where_shift.astype(int)) 
    
    #Do not want negative flux so we add the min value as a continuum
    sky_spectra += np.abs(np.min(sky_spectra)) +0.01
           
    skyline_mask_1d = np.zeros(num_pixels)
    skyline_mask_1d[20-shift_amount:40+shift_amount] = 1
    
    skyline_mask_1d[85-shift_amount:90+shift_amount] = 1
    
    skyline_mask_1d[100-shift_amount:105+shift_amount] = 1
    skyline_mask_1d[200-shift_amount:206+shift_amount] = 1
    skyline_mask_1d[208-shift_amount:214+shift_amount] = 1
    skyline_mask_1d[216-shift_amount:222+shift_amount] = 1
    
    skyline_mask_1d[300-shift_amount:305+shift_amount] = 1
    
    skyline_mask_1d[333-shift_amount:353+shift_amount] = 1
    
    skyline_mask_1d[400-shift_amount:420+shift_amount] = 1 
    skyline_mask_1d[418-shift_amount:438+shift_amount] = 1
    
    skyline_mask_1d[453-shift_amount:463+shift_amount] = 1

    skyline_mask_1d[500-shift_amount:510+shift_amount] = 1
    skyline_mask_1d[515-shift_amount:525+shift_amount] = 1
    skyline_mask_1d[530-shift_amount:540+shift_amount] = 1
    skyline_mask_1d[545-shift_amount:555+shift_amount] = 1
    skyline_mask_1d[560-shift_amount:570+shift_amount] = 1
    
    skyline_mask_1d[620-shift_amount:625+shift_amount] = 1
    
    skyline_mask_1d[670-shift_amount:674+shift_amount] = 1
    skyline_mask_1d[678-shift_amount:682+shift_amount] = 1
    skyline_mask_1d[686-shift_amount:691+shift_amount] = 1
    skyline_mask_1d[695-shift_amount:700+shift_amount] = 1
    skyline_mask_1d[703-shift_amount:709+shift_amount] = 1
    skyline_mask_1d[711-shift_amount:718+shift_amount] = 1
    skyline_mask_1d[721-shift_amount:727+shift_amount] = 1
    skyline_mask_1d[731-shift_amount:736+shift_amount] = 1
    skyline_mask_1d[740-shift_amount:745+shift_amount] = 1
    skyline_mask_1d[750-shift_amount:754+shift_amount] = 1
    skyline_mask_1d[758-shift_amount:762+shift_amount] = 1
        
    fake_cont = np.linspace(0,num_pixels,num_pixels)**cont_pow 
    fake_cont[0]
    fake_cont = fake_cont[::-1] - min(fake_cont)
    sky_spectra += fake_cont
    
    #Making noise higher where there is more flux    
    fake_poisson_noise = np.abs(sky_spectra)#**0.5 
    
    mean_fake_poisson_noise = np.mean(fake_poisson_noise, axis=0)**0.5

    
    error_spectra = random.random([num_spec,num_pixels])
    dont_keep = error_spectra>(1-frac_errors_per_spec)
    error_spectra[dont_keep] = 0
    error_spectra[~dont_keep] = 1
    error_spectra[bad_values] = 0
    
    #Subtract will better emulate what the PCA will be doing
    if subtract:
        sky_spectra -= mean_fake_poisson_noise
    
    return sky_spectra, skyline_mask_1d, mean_fake_poisson_noise, error_spectra
