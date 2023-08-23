# -*- coding: utf-8 -*-
"""
@author: Elizabeth Watkins
"""

#VARIABLE_NAMES
variable_names = [
    'noise_gen_data', 
    'error_spectra', 
    'observation_numbers', 
    'skyline_mask', 
    'science_line_mask'
]

prep_steps = {
    1:'continuum',
    2:'normalise',
    3:'skylines',
    4:'science_lines',
    5:'outliers'
}

prep_names = {
    'continuum':1,
    'normalise':2,
    'skylines':3,
    'science_lines':4,
    'outliers':5
}

pca_kwargs = {
    'amount_of_eigen':100,
    'save_extra_param':False,
    'c_sq':0.787**2,
    'number_of_iterations':3,
    'pca_library': None
}