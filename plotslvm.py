# -*- coding: utf-8 -*-
"""

@author: Elizabeth Watkins

Plots to see how the pipeline (tests) look like

"""

import matplotlib.pyplot as plt
import numpy as np

class DiagnosticPlots():
    
    def __init__(self, sky_spectra, wavelength, skyline_mask_1d=None, spectrum_number=0, fake=True, perc=1):
        self.sky_spectra = sky_spectra
        self.wavelength = wavelength
        self.sn = spectrum_number
        self.perc = perc
        
        self.num_spec, self.num_pixels = sky_spectra.shape
        
        self.skyline_mask_1d = skyline_mask_1d
        
        if fake:
            self.fake = ''
        else:
            self.fake = 'fake '
            
        self.vmin_prev, self.vmax_prev = np.nanpercentile(sky_spectra, [perc, 100-perc]) 
        self.vmin_after, self.vmax_after = [None, None]
    
        #spectrum 1
        self.fig1, self.ax1 = plt.subplots(2, 1, figsize=(14,8))#, sharex=True, sharey=True)
        
        #all spectra
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10,10))
        
        self._view_init_spectrum()
        self._view_all_spectra_init()
        
        self.ls = ['r--', 'k-']
        
    def _f(self, string):
        string = string.replace(self.fake, '')
        string.replace(self.fake.capitalize(), '')
        
        return string
        
    def view_spectrum(self, data, which, processed, n_comp_used=None):
        # pl.view_spectrum(res_remo_pre_proc_remo, which='after', processed=False)
        ind=0
        if processed:
            ind=1
        
        if which.lower() == 'before':
            lab = 'Before pca'
            ind_co=1 
            
        else:
            ind_co = 0
            lab = 'After pca'
            if n_comp_used is not None:
                lab +=  ': components needed: %d' % n_comp_used[self.sn]
            

        self.ax1[ind].plot(self.wavelength, data[self.sn], self.ls[ind_co], label=lab)
        if processed and which.lower() == 'before':
            if self.skyline_mask_1d is not None: 
                y1 = self.ax1[1].get_ylim()
                self.ax1[1].fill_between(self.wavelength, *y1, where=self.skyline_mask_1d==1, facecolor='grey', alpha=0.25, label='Where skylines')
                self.ax1[1].set_ylim(*y1)
                
        self.ax1[ind].legend(loc='best')

    def view_all_spectra(self, data, which, processed):
        ind_col=0
        if processed:
            ind_col=1
            if self.vmin_after is None:
                self.vmin_after, self.vmax_after = np.nanpercentile(data, [self.perc, 100-self.perc]) 
            vmin, vmax = self.vmin_after, self.vmax_after
        else:
            vmin, vmax = self.vmin_prev, self.vmax_prev
            
            
        ind_row = 1
        cb=False
        if which.lower() == 'before':
            ind_row = 0
            
            cb = True

        # self.vmin_prev, self.vmax_prev
        im = self.ax[ind_row, ind_col].imshow(data, vmin=vmin, vmax=vmax, origin='lower', extent=[np.min(self.wavelength), np.max(self.wavelength), 1, data.shape[0]])
        if cb:
            plt.colorbar(im, ax=self.ax[0,1], label=self._f('(Fake flux - cont) / Poisson noise'), location='bottom')    

    def _view_init_spectrum(self):
       
        ax1 = self.ax1
        
        sn = self.sn
            
    
        ax1[0].set_title(self._f('Example of origonal fake data before and after pca'))
        ax1[1].set_title(self._f('Example of processed fake data after pca'))
        
        
        ax1[0].set_ylabel(self._f('Fake flux'))
        ax1[1].set_ylabel(self._f('Fake flux'))
        ax1[1].set_xlabel(self._f(r'Fake wavelength [$\AA$]'))
        
        ax1[0].plot(self.wavelength, self.sky_spectra[sn], 'k-', label='Before pca')#, alpha=0.7, lw=0.7)
        
        if self.skyline_mask_1d is not None:
            y0 = ax1[0].get_ylim()
            ax1[0].fill_between(self.wavelength, *y0, where=self.skyline_mask_1d==1, facecolor='grey', alpha=0.25, label='Where skylines')
            ax1[0].set_ylim(*y0)
        
        plt.show()
        
    def _view_all_spectra_init(self):
        ax = self.ax
    
        ax[0,0].set_title(self._f('All fake data'))
        ax[0,1].set_title(self._f('All fake data processed'))
        ax[1,0].set_title('Original with PCA')        
        ax[1,1].set_title('Processed with PCA')
        
        im = ax[0,0].imshow(self.sky_spectra, vmin=self.vmin_prev, vmax=self.vmax_prev, origin='lower', extent=[np.min(self.wavelength), np.max(self.wavelength), 1, self.num_spec])

        plt.colorbar(im, ax=ax[0,0], label=self._f('Fake flux'), location='bottom')
        
        ax[0,0].set_ylabel('Spectrum number')
        ax[0,1].set_ylabel('Spectrum number')
        ax[1,0].set_xlabel(self._f(r'Fake wavelength [$\AA$]'))
        ax[1,1].set_xlabel(self._f(r'Fake wavelength [$\AA$]'))
        
        plt.show()
    
    def view_Poisson(self, poisson_noise):
        
        plt.figure('poisson_noise')
        plt.title('Poisson noise estimate')
        plt.plot(self.wavelength, poisson_noise)
        plt.ylabel(self._f('Fake flux'))
        plt.xlabel(self._f(r'Fake wavelength \AA'))
        
        plt.show()
        
    def view_number_comps_hist(self, n_comp_used, bins=5):
    
        plt.figure('Components needed')
        plt.title('Total number of components needed')
        plt.hist(n_comp_used, bins=50)#bins=np.arange(-0.5, max_recon_number+1.5, 1))
        plt.xlabel('Number of components needed to recon')
        plt.ylabel('Number')
        
        plt.show()
    
    def plot_comp_plots(self, pp_obj, eigen_system_dict_gap, tracked_dict, max_comp=5, skyline_mask_1d=None):
        
        if skyline_mask_1d is None and self.skyline_mask_1d is None:
            raise TypeError('`skyline_mask_1d` needs to be provided')
        elif skyline_mask_1d is None:
            skyline_mask_1d = self.skyline_mask_1d

        wavelength = self.wavelength[self.skyline_mask_1d==1]
        wavelength1 = self.wavelength
            
        sub = pp_obj
    
        error_array = sub.pca_use_mask_and_errors[:, skyline_mask_1d==1]
        spectra = sub.spectra[:, skyline_mask_1d==1]
        
        mean_array1 = eigen_system_dict_gap['m']
        eigen_vectors1 = eigen_system_dict_gap['U']
        fig, axs = plt.subplots(max_comp+1,1, sharex=True, figsize=(10,15))
        axs[0].plot(wavelength, mean_array1)
        
        for i in range(1, max_comp+1, 1):
            reverse = 1
            eig = eigen_vectors1[:,i-1]
            if max(eig) < abs(min(eig)) :
                reverse = -1
            axs[i].plot(wavelength, eig * reverse)
            axs[i].set_ylim(-0.1, 0.3)
        
        axs[0].set_xlim(min(wavelength), max(wavelength))
        axs[0].set_title('Mean and top 5 eigenspectra')
        axs[-1].set_xlabel(r'Wavelength $\AA$')
        axs[2].set_ylabel(r'Flux')
        
        reconstruct1 = sub.run(projection_method='simple', 
                                                  max_comp=max_comp,
                                                  )
               
        fig, axs = plt.subplots(max_comp,1, sharex=True, figsize=(10,15))
        for i in range(max_comp):
            axs[i].plot(wavelength, spectra[i,:], self.ls[1], label='Original spectra')
            recon = np.copy(reconstruct1[i])
            recon[skyline_mask_1d==0] = np.nan
            axs[i].plot(wavelength1, reconstruct1[i], self.ls[0], label='Reconstructed Spectra')
            axs[i].legend(loc='best')

        axs[0].set_title('Spectrum vs pca reconstruction using %d components' %max_comp)
        axs[0].set_xlim(min(wavelength), max(wavelength))
        axs[-1].set_xlabel(r'Wavelength $\AA$')
        axs[2].set_ylabel(r'Flux')
        
        plt.figure()
        plt.plot(tracked_dict['W']/np.sum(tracked_dict['W'],axis=1)[:,None])
        plt.xlabel('Number of streamed spectra')
        plt.ylabel('Normalised eigenvalues')
    
        plt.show()
