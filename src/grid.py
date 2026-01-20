# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 18:02:30 2025

@author: bhuyn
"""
import numpy as np
import config
from linreg import LinReg

class Grid:
    def __init__(self, x, y, z, cts, grid_type, percent_zero):
        self.x = x # t
        self.y = y # x
        self.z = z
        self.q = None
        self.w = None
        self.z_tilde = None
        self.shifted = False
        self.squared = False
        self.cts = cts
        self.grid_type = grid_type
        self.log_scale = False
        self.mask_applied = False
        self.percent_zero = percent_zero
        
    
    def fourier_transform(self, shift_fft=False, square_fft=False):
        if self.grid_type == 'default':
            self.w = np.fft.fftfreq(n=len(self.x), d=config.dt)
            self.w = np.fft.fftshift(self.w)
            self.q = np.fft.fftfreq(n=len(self.y), d=config.dx)
            self.q = np.fft.fftshift(self.q)
            self.z_tilde = np.fft.fft2(self.z)
            if shift_fft:
                self.z_tilde = np.fft.fftshift(self.z_tilde)
                self.shifted = True
            if square_fft:
                self.z_tilde = np.abs(self.z_tilde) ** 2
                self.squared = True
            self.grid_type = 'fourier'
            
            return self
    
    
    def get_grid_range(self, dim): # TODO: 
        match dim:
            case 't':
                dim_range = self.x.max() - self.x.min()
            case 'x': 
                dim_range = self.y.max() - self.y.min()
        return dim_range
    
    
    def log_transform(self, x, y):
        x = np.log10(x)
        y = np.log10(y)
        return x, y
    
    
    def grid_to_linreg_over(self, dim):

        mask = self.get_mask(dim)
        if dim == 'q':
            x = self.q[mask]
            y = self.z_tilde.mean(axis=0)[mask] # Average over omega
        if dim == 'w':
            x = self.w[mask]
            y = self.z_tilde.mean(axis=1)[mask] # Average over q
        
        x, y = self.log_transform(x, y)
        
        linreg = LinReg(x, y, xlabel=dim)
        return linreg
    
    
    def get_mask(self, x):
        if self.grid_type == 'fourier':
            if x == 'q':
                mask = (self.q > 0) & (self.q < 10 ** config.TANGENT_CUTOFF)
            if x == 'w':
                mask = (self.w > 0) & (self.w < 10 ** config.TANGENT_CUTOFF_TIME)
        return mask
