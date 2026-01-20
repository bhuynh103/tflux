# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 17:35:08 2025

@author: bhuyn
"""
import numpy as np
import config


class Mesh():
    def __init__(self, x, y, z, log_scale):
        self.x: np.ndarray = x # q
        self.y: np.ndarray = y # w
        self.z: np.ndarray = z # u^2
        self.masked: bool = False
        self.denoised: bool = False
        self.log_scale: bool = log_scale
        self.z_residuals: np.ndarray = None
        self.a = None
        self.b = None
        self.c = None      
        
        
    def log_transform(self):
        if not self.log_scale:
            self.x = np.log10(self.x)
            self.y = np.log10(self.y)
            self.z = np.log10(self.z)
            self.log_scale = True
            
        return self
        
        
    def exp_transform(self):
        if self.log_scale:
            self.x = 10 ** self.x
            self.y = 10 ** self.y
            self.z = 10 ** self.z
            self.log_scale = False
        
        return self
    
    
    def get_residuals(self):
        if self.log_scale:
            self.z_residuals = self.z - self.a * self.x - self.b * self.y - self.c
            return self.z_residuals
    
    
    def apply_masks(self, denoise=False):
        # Only positive frequencies and finite log amplitude
        positive_mask = ((self.x > 0) & (self.y > 0))
        finite_mask = np.isfinite(self.z)  # May bug
        fit_mask = positive_mask & finite_mask
        if denoise:
            noise_mask = ((self.x < 10 ** config.TANGENT_CUTOFF) & (self.y < 10 ** config.TANGENT_CUTOFF_TIME))
            fit_mask = fit_mask & noise_mask
            self.denoised = True
        
        # Prepare the data we actually plot, meshes
        self.x = self.x[fit_mask].ravel(order='F')
        self.y = self.y[fit_mask].ravel(order='F')
        self.z = self.z[fit_mask].ravel()
        self.masked = True
        
        return self
    
    
    def find_loglog_gradient(self):
        if not self.log_scale:
            self.log_transform()
        if self.masked and self.denoised:
            A = np.c_[self.x, self.y, np.ones_like(self.x)]
            coeffs, _, _, _ = np.linalg.lstsq(A, self.z, rcond=None)
            self.a, self.b, self.c = coeffs   # z ~ a*x + b*y + c
            
        return self