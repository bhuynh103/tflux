import tflux.pipeline.config as config
import numpy as np
from scipy.stats import linregress

class Sample:
    def __init__(self):
        self.juncs = []
    
    
    def append_junction(self, junc):
        self.juncs.append(junc)
        return self
    
    
    def find_average(self, attr):
        match attr:
            case 'a':
                attr_list = [junc.mesh.a for junc in self.juncs]
            case 'b':
                attr_list = [junc.mesh.b for junc in self.juncs]
            case 'q_m':
                attr_list = [junc.linreg_q.m for junc in self.juncs]
            case 'w_m':
                attr_list = [junc.linreg_w.m for junc in self.juncs]
        
        mean = np.mean(attr_list)
        std = np.std(attr_list)
        return mean, std


class Junction:
    def __init__(self, vertices, is_top, filename):
        self.filename = filename
        self.vertices = vertices
        self.is_top = is_top
        self.grid: Grid = None
        self.mesh: Mesh = None
        self.linreg_q: LinReg = None
        self.linreg_w: LinReg = None


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
    

class LinReg:
    def __init__(self, x, y, xlabel):
        self.x = x
        self.xlabel = xlabel # q or w
        self.y = y
        self.m, self.int, self.r, self.p, self.std_err = linregress(x, y)