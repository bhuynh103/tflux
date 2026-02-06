import numpy as np
from scipy.stats import linregress
from pathlib import Path
from typing import Self
import tflux.pipeline.config as config


class Junction:
    def __init__(self, vertices: np.ndarray, is_top: bool) -> None:
        self.source_file: Path | None = None
        self.vertices = vertices
        self.is_top = is_top
        self.grid: Grid | None = None
        self.fft: GridFFT | None = None
        self.mesh: Mesh | None = None
        self.linreg_q: LinReg | None = None
        self.linreg_w: LinReg | None = None


class Sample:
    def __init__(self) -> None:
        self.juncs: list[Junction] = []
    
    
    def append_junction(self, junc: Junction) -> Self:
        self.juncs.append(junc)
        return self
    

    def list_metric(self, attr: str) -> list[float]:
        match attr:
            case 'a':
                attr_list = [junc.mesh.a for junc in self.juncs]
            case 'b':
                attr_list = [junc.mesh.b for junc in self.juncs]
            case 'q_m':
                attr_list = [junc.linreg_q.m for junc in self.juncs]
            case 'w_m':
                attr_list = [junc.linreg_w.m for junc in self.juncs]
        return attr_list

    
    def find_average_metric(self, attr: str) -> tuple[float, float]:
        attr_list: list[float] = self.list_metric(attr)
        mean: float = np.mean(attr_list)
        std: float = np.std(attr_list)
        return mean, std


class LinReg:
    def __init__(self, x: np.ndarray, y: np.ndarray, xlabel: str) -> None:
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.xlabel: str = xlabel  # q or w
        
        result: tuple = linregress(x, y)  # LinregressResult class is not in this version of scipy, use tuple instead
        self.m: float = result.slope
        self.int: float = result.intercept
        self.r: float = result.rvalue
        self.p: float = result.pvalue
        self.std_err: float = result.stderr


class Mesh:
    def __init__(self, q: np.ndarray, w: np.ndarray, z: np.ndarray, log_scale: bool = False) -> None:
        self.q: np.ndarray = q # q
        self.w: np.ndarray = w # w
        self.z: np.ndarray = z # u^2
        self.masked: bool = False
        self.denoised: bool = False
        self.log_scale: bool = log_scale # Default behavior
        self.z_residuals: np.ndarray | None = None
        self.a: float | None = None
        self.b: float | None = None
        self.c: float | None = None      
        
        
    def log_transform(self) -> Self:
        if not self.log_scale:
            self.q = np.log10(self.q)
            self.w = np.log10(self.w)
            self.z = np.log10(self.z)
            self.log_scale = True
            
        return self
        
        
    def exp_transform(self) -> Self:
        if self.log_scale:
            self.q = 10 ** self.q
            self.w = 10 ** self.w
            self.z = 10 ** self.z
            self.log_scale = False
        
        return self
    
    
    def get_residuals(self) -> np.ndarray | None:
        if self.log_scale:
            self.z_residuals = self.z - self.a * self.q - self.b * self.w - self.c
            return self.z_residuals
    
    
    def apply_masks(self, denoise: bool=False) -> Self:
        # Only positive frequencies and finite log amplitude
        positive_mask = ((self.q > 0) & (self.w > 0))
        finite_mask = np.isfinite(self.z)
        fit_mask = positive_mask & finite_mask
        if denoise:
            noise_mask = ((self.q < 10 ** config.TANGENT_CUTOFF) & (self.w < 10 ** config.TANGENT_CUTOFF_TIME))
            fit_mask = fit_mask & noise_mask
            self.denoised = True
        
        # Prepare the data we actually plot, meshes
        self.q = self.q[fit_mask].ravel(order='F')
        self.w = self.w[fit_mask].ravel(order='F')
        self.z = self.z[fit_mask].ravel()
        self.masked = True
        
        return self
    

    def find_loglog_gradient(self) -> Self:
        """
        Compute log-log gradient: z ~ a*q + b*w + c
        
        Automatically applies log transform if needed.
        Requires mesh to be masked and denoised first.
        
        Raises:
            ValueError: If mesh is not masked and denoised.
        """
        # Auto-fix: log scale is a reversible, safe transformation
        if not self.log_scale:
            self.log_transform()
        
        # Strict requirement: masking affects data quality, user must decide
        if not (self.masked and self.denoised):
            raise ValueError(
                "Mesh must be masked and denoised. "
                "Call apply_masks(denoise=True) before computing gradient."
            )
        
        # Compute gradient
        A: np.ndarray = np.c_[self.q, self.w, np.ones_like(self.q)]
        coeffs: np.ndarray
        coeffs, _, _, _ = np.linalg.lstsq(A, self.z, rcond=None)
        self.a, self.b, self.c = coeffs
        
        return self


class GridFFT():
    def __init__(self, q: np.ndarray, w: np.ndarray, z_tilde: np.ndarray, shifted=False, squared=False, log_scale=False, mask_applied=False):
        self.q = q
        self.w = w
        self.z_tilde = z_tilde
        self.shifted = shifted
        self.squared = squared
        self.log_scale = log_scale # Is this used?
        self.mask_applied = mask_applied
    
    
    def log_transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.log10(x)
        y = np.log10(y)
        return x, y


    def fft_to_linreg_over(self, dim: str) -> LinReg:
        mask = self.get_mask(dim)
        if dim == 'q':
            x = self.q[mask]
            y = self.z_tilde.mean(axis=1)[mask] # Averages over omega
            # print(f'q: {len(x)}, z: {len(y)}')
        if dim == 'w':
            x = self.w[mask]
            y = self.z_tilde.mean(axis=0)[mask] # Averages over q
            # print(f'w: {len(x)}, z: {len(y)}')
        
        # log transform after masking
        x, y = self.log_transform(x, y)
        
        linreg = LinReg(x, y, xlabel=dim)
        return linreg
    

    def get_mask(self, dim: str) -> np.ndarray:
        if dim == 'q':
            mask = (self.q > 0) & (self.q < 10 ** config.TANGENT_CUTOFF)
        if dim == 'w':
            mask = (self.w > 0) & (self.w < 10 ** config.TANGENT_CUTOFF_TIME)
        return mask


class Grid:
    def __init__(self, x, t, z, cts, percent_zero, log_scale=False):
        self.x: np.ndarray = x # x bins
        self.t: np.ndarray = t # t bins
        self.z: np.ndarray = z # shape (len(x_bins), len(t_bins))
        self.cts: np.ndarray = cts
        self.percent_zero: float = percent_zero
        self.log_scale: bool = log_scale # Is this used?
        
    
    def fourier_transform(self, shift_fft=False, square_fft=False) -> GridFFT:
        w = np.fft.fftfreq(n=len(self.t), d=config.dt)
        w = np.fft.fftshift(w)
        q = np.fft.fftfreq(n=len(self.x), d=config.dx)
        q = np.fft.fftshift(q)
        z_tilde = np.fft.fft2(self.z)

        if shift_fft:
            z_tilde = np.fft.fftshift(z_tilde)
        if square_fft:
            z_tilde = np.abs(z_tilde) ** 2
        
        grid_fft = GridFFT(q=q, w=w, z_tilde=z_tilde, shifted=shift_fft, squared=square_fft)

        return grid_fft
    
    
    def get_grid_range(self, dim): 
        match dim:
            case 't':
                dim_range = self.t.max() - self.t.min()
            case 'x': 
                dim_range = self.x.max() - self.x.min()
        return dim_range
    
    
    def log_transform(self, x, y):
        x = np.log10(x)
        y = np.log10(y)
        return x, y

