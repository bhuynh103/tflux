import matplotlib.pyplot as plt
from scipy.stats import linregress
from tflux.dtypes import LinReg

def plot_2d_fft_slope(linreg: LinReg, ax=None, scale=None):
        
    log_kx = linreg.x
    log_msd = linreg.y
    # log_std_err = linreg["yerr"]
    fit_tangent = linreg.x * linreg.m + linreg.int
    
    kx = 10 ** log_kx
    msd = 10 ** log_msd
    fit_tangent_10 = 10 ** fit_tangent    

    if ax == None:
        fig, ax = plt.subplots()
    
    # Plot the scatterplot
    
    # ax.errorbar(log_kx, log_msd, yerr=log_std_err, fmt="k.", c='red', capsize=0, elinewidth=0.5)
    ax.errorbar(kx, msd, yerr=0, fmt=".", c='black', capsize=0, elinewidth=0.5, ms=16, lw=0.25)
    # ax.plot(log_kx, fit_best, color='red')
    ax.plot(kx, fit_tangent_10, color='black')
    
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        match linreg.xlabel:
            case 'q':
                ax.set_xlabel(r"$q \; (m^{-1})$")
            case 'w':
                ax.set_xlabel(r"$\omega \; (s^{-1})$")
        ax.set_ylabel(r"$\langle |\tilde{u}^2| \rangle \; (m^4 s^2)$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.text(
        0.95, 0.95,
        f"slope $= {linreg.m:.2f}$",
        transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.5),
        fontsize=18,
    )
    return ax