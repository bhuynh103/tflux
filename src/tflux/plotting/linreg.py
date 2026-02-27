import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_2d_fft_slope(linreg, ax, scale=None):
        
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
    ax.errorbar(kx, msd, yerr=0, fmt=".", c='black', capsize=0, elinewidth=0.5, ms=4, lw=0.25)
    # ax.plot(log_kx, fit_best, color='red')
    ax.plot(kx, fit_tangent_10, color='black')
    
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        match linreg.xlabel:
            case 'q':
                ax.set_xlabel("log q (1/m)")
            case 'w':
                ax.set_xlabel("log omega (1/s)")
        ax.set_ylabel("log amp squared (m^4)") # r"Log $\langle |u^2(q)| \rangle$ $(m^4)$"
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    return ax