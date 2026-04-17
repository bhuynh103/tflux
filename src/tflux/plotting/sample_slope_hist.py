import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import tflux.io.paths as paths
from tflux.dtypes import Sample


def _log_formatter(val, pos):
    """Format a log10 tick value x as 10^x."""
    exp = int(round(val))
    return rf"$10^{{{exp}}}$"


def plot_linreg_fits(sample: Sample) -> plt.Figure:
    """
    Plot individual junction best-fit lines and a sample mean best-fit line
    for both q and w spectral dimensions.
 
    Each junction's LinReg is plotted as a faded line over its own data domain.
    The mean best-fit line uses the mean slope and intercept across all valid
    junctions, plotted over the union of all x domains.
 
    Parameters
    ----------
    sample : Sample
        Sample object with populated valid_juncs. Each junction must have
        non-None linreg_q and linreg_w attributes.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
        1x2 figure with q fit (left) and w fit (right).
    """
    juncs = [j for j in sample.valid_juncs if j.linreg_q is not None and j.linreg_w is not None]
    if not juncs:
        raise ValueError("No valid junctions with linreg_q and linreg_w found in sample.")
 
    dims = [
        ("q", [j.linreg_q for j in juncs], r"$q \; (m^{-1})$", r"$\langle |u|^2 \rangle_\omega$"),
        (r"\omega", [j.linreg_w for j in juncs], r"$\omega \; (s^{-1})$", r"$ \langle |u|^2 \rangle_q$"),
    ]
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
 
    for ax, (dim_label, linregs, xlabel, ylabel) in zip(axes, dims):
        slopes = np.array([lr.m for lr in linregs])
        intercepts = np.array([lr.int for lr in linregs])
        mean_m = slopes.mean()
        mean_int = intercepts.mean()
 
        # Global x domain for mean line
        all_x = np.concatenate([lr.x for lr in linregs])
        x_global = np.linspace(all_x.min(), all_x.max(), 200)
 
        # Individual junction lines
        for lr in linregs:
            x_fit = np.linspace(lr.x.min(), lr.x.max(), 100)
            y_fit = lr.m * x_fit + lr.int
            ax.plot(x_fit, y_fit, color="steelblue", alpha=0.25, linewidth=0.9, zorder=1)
 
        # Mean best-fit line
        y_mean = mean_m * x_global + mean_int
        ax.plot(
            x_global, y_mean,
            color="crimson", linewidth=2.2, zorder=3,
            label=rf"$\alpha_{dim_label}={mean_m:.2f}$",
        )
 
        ax.set_xlabel(xlabel, fontsize=32)
        ax.set_ylabel(ylabel, fontsize=32)
        ax.tick_params(axis="both", labelsize=24)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_log_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_log_formatter))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=16, framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
 
    return fig


def plot_linreg_hist(sample: Sample) -> plt.Figure:
    """
    Plot a histogram of individual junction slopes (m) for both q and w 
    spectral dimensions, with a vertical line representing the sample mean.
    """
    juncs = [j for j in sample.valid_juncs if j.linreg_q is not None and j.linreg_w is not None and j.roi_index != -1]
    
    if not juncs:
        raise ValueError("No valid junctions with linreg_q and linreg_w found in sample.")

    dims = [
        ("q", [j.linreg_q for j in juncs], r"$\alpha_q$"),
        (r"\omega", [j.linreg_w for j in juncs], r"$\alpha_\omega$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (dim_label, linregs, xlabel) in zip(axes, dims):
        # Extract all slopes (m) for this dimension
        slopes = np.array([lr.m for lr in linregs])
        mean_m = slopes.mean()
        std_m = slopes.std()

        # Plot the histogram
        # 'density=True' makes the area under the histogram sum to 1
        n, bins, patches = ax.hist(
            slopes, 
            bins=15, 
            color="steelblue", 
            edgecolor="white", 
            alpha=0.7, 
        )

        # Add a vertical line for the mean slope
        ax.axvline(
            mean_m, 
            color="crimson", 
            linestyle="--", 
            linewidth=2.5, 
            zorder=3,
            label=rf"$\bar{{\alpha}}_{{{dim_label}}} = {mean_m:.2f} \pm {std_m:.2f}$"
        )

        # Formatting
        ax.set_xlabel(xlabel, fontsize=28)
        ax.set_ylabel("Frequency", fontsize=24)
        ax.tick_params(axis="both", labelsize=22)
        
        # Remove log formatters as slopes are typically linear 
        # (Unless your slopes are also logarithmic)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        ax.legend(fontsize=16, framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    return fig


def compare_linreg_fits(sample_a: Sample, sample_b: Sample, labels=("A", "B")) -> plt.Figure:
    """Compare individual and mean best-fit lines between two samples."""
    samples = [sample_a, sample_b]
    colors = ["steelblue", "darkorange"]
    mean_colors = ["blue", "crimson"]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    dims = [
        ("q", "linreg_q", r"$q \; (m^{-1})$", r"$\langle |u|^2 \rangle_\omega$"),
        (r"\omega", "linreg_w", r"$\omega \; (s^{-1})$", r"$ \langle |u|^2 \rangle_q$"),
    ]

    for ax, (dim_label, attr, xlabel, ylabel) in zip(axes, dims):
        for s_idx, sample in enumerate(samples):
            juncs = [j for j in sample.valid_juncs if getattr(j, attr) is not None]
            linregs = [getattr(j, attr) for j in juncs]
            
            slopes = np.array([lr.m for lr in linregs])
            intercepts = np.array([lr.int for lr in linregs])
            
            # Individual Faded Lines
            for lr in linregs:
                x_fit = np.linspace(lr.x.min(), lr.x.max(), 100)
                ax.plot(x_fit, lr.m * x_fit + lr.int, color=colors[s_idx], alpha=0.15, lw=0.8, zorder=1)

            # Mean Best-Fit Line
            all_x = np.concatenate([lr.x for lr in linregs])
            x_glob = np.linspace(all_x.min(), all_x.max(), 200)
            y_mean = slopes.mean() * x_glob + intercepts.mean()
            
            ax.plot(x_glob, y_mean, color=mean_colors[s_idx], lw=2.5, zorder=3,
                    label=rf"{labels[s_idx]}: $\bar{{\alpha}}_{{{dim_label}}}={slopes.mean():.2f}$")

        # Standard Formatting
        ax.set_xlabel(xlabel, fontsize=32); ax.set_ylabel(ylabel, fontsize=32)
        ax.tick_params(axis="both", labelsize=24)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_log_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_log_formatter))
        ax.legend(fontsize=14, framealpha=0.8)
        ax.grid(True, alpha=0.3, linestyle="--")
        
    plt.tight_layout()
    return fig


def compare_linreg_hists(sample_a: Sample, sample_b: Sample, labels=("A", "B")) -> plt.Figure:
    """Compare slope distributions between two samples using overlaid histograms."""
    samples = [sample_a, sample_b]
    colors = ["steelblue", "darkorange"]
    mean_colors = ["blue", "crimson"]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    dims = [("q", "linreg_q", r"$\bar{{\alpha}}_q$"), (r"\omega", "linreg_w", r"$\bar{{\alpha}}_\omega$")]

    for ax, (dim_label, attr, xlabel) in zip(axes, dims):
        for s_idx, sample in enumerate(samples):
            juncs = [j for j in sample.valid_juncs if getattr(j, attr) is not None]
            slopes = np.array([getattr(j, attr).m for j in juncs])
            
            # Histogram
            ax.hist(slopes, bins=15, alpha=0.5, color=colors[s_idx], 
                    edgecolor="white", label=f"Sample {labels[s_idx]}", density=True)
            
            # Mean Vertical Line
            ax.axvline(slopes.mean(), color=mean_colors[s_idx], ls='--', lw=2.5,
                       label=rf"{labels[s_idx]} $\bar{{\alpha}}_{{{dim_label}}} = {slopes.mean():.2f} \pm {slopes.std():.2f}$")

        ax.set_xlabel(xlabel, fontsize=28); ax.set_ylabel("Density", fontsize=24)
        ax.tick_params(axis="both", labelsize=22)
        ax.legend(fontsize=12, framealpha=0.8)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


def plot_all_gradient_histograms(csv_path_list: list[Path], output_dir: Path):
    "Remember that config.CSV_PATHS include metrics that may be outdated"
    for csv_path in csv_path_list:
        print(csv_path)
        root = paths.find_root()
        csv_path = root / csv_path

        fig = plot_gradient_histograms(csv_path=csv_path, bins=16)
        png_name = f'{csv_path.stem}_hist.png'

        fig.set_subplot_titles([
            r'From Gradient: $\frac{\partial u^2}{\partial q}$',
            r'From Gradient: $\frac{\partial u^2}{\partial \omega}$', 
            r'From Averaging: $\frac{d \langle u^2 \rangle_\omega}{d q}$',
            r'From Averaging: $\frac{d \langle u^2 \rangle_q}{d \omega}$'
        ])

        fig.set_subplot_xlabels([
            r'$\alpha$',
            r'$\alpha$',
            r'$\alpha$', 
            r'$\alpha$'
        ])
        fig.savefig(output_dir / png_name)
        # plt.show()


def plot_gradient_histograms(csv_path, bins=30, title=None) -> plt.Figure:
    """
    Plot 2x2 histograms for grad_q, grad_w, linreg_q, linreg_w from a CSV.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file.
    bins : int, optional
        Number of histogram bins.
    title : str, optional
        Main title for the entire figure. If None, uses csv_path starting after 'processed_trimmed'.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with methods to set titles
    axes : array of matplotlib.axes.Axes
        Array of axes objects
    """
    df = pd.read_csv(csv_path)
    cols = ["grad_q", "grad_w", "linreg_q", "linreg_w"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    # Different colors for each subplot
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
    
    # Create figure with more spacing
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.35, top=0.92, bottom=0.08, left=0.1, right=0.95)
    axes = axes.flatten()
    
    # Generate title from csv_path if not provided
    if title is None:
        csv_path_obj = Path(csv_path)
        parts = csv_path_obj.parts
        
        # Find 'processed_trimmed' and take everything after it
        start_idx = None
        for i, part in enumerate(parts):
            if part == 'processed_trimmed':
                start_idx = i + 1  # Start from the directory AFTER processed_trimmed
                break
        
        if start_idx is not None and start_idx < len(parts):
            # Join from that directory onwards
            title = str(Path(*parts[start_idx:]))
        else:
            # Fallback to just the filename if 'processed_trimmed' not found
            title = csv_path_obj.name
    
    # Set main title
    fig.suptitle(title, fontsize=14)
    
    for ax, col, color in zip(axes, cols, colors):
        data = df[col].dropna()
        
        # Flip sign if data is predominantly negative
        if data.mean() < 0:
            data = -data
        
        sns.histplot(
            data,
            bins=bins,
            kde=False,
            ax=ax,
            color=color
        )
        ax.set_title("", fontsize=12)  # Empty, set manually with larger font
        ax.set_xlabel("")  # Empty, set manually
        ax.set_ylabel("Count")
        
        # Make y-axis show integer counts
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Add grid lines
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)  # Put grid behind the bars
    
    # Add methods to easily set titles
    fig.set_main_title = lambda title: fig.suptitle(title, fontsize=12)
    
    def set_subplot_titles(titles):
        """Set titles for all 4 subplots. Expects list of 4 strings."""
        if len(titles) != 4:
            raise ValueError("Must provide exactly 4 titles")
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=12)
    
    def set_subplot_xlabels(labels):
        """Set x-axis labels for all 4 subplots. Expects list of 4 strings."""
        if len(labels) != 4:
            raise ValueError("Must provide exactly 4 labels")
        for ax, label in zip(axes, labels):
            ax.set_xlabel(label)
    
    fig.set_subplot_titles = set_subplot_titles
    fig.set_subplot_xlabels = set_subplot_xlabels
    
    return fig
