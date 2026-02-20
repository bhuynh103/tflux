import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

<<<<<<< HEAD
def plot_gradient_histograms(csv_path, bins=30) -> plt.Figure:
=======
def plot_gradient_histograms(csv_path, bins=30, title=None):
>>>>>>> 6ba60b0 (Code for histograms of 4 slopes for each sample)
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
<<<<<<< HEAD

    return fig
=======
        
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
    
    return fig, axes
>>>>>>> 6ba60b0 (Code for histograms of 4 slopes for each sample)
