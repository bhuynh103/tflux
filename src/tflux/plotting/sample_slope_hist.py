import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_gradient_histograms(csv_path, bins=30):
    """
    Plot 2x2 histograms for grad_q, grad_w, linreg_q, linreg_w from a CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file.
    bins : int, optional
        Number of histogram bins.
    """
    df = pd.read_csv(csv_path)

    cols = ["grad_q", "grad_w", "linreg_q", "linreg_w"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.histplot(
            df[col].dropna(),
            bins=bins,
            kde=True,
            ax=ax
        )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    return fig, axes