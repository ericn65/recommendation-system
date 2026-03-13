import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(results: dict, name_fig: str = "metrics.png") -> None:
    """Plot comparison of models across metrics."""
    df = pd.DataFrame(results).T

    df.plot(kind="bar", figsize=(8, 5))

    plt.title("Model comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(name_fig)
