import pandas as pd
from scipy.stats import spearmanr


def generate_correlation_markdown_report(df_corr: pd.DataFrame) -> str:
    """
    Generate a markdown report summarizing the correlation analysis.

    Args:
        df_corr (pd.DataFrame): DataFrame containing global correlation data.

    Returns
    -------
        str: Markdown formatted report.
    """
    # Global summary
    global_summary = df_corr.groupby("feature").agg(
        mean_corr=("correlation", "mean"),
        std_corr=("correlation", "std"),
        count=("correlation", "count"),
    )
    global_summary = global_summary.sort_values("mean_corr", ascending=False)

    # Summary by modality
    modality_summary = df_corr.groupby(["feature", "modality"]).agg(
        mean_corr=("correlation", "mean"),
        std_corr=("correlation", "std"),
        count=("correlation", "count"),
    )
    modality_summary = modality_summary.reset_index()

    # Build markdown string
    md = "# Correlation Analysis Report\n\n"

    md += "## Global Summary\n\n"
    md += global_summary.to_markdown() + "\n\n"

    md += "## Summary by Modality\n\n"
    for feature in global_summary.index:
        md += f"### Feature: {feature}\n\n"
        feature_modality = modality_summary[
            modality_summary["feature"] == feature
        ].set_index("modality")
        md += feature_modality.to_markdown() + "\n\n"

    return md


def main():
    """Trial to ensure we can read everythin."""
    # Load data
    df = pd.read_csv("data.csv")

    # Calculate global correlations
    features = [col for col in df.columns if col not in ["target", "modality"]]
    correlations = []
    for feature in features:
        corr, _ = spearmanr(df[feature], df["target"])
        correlations.append({"feature": feature, "correlation": corr})

    df_corr = pd.DataFrame(correlations)

    # Add modality information if available
    if "modality" in df.columns:
        modalities = df["modality"].unique()
        modality_corrs = []
        for modality in modalities:
            df_mod = df[df["modality"] == modality]
            for feature in features:
                corr, _ = spearmanr(df_mod[feature], df_mod["target"])
                modality_corrs.append(
                    {"feature": feature, "modality": modality, "correlation": corr}
                )
        df_corr_modality = pd.DataFrame(modality_corrs)
        df_corr = df_corr.merge(
            df_corr_modality.groupby("feature")["correlation"].mean().reset_index(),
            on="feature",
            suffixes=("", "_modality_mean"),
        )

    # Generate report
    report = generate_correlation_markdown_report(df_corr)

    # Save report
    with open("correlation_report.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
