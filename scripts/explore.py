"""Entry point: data exploration — run before training."""

from pathlib import Path

import pandas as pd

from novozymes.exploration.bin_advisor import BinAdvisor
from novozymes.exploration.group_balance import GroupBalanceAnalyzer
from novozymes.exploration.label_stats import LabelAnalyzer
from novozymes.exploration.sequence_stats import SequenceAnalyzer


def explore(train_path: Path = Path("data/raw/train.csv")) -> None:
    """Run all exploration analyses and print summary."""
    df = pd.read_csv(train_path)
    print(f"Dataset: {train_path} ({len(df)} samples)\n")

    # Label stats
    label_analyzer = LabelAnalyzer()
    stats = label_analyzer.analyze(df)
    print("=== Label Distribution ===")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    outliers = label_analyzer.detect_outliers(df)
    print(f"\n  Outliers (IQR): {len(outliers)} samples")

    # Bin advisor
    print("\n=== BinAdvisor ===")
    advisor = BinAdvisor()
    bins = advisor.recommend(df["tm"])
    for key, val in bins.items():
        print(f"  {key}: {val}")

    # Group balance
    print("\n=== Group Balance ===")
    group_analyzer = GroupBalanceAnalyzer()
    groups = group_analyzer.analyze(df)
    for source, source_stats in groups.items():
        print(f"  {source}: {source_stats}")

    # Sequence stats
    print("\n=== Sequence Stats ===")
    seq_analyzer = SequenceAnalyzer()
    seq_stats = seq_analyzer.analyze(df)
    for key, val in seq_stats.items():
        if key != "aa_frequency":
            print(f"  {key}: {val}")
    print("  aa_frequency: (see plot)")


if __name__ == "__main__":
    explore()
