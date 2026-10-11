
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = DATA_DIR / "train_identity.csv"


def inspect_file(file_path, sample_rows=10_000):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path.resolve()}")

    # Read only a small sample to protect RAM.
    df = pd.read_csv(file_path, nrows=sample_rows)

    print(f"\n{'=' * 60}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 60}")

    print(f"Sample shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nTop missing-value percentages:")
    print(
        (df.isna().mean() * 100)
        .sort_values(ascending=False)
        .head(15)
        .round(2)
        .to_string()
    )

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"\nSample memory usage: {memory_mb:.2f} MB")

    if "isFraud" in df.columns:
        print("\nTarget distribution:")
        print(df["isFraud"].value_counts(dropna=False))
        print(f"Fraud rate: {df['isFraud'].mean() * 100:.3f}%")

    del df


def main():
    inspect_file(TRANSACTION_FILE)
    inspect_file(IDENTITY_FILE)


if __name__ == "__main__":
    main()
