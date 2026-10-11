
from pathlib import Path
import gc
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
OUTPUT_FILE = PROCESSED_DIR / "train_merged.parquet"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading transaction dataset...")
    transactions = pd.read_csv(
        TRANSACTION_FILE,
        low_memory=False
    )

    print("Loading identity dataset...")
    identity = pd.read_csv(
        IDENTITY_FILE,
        low_memory=False
    )

    # Validate the join key before merging.
    for name, df in [
        ("transactions", transactions),
        ("identity", identity),
    ]:
        if "TransactionID" not in df.columns:
            raise ValueError(f"{name} is missing TransactionID")

        duplicate_count = df["TransactionID"].duplicated().sum()
        print(f"{name}: {duplicate_count:,} duplicate TransactionIDs")

        if duplicate_count:
            raise ValueError(
                f"Duplicate TransactionIDs found in {name}. "
                "Investigate before merging."
            )

    original_rows = len(transactions)

    print("\nMerging datasets...")
    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_identity"),
    )

    matched = int((merged["_merge"] == "both").sum())
    unmatched = int((merged["_merge"] == "left_only").sum())

    merged.drop(columns="_merge", inplace=True)

    print("\nMerge validation")
    print("-" * 40)
    print(f"Original transaction rows: {original_rows:,}")
    print(f"Merged rows:               {len(merged):,}")
    print(f"Identity matches:          {matched:,}")
    print(f"No identity match:         {unmatched:,}")
    print(f"Final columns:             {len(merged.columns)}")

    if len(merged) != original_rows:
        raise ValueError("Row count changed unexpectedly!")

    fraud_rate = merged["isFraud"].mean() * 100
    print(f"Fraud rate after merge:     {fraud_rate:.4f}%")

    print("\nSaving Parquet file...")
    merged.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print(
        f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB"
    )

    del transactions, identity, merged
    gc.collect()

    print("\nDataset merge completed successfully!")


if __name__ == "__main__":
    main()
