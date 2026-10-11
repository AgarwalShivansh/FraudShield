
from pathlib import Path
import gc
import pandas as pd

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
CHUNK_SIZE = 50_000

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def profile_csv(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path.resolve()}")

    print(f"\nProfiling {file_path.name}...")
    total_rows = 0
    missing_counts = None
    fraud_counts = {0: 0, 1: 0}

    # Read only one chunk at a time to limit RAM usage.
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        current_missing = chunk.isna().sum()

        if missing_counts is None:
            missing_counts = current_missing
        else:
            missing_counts = missing_counts.add(
                current_missing, fill_value=0
            )

        if "isFraud" in chunk.columns:
            counts = chunk["isFraud"].value_counts()
            for label, count in counts.items():
                fraud_counts[int(label)] = (
                    fraud_counts.get(int(label), 0) + int(count)
                )

        del chunk
        gc.collect()

    if total_rows == 0:
        raise ValueError(f"No rows found in {file_path.name}")

    missing_pct = (
        missing_counts / total_rows * 100
    ).sort_values(ascending=False)

    report = pd.DataFrame({
        "missing_count": missing_counts.astype("int64"),
        "missing_percentage": missing_pct
    })

    report.index.name = "column"
    report.to_csv(
        REPORT_DIR / f"{file_path.stem}_missingness.csv"
    )

    print(f"Total rows: {total_rows:,}")
    print(f"Total columns: {len(missing_counts)}")

    print("\nTop 20 columns by missing percentage:")
    print(report.head(20).round(2).to_string())

    if "isFraud" in file_path.name:
        pass

    if file_path.name == "train_transaction.csv":
        fraud_total = fraud_counts.get(1, 0)
        legitimate_total = fraud_counts.get(0, 0)

        print("\nFull target distribution:")
        print(f"Legitimate: {legitimate_total:,}")
        print(f"Fraudulent: {fraud_total:,}")
        print(
            f"Fraud rate: {fraud_total / total_rows * 100:.4f}%"
        )

    return total_rows


def main():
    transaction_path = RAW_DIR / "train_transaction.csv"
    identity_path = RAW_DIR / "train_identity.csv"

    profile_csv(transaction_path)
    profile_csv(identity_path)

    print("\nReports saved in:", REPORT_DIR.resolve())


if __name__ == "__main__":
    main()
