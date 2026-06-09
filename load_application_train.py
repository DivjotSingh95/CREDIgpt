from pathlib import Path

import pandas as pd

data_dir = Path(__file__).parent

for csv_path in sorted(data_dir.glob("*.csv")):
    print(f"\n{'=' * 60}")
    print(f"File: {csv_path.name}")
    print("=" * 60)

    df = pd.read_csv(csv_path, encoding="latin-1")
    print(df.head())
