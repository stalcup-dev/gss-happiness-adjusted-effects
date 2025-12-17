from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve


def main() -> int:
    """Download a CSV into data/raw/happiness.csv.

    Usage:
      set HAPPINESS_DATA_URL=<direct_csv_url>
      python scripts/download_data.py

    Notes:
      - The URL must point directly to a CSV file (no auth prompts).
    """

    url = os.getenv("HAPPINESS_DATA_URL")
    if not url:
        raise SystemExit("Set HAPPINESS_DATA_URL to a direct CSV URL")

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "data" / "raw" / "happiness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {url}")
    print(f"To: {out_path}")
    urlretrieve(url, out_path)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
