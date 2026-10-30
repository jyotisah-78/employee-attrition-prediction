"""Create FAKE demo CSVs (same column names as the real ones) in data/raw/.
Only for checking that the code runs. Do NOT report results from this data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attrition.config import Config  # noqa: E402
from attrition.synthetic import write_synthetic_files  # noqa: E402

if __name__ == "__main__":
    a, b = write_synthetic_files(Config())
    print(f"Wrote FAKE demo data:\n  {a}\n  {b}\nDo not report results from this data.")
