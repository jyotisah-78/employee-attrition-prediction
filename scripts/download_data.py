"""Download both datasets from Kaggle into data/raw/.

Credentials: either ~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY in a
.env file in the project root (see .env.example).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attrition.config import Config  # noqa: E402


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def download(dataset: str, dest: Path) -> bool:
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED for {dataset}:\n{result.stderr or result.stdout}")
        return False
    return True


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    cfg = Config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        sys.exit("The 'kaggle' command was not found. Activate your virtual environment "
                 "and run: pip install -r requirements.txt")
    ok = all([download(cfg.primary_dataset, cfg.data_dir),
              download(cfg.secondary_dataset, cfg.data_dir)])
    for name in (cfg.ibm_filename, cfg.emp_filename):
        found = list(cfg.data_dir.rglob(name))
        print(("FOUND   " if found else "MISSING ") + name)
        ok = ok and bool(found)
    sys.exit(0 if ok else 1)
