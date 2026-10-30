"""Run the whole project:  python main.py --help"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from attrition.config import Config  # noqa: E402
from attrition.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Employee attrition prediction pipeline")
    p.add_argument("--mode", choices=["merged", "ibm", "employee"], default="merged",
                   help="which data to use (default: merged)")
    p.add_argument("--models", default=None,
                   help="comma-separated subset, e.g. RandomForest,HistGB,XGBoost")
    p.add_argument("--no-cv", action="store_true", help="skip cross-validation (faster)")
    p.add_argument("--quick", action="store_true", help="fewer trees + 3 folds (fast demo run)")
    p.add_argument("--no-smote", action="store_true", help="turn SMOTE off")
    p.add_argument("--data-dir", default=None, help="folder that holds the CSV files")
    p.add_argument("--output-dir", default=None, help="folder for results/plots")
    p.add_argument("--model-dir", default=None, help="folder for the saved model")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = Config()
    if args.quick:
        cfg.n_estimators, cfg.cv_folds = 100, 3
    if args.no_smote:
        cfg.use_smote = False
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)

    names = [m.strip() for m in args.models.split(",")] if args.models else None
    run_pipeline(cfg, mode=args.mode, model_names=names, run_cv=not args.no_cv)


if __name__ == "__main__":
    main()
