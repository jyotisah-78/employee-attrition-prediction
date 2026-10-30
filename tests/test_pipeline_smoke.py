import tempfile
import unittest
from pathlib import Path

from tests import helpers  # noqa: F401
from attrition.config import Config
from attrition.pipeline import run_pipeline
from attrition.synthetic import write_synthetic_files


class PipelineSmokeTest(unittest.TestCase):
    def test_end_to_end_on_fake_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(data_dir=root / "data", output_dir=root / "out", model_dir=root / "models",
                         n_estimators=30, cv_folds=3)
            write_synthetic_files(cfg, n_ibm=400, n_emp=600)
            out = run_pipeline(cfg, mode="merged", model_names=["RandomForest", "HistGB", "Stacked"])
            self.assertEqual(set(out["results"]), {"RandomForest", "HistGB", "Stacked"})
            self.assertIn(out["summary"]["best_model"], out["results"])
            for f in ("merged_results.csv", "merged_summary.json", "merged_roc_curves.png",
                      "merged_model_comparison.png", "merged_confusion_matrix.png", "merged_cv_folds.png"):
                self.assertTrue((root / "out" / f).exists(), f)
            self.assertTrue((root / "models" / "merged_best_model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
