import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import helpers  # noqa: F401  (adds src/ to the path)
from attrition.config import Config
from attrition.data import load_dataset, rescale_satisfaction, to_binary
from attrition.features import assert_no_leakage, engineer_features
from attrition.synthetic import write_synthetic_files


class DataAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(data_dir=Path(self.tmp.name))
        write_synthetic_files(self.cfg, n_ibm=300, n_emp=400)

    def tearDown(self):
        self.tmp.cleanup()

    def test_to_binary(self):
        self.assertEqual(to_binary(pd.Series(["Yes", "no", "YES"])).tolist(), [1, 0, 1])
        self.assertEqual(to_binary(pd.Series([0, 1, 1])).tolist(), [0, 1, 1])
        with self.assertRaises(ValueError):
            to_binary(pd.Series(["maybe"]))

    def test_satisfaction_rescale(self):
        out = rescale_satisfaction(pd.Series([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(out.min(), 1.0)
        self.assertAlmostEqual(out.max(), 4.0)
        same = rescale_satisfaction(pd.Series([1, 2, 3, 4]))
        self.assertEqual(same.tolist(), [1, 2, 3, 4])

    def test_modes_and_row_counts(self):
        self.assertEqual(len(load_dataset(self.cfg, "ibm")), 300)
        self.assertEqual(len(load_dataset(self.cfg, "employee")), 400)
        self.assertEqual(len(load_dataset(self.cfg, "merged")), 700)
        with self.assertRaises(ValueError):
            load_dataset(self.cfg, "nonsense")

    def test_no_label_columns_in_features(self):
        X, y = engineer_features(load_dataset(self.cfg, "merged"), self.cfg)
        for bad in ("Attrition", "Churn", "left", "Target", "data_source", "EmpId", "EmployeeNumber"):
            self.assertNotIn(bad, X.columns)
        self.assertEqual(len(X), len(y))
        self.assertIn("income_log", X.columns)

    def test_leak_guard_raises(self):
        X = pd.DataFrame({"Churn": [0, 1], "a": [1, 2]})
        with self.assertRaises(ValueError):
            assert_no_leakage(X, self.cfg)

    def test_missing_file_message(self):
        cfg = Config(data_dir=Path(self.tmp.name) / "empty")
        with self.assertRaises(FileNotFoundError):
            load_dataset(cfg, "ibm")


if __name__ == "__main__":
    unittest.main()
