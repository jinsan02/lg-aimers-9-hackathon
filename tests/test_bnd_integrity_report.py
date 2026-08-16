import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bnd_integrity_report", ROOT / "tools" / "bnd_integrity_report.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class BndIntegrityHelpersTest(unittest.TestCase):
    def test_historical_surface_requires_asymmetric_f_filter(self):
        self.assertEqual(MOD._required_drop_f_pre(2022), 0)
        self.assertEqual(MOD._required_drop_f_pre(2023), 2022)

    def test_object_array_hash_is_stable_and_order_sensitive(self):
        import numpy as np
        a = np.array(["TRAIN_2", "TRAIN_1"], dtype=object)
        b = np.array(["TRAIN_2", "TRAIN_1"], dtype=object)
        c = np.array(["TRAIN_1", "TRAIN_2"], dtype=object)
        self.assertEqual(MOD._sha_array(a), MOD._sha_array(b))
        self.assertNotEqual(MOD._sha_array(a), MOD._sha_array(c))

    def test_missing_artifacts_fail_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "LEDGER.tsv"
            ledger.write_text("x\n", encoding="utf-8")
            report = MOD.audit(root / "model", root / "out", ledger)
            self.assertFalse(report["ok"])
            self.assertEqual(report["expected_models"], 24)
            self.assertEqual(len(report["errors"]), 4)


if __name__ == "__main__":
    unittest.main()
