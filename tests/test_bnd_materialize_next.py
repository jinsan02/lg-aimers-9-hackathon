import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bnd_materialize_next", ROOT / "tools" / "bnd_materialize_next.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class BndMaterializeContractTest(unittest.TestCase):
    def test_existing_array_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.npz"
            payload = {
                "y": np.array([0., 1.]),
                "pred": np.array([.4, .6]),
                "row_id": np.array(["A", "B"]),
            }
            np.savez_compressed(p, **payload)
            self.assertTrue(MOD._same_or_raise(p, payload))
            bad = dict(payload); bad["pred"] = np.array([.4, .7])
            with self.assertRaises(ValueError):
                MOD._same_or_raise(p, bad)

    def test_boundary_contract_is_exact(self):
        self.assertEqual(MOD.SPECS["BND22_base"], (2022, 2023))
        self.assertEqual(MOD.SPECS["BND23_cell"], (2023, 2024))
        self.assertEqual(MOD.SEEDS, (3, 4, 5, 6, 8, 13))


if __name__ == "__main__":
    unittest.main()
