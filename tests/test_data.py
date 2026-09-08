"""Guard against client overlap and invalid training inputs."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_partition


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "data.csv"
        self.frame = pd.DataFrame(np.repeat(np.arange(103)[:, None], 23, axis=1),
                                  columns=[f"f{i}" for i in range(23)])
        self.frame["class"] = np.arange(103) % 5
        self.frame.to_csv(self.path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_partitions_are_disjoint_complete_and_repeatable(self):
        seen = set()
        for client in range(4):
            parts = load_partition(self.path, client, 4)
            repeat = load_partition(self.path, client, 4)
            for a, b in zip(parts, repeat):
                np.testing.assert_array_equal(a, b)
            train, _, val, _ = parts
            self.assertTrue(set(train[:, 0]).isdisjoint(val[:, 0]))
            ids = set(train[:, 0]) | set(val[:, 0])
            self.assertTrue(seen.isdisjoint(ids))
            seen |= ids
        self.assertEqual(seen, set(range(103)))

    def test_rejects_invalid_labels(self):
        self.frame.loc[0, "class"] = 0.5
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "Class labels"):
            load_partition(self.path, 0, 2)

    def test_rejects_invalid_client(self):
        with self.assertRaises(ValueError):
            load_partition(self.path, 4, 4)


if __name__ == "__main__":
    unittest.main()
