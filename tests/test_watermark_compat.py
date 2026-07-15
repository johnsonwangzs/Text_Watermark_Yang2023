import gzip
from pathlib import Path
import tempfile
import unittest

from models.watermark_compat import (
    GloveVectors,
    WatermarkModel,
    binary_encoding_function,
)


class WatermarkCoreTests(unittest.TestCase):
    def test_binary_encoding_is_deterministic_and_binary(self):
        first = binary_encoding_function("previouscurrent")
        self.assertEqual(first, binary_encoding_function("previouscurrent"))
        self.assertIn(first, (0, 1))

    def test_glove_reader_supports_gensim_gzip_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("3 2\n")
                handle.write("king 1.0 0.0\n")
                handle.write("queen 0.8 0.2\n")
                handle.write("unused 0.0 1.0\n")
            vectors = GloveVectors(path, {"king", "queen"})
            self.assertEqual(len(vectors), 2)
            self.assertAlmostEqual(vectors.similarity("king", "king"), 1.0)
            self.assertGreater(vectors.similarity("king", "queen"), 0.9)
            self.assertEqual(vectors.similarity("missing", "queen"), 0.0)

    def test_z_test_detects_a_strong_bit_one_bias(self):
        result = WatermarkModel._test([1] * 20, alpha=0.05, mode="fast")
        self.assertTrue(result.is_watermarked)
        self.assertEqual(result.ones, 20)
        self.assertLess(result.p_value, 0.05)

    def test_z_test_handles_no_eligible_words(self):
        result = WatermarkModel._test([], alpha=0.05, mode="fast")
        self.assertFalse(result.is_watermarked)
        self.assertEqual(result.n, 0)
        self.assertEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
