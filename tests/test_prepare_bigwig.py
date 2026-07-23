import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "pausenet" / "prepare_bigwig.py"
SPEC = importlib.util.spec_from_file_location("prepare_bigwig", MODULE_PATH)
prepare_bigwig = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_bigwig)


class AnchorTableTests(unittest.TestCase):
    def test_headered_anchors_keep_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anchors.tsv"
            path.write_text(
                "chrom\tstart\tend\tstrand\tsplit\tanchor_type\n"
                "chr1\t1000\t2000\t+\ttest\tTSS\n"
            )
            columns, rows = prepare_bigwig.iter_anchor_rows(path)
            self.assertEqual(columns[-1], "anchor_type")
            self.assertEqual(next(rows)["anchor_type"], "TSS")
            rows.close()

    def test_four_column_bed_maps_the_fourth_field_to_strand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anchors.bed"
            path.write_text("chr1\t1000\t2000\t-\n")
            columns, rows = prepare_bigwig.iter_anchor_rows(path)
            self.assertEqual(columns, ["chrom", "start", "end", "strand"])
            self.assertEqual(next(rows)["strand"], "-")
            rows.close()

    def test_anchor_geometry_uses_the_existing_output_interval(self):
        geometry = prepare_bigwig.anchor_geometry(
            {"chrom": "chr1", "start": "10000", "end": "11000", "strand": "-"},
            {"chr1": 50000},
            input_length=2114,
            output_length=1000,
        )
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual(geometry["output_start"], 10000)
        self.assertEqual(geometry["output_end"], 11000)
        self.assertEqual(geometry["input_start"], 9443)
        self.assertEqual(geometry["input_end"], 11557)

    def test_anchor_geometry_rejects_chromosome_missing_from_signal_track(self):
        geometry = prepare_bigwig.anchor_geometry(
            {"chrom": "chrY", "start": "10000", "end": "10001", "strand": "+"},
            {"chrY": 50000},
            input_length=2114,
            output_length=1000,
            signal_chrom_sizes={"chr1": 50000},
        )
        self.assertIsNone(geometry)

    def test_anchor_geometry_rejects_profile_beyond_signal_track_bounds(self):
        geometry = prepare_bigwig.anchor_geometry(
            {"chrom": "chr1", "start": "10000", "end": "11000", "strand": "+"},
            {"chr1": 50000},
            input_length=2114,
            output_length=1000,
            signal_chrom_sizes={"chr1": 10500},
        )
        self.assertIsNone(geometry)

    def test_explicit_split_takes_precedence_over_chromosome_rules(self):
        split = prepare_bigwig.assign_split(
            {"chrom": "chr1", "split": "test"},
            split_column="split",
            train_chroms={"chr1"},
            validation_chroms=set(),
            test_chroms=set(),
            default_split="train",
        )
        self.assertEqual(split, "test")

    def test_cli_keeps_the_generic_anchors_bed_interface(self):
        parser = argparse.ArgumentParser()
        prepare_bigwig.add_prepare_bigwig_args(parser)
        option_names = {action.dest for action in parser._actions}
        self.assertIn("anchors_bed", option_names)
        self.assertNotIn("gtf", option_names)


if __name__ == "__main__":
    unittest.main()
