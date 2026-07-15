import importlib.util
import argparse
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "pausenet" / "prepare_bigwig.py"
SPEC = importlib.util.spec_from_file_location("prepare_bigwig", MODULE_PATH)
prepare_bigwig = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_bigwig)

GENE_STRUCTURE_WINDOWS = prepare_bigwig.GENE_STRUCTURE_WINDOWS
anchor_position = prepare_bigwig.anchor_position
build_gene_structure_samples = prepare_bigwig.build_gene_structure_samples
gene_relative_to_genome = prepare_bigwig.gene_relative_to_genome
junction_indexes = prepare_bigwig.junction_indexes
add_prepare_bigwig_args = prepare_bigwig.add_prepare_bigwig_args


class GeneStructureWindowTests(unittest.TestCase):
    def setUp(self):
        self.plus_transcript = {
            "gene_id": "GENE1",
            "gene_id_base": "GENE1",
            "gene_name": "GENE1",
            "transcript_id": "TX1",
            "chrom": "chr1",
            "strand": "+",
            "exons": [(5000, 5100), (6000, 6100)],
        }
        self.minus_transcript = {**self.plus_transcript, "strand": "-"}

    def test_anchor_positions_respect_transcription_direction(self):
        self.assertEqual(anchor_position(self.plus_transcript, "TSS"), 5000)
        self.assertEqual(anchor_position(self.plus_transcript, "TES"), 6100)
        self.assertEqual(anchor_position(self.plus_transcript, "5SS", 0), 5100)
        self.assertEqual(anchor_position(self.plus_transcript, "3SS", 0), 6000)

        self.assertEqual(anchor_position(self.minus_transcript, "TSS"), 6100)
        self.assertEqual(anchor_position(self.minus_transcript, "TES"), 5000)
        self.assertEqual(anchor_position(self.minus_transcript, "5SS", 0), 6000)
        self.assertEqual(anchor_position(self.minus_transcript, "3SS", 0), 5100)

    def test_gene_relative_coordinates_flip_on_minus_strand(self):
        self.assertEqual(gene_relative_to_genome(5000, -1000, 0, "+"), (4000, 5000))
        self.assertEqual(gene_relative_to_genome(5000, -1000, 0, "-"), (5000, 6000))

    def test_each_landmark_expands_to_five_one_kilobase_windows(self):
        samples = build_gene_structure_samples(
            [self.plus_transcript],
            chrom_sizes={"chr1": 20000},
            input_length=2114,
            output_length=1000,
            max_ss_per_gene=20,
            train_chroms={"chr1"},
            validation_chroms=set(),
            test_chroms=set(),
            default_split="train",
        )

        self.assertEqual(len(samples), 20)
        for anchor_type, offsets in GENE_STRUCTURE_WINDOWS.items():
            selected = [sample for sample in samples if sample["anchor_type"] == anchor_type]
            self.assertEqual([sample["window_index"] for sample in selected], list(range(5)))
            self.assertEqual(
                [(sample["relative_start"], sample["relative_end"]) for sample in selected],
                offsets,
            )
            self.assertTrue(all(sample["output_end"] - sample["output_start"] == 1000 for sample in selected))
            self.assertTrue(all(sample["input_end"] - sample["input_start"] == 2114 for sample in selected))

    def test_junction_limit_is_evenly_spaced(self):
        self.assertEqual(junction_indexes(0, 20), [])
        self.assertEqual(junction_indexes(3, 20), [0, 1, 2])
        self.assertEqual(junction_indexes(30, 20)[0], 0)
        self.assertEqual(junction_indexes(30, 20)[-1], 29)
        self.assertEqual(len(junction_indexes(30, 20)), 20)

    def test_cli_requires_annotation_not_arbitrary_anchors(self):
        parser = argparse.ArgumentParser()
        add_prepare_bigwig_args(parser)
        option_names = {action.dest for action in parser._actions}
        self.assertIn("gtf", option_names)
        self.assertIn("chrom_sizes", option_names)
        self.assertNotIn("anchors_bed", option_names)


if __name__ == "__main__":
    unittest.main()
