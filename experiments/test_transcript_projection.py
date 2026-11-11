import itertools
import unittest

import numpy as np

from transcript_projection import align_component, project_transcript, run_transcript


def objective(sequence, proposal, reference):
    return (int(np.count_nonzero(sequence != proposal)),
            int(np.count_nonzero(sequence != reference)))


class TranscriptProjectionTests(unittest.TestCase):
    def test_exact_lexicographic_optimum_against_exhaustive_sequences(self):
        for length in range(1, 7):
            sequences = [np.array(value) for value in itertools.product("HA", repeat=length)]
            for reference in sequences:
                transcript = run_transcript(reference)
                valid = [value for value in sequences if np.array_equal(run_transcript(value), transcript)]
                for proposal in sequences:
                    actual = align_component(reference, proposal)
                    expected = min(objective(value, proposal, reference) for value in valid)
                    self.assertEqual(objective(actual, proposal, reference), expected)
                    np.testing.assert_array_equal(run_transcript(actual), transcript)

    def test_identity_and_secondary_objective_prefer_reference(self):
        reference = np.array(list("HHHAAA"))
        np.testing.assert_array_equal(align_component(reference, reference), reference)
        # Boundaries after frames 2 and 4 each make one proposal error; the
        # secondary objective uniquely chooses the unchanged reference.
        tied_reference = np.array(list("HHAAA"))
        tied_proposal = np.array(list("HHAHA"))
        np.testing.assert_array_equal(align_component(tied_reference, tied_proposal), tied_reference)

    def test_equal_complete_costs_prefer_staying(self):
        reference = np.array(list("HHAA"))
        proposal = np.array(list("HAHA"))
        # HAAA and HHHA have equal primary and secondary Hamming costs. At
        # the final merge, staying in A retains the earlier H->A transition.
        np.testing.assert_array_equal(align_component(reference, proposal), list("HAAA"))

    def test_every_tiny_run_retained(self):
        reference = np.array(list("HHHAHHH"))
        result = align_component(reference, np.array(list("HHHHHHH")))
        np.testing.assert_array_equal(run_transcript(result), list("HAH"))
        self.assertEqual(int((result == "A").sum()), 1)
        np.testing.assert_array_equal(result, reference)

    def test_dead_unknown_and_decode_blocks_separate_alignments(self):
        reference = np.array(list("HHAA DAAHH XHHAA".replace(" ", "")))
        proposal = np.array(list("AAAA HHHHH AAAAA".replace(" ", "")))
        blocks = [(0, 4), (5, 9), (10, 12), (12, 14)]
        actual = project_transcript(reference, proposal, blocks)
        for start, end in blocks:
            np.testing.assert_array_equal(run_transcript(actual[start:end]), run_transcript(reference[start:end]))
        np.testing.assert_array_equal(actual[[4, 9]], reference[[4, 9]])
        np.testing.assert_array_equal(actual[10:12], list("HH"))
        np.testing.assert_array_equal(actual[12:14], list("AA"))

    def test_outside_blocks_remains_exact(self):
        reference = np.array(list("HHAAHHAA"))
        proposal = np.array(list("AAAAAAAA"))
        actual = project_transcript(reference, proposal, [(2, 6)])
        np.testing.assert_array_equal(actual[[0, 1, 6, 7]], reference[[0, 1, 6, 7]])

    def test_internal_dead_and_unknown_split_one_decode_block(self):
        reference = np.array(list("HHADAHAAXHA"))
        proposal = np.array(list("AAAHHHHHAHA"))
        actual = project_transcript(reference, proposal, [(0, len(reference))])
        for start, end in ((0, 3), (4, 8), (9, 11)):
            np.testing.assert_array_equal(actual[start:end], align_component(reference[start:end], proposal[start:end]))
        np.testing.assert_array_equal(actual[[3, 8]], reference[[3, 8]])

    def test_rejects_shapes_labels_and_overlapping_blocks(self):
        reference = np.array(list("HHAD"))
        with self.assertRaises(ValueError):
            project_transcript(reference, list("HHA"), [(0, 4)])
        with self.assertRaises(ValueError):
            project_transcript(reference, list("HHAQ"), [(0, 4)])
        with self.assertRaises(ValueError):
            project_transcript(reference, list("HDAD"), [(0, 4)])
        for blocks in ([(0, 3), (2, 4)], [(-1, 4)], [(0, 5)], [(0, 0)], [(0., 4)]):
            with self.subTest(blocks=blocks):
                with self.assertRaises(ValueError):
                    project_transcript(reference, reference, blocks)


if __name__ == "__main__":
    unittest.main()
