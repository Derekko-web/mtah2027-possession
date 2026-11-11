"""The research ledger must retain exact inherited bytes and append once."""
from pathlib import Path
import tempfile
import unittest
from detector_ledger import append_score_rows


class DetectorLedgerTests(unittest.TestCase):
    def test_append_preserves_prior_bytes_and_rejects_changed_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger, scores = root/'ledger.csv', root/'scores.csv'
            original = b'round,record_type,protocol,model,stitched_spellF1,source,source_sha256\nold,original,p,m,0.1234567890123456789,original.csv,abc\n'
            ledger.write_bytes(original)
            scores.write_text('protocol,model,spellF1\np,new,0.75\n')
            first = append_score_rows(ledger, scores, 'new_round', 'new.csv')
            self.assertEqual(first['added'], 1)
            self.assertTrue(ledger.read_bytes().startswith(original))
            saved = ledger.read_bytes()
            self.assertEqual(append_score_rows(ledger, scores, 'new_round', 'new.csv')['added'], 0)
            self.assertEqual(ledger.read_bytes(), saved)
            scores.write_text('protocol,model,spellF1\np,new,0.76\n')
            with self.assertRaises(ValueError):
                append_score_rows(ledger, scores, 'new_round', 'new.csv')
            self.assertEqual(ledger.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
