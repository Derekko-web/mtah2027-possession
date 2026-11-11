"""Synthetic checks for reviewed supervision and training-only weighting."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from human_annotation_training import reviewed_rows,fitting_arrays,append_scores


class HumanTrainingTests(unittest.TestCase):
    def annotation(self):
        window=dict(window_id='window_01',split='train',start=1000,end=1005)
        document=dict(**window,source_start=1000,source_end=1005,freeze_sha256='freeze',clip_sha256='clip',fps=25,
                      label_source='human_video',status='reviewed',annotator='One',reviewer='Two',
                      segments=[dict(start=0,end=1,label='H'),dict(start=2,end=3,label='U',note='Occluded'),dict(start=4,end=5,label='A')])
        return document,window

    def test_uncertain_rows_excluded_and_native_coordinates_exact(self):
        document,window=self.annotation()
        self.assertEqual(reviewed_rows(document,window,'freeze','clip'),[(1000,'H','window_01'),(1001,'H','window_01'),(1004,'A','window_01'),(1005,'A','window_01')])

    def test_audit_draft_unreviewed_and_provenance_changes_rejected(self):
        for field,value in [('split','audit'),('status','draft'),('reviewer','one'),('source_start',1001),('clip_sha256','other')]:
            document,window=self.annotation();document[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):reviewed_rows(document,window,'freeze','clip')

    def test_gaps_overlap_and_missing_uncertainty_reason_rejected(self):
        for mutation in ('gap','overlap','note'):
            document,window=self.annotation()
            if mutation=='gap':document['segments'][0]['end']=0
            elif mutation=='overlap':document['segments'][1]['start']=1
            else:document['segments'][1]['note']=''
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):reviewed_rows(document,window,'freeze','clip')

    def fitting(self,labels=None):
        if labels is None:labels=np.array(['H','A','D','H','A','D']+['X']*994)
        return fitting_arrays(labels,np.arange(6),np.array([900,901,902]),np.array(['H','A','D']),np.array([400,401]))

    def test_source_weights_keep_public_mass_and_balance_each_source(self):
        arrays=self.fitting();y,w=arrays['target'],arrays['weight']
        self.assertAlmostEqual(w[12:].sum()/w[:12].sum(),.5)
        for start,end in ((0,12),(12,18)):
            masses=[w[start:end][y[start:end]==label].sum() for label in range(3)]
            np.testing.assert_allclose(masses,np.repeat(masses[0],3))

    def test_held_labels_cannot_change_targets_or_weights(self):
        labels=np.array(['H','A','D','H','A','D']+['X']*994)
        a=self.fitting(labels);labels[300:700]='D';b=self.fitting(labels)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_inclusive_buffer_and_duplicate_human_rows_rejected(self):
        labels=np.array(['H','A','D']+['X']*997)
        for frames in ([650,900,901],[900,900,901]):
            with self.subTest(frames=frames),self.assertRaises(ValueError):
                fitting_arrays(labels,[0,1,2],frames,['H','A','D'],[400])

    def test_ledger_append_is_idempotent_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'experiments').mkdir();(root/'data').mkdir()
            ledger=root/'experiments/ledger.csv';scores=root/'data/scores.csv'
            fields=['round','record_type','protocol','model','source','source_sha256','stitched_spellF1','macroF1']
            stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
            writer.writerow(dict(round='old',protocol='test',model='old',source='old',stitched_spellF1='.5'))
            ledger.write_text(stream.getvalue());original=ledger.read_bytes()
            scores.write_text('protocol,model,spellF1,macroF1\ntest,human,.6,.7\n')
            self.assertEqual(append_scores(ledger,scores,'human_v1')['added'],1)
            after=ledger.read_bytes();self.assertTrue(after.startswith(original))
            self.assertEqual(append_scores(ledger,scores,'human_v1')['added'],0)
            self.assertEqual(ledger.read_bytes(),after)


if __name__=='__main__':unittest.main()
