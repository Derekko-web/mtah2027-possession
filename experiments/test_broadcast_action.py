"""Synthetic source/timeline/leakage contracts; no competition media or targets."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import pandas as pd
from broadcast_action_sampling import schedule, source_centers, source_ranges, selector, mapping_components, validate_video_metadata
from broadcast_action_features import dense_actions, feature_view, support_summary
from sports_action_sampling import clip_frames


class BroadcastActionTests(unittest.TestCase):
    def test_mixed_parity_union_and_independent_ffmpeg_positions(self):
        needed = np.unique(np.r_[clip_frames(110), clip_frames(131), clip_frames(170)])
        runs = source_ranges(needed)
        rebuilt = np.unique(np.concatenate([np.arange(a,b+1,2) for a,b in runs]))
        np.testing.assert_array_equal(needed, rebuilt)
        expression = selector(needed)
        # Synthetic source gray intensity encodes its frame index modulo256.
        first, last = int(needed[0]), int(needed[-1])
        command = ['ffmpeg','-v','error','-f','lavfi','-i',
                   f'nullsrc=s=4x4:r=25:d={(last-first+1)/25}', '-vf',
                   f"geq=lum='mod(N+{first},256)':cb=128:cr=128,select='{expression}'",
                   '-fps_mode','passthrough','-pix_fmt','gray','-f','rawvideo','pipe:1']
        values = np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, np.uint8).reshape(-1,16)
        # yuv limited-range -> full-range conversion; compare against unselected synthetic decode.
        plain = command.copy(); plain[plain.index('-vf')+1] = f"geq=lum='mod(N+{first},256)':cb=128:cr=128"
        all_values = np.frombuffer(subprocess.run(plain,capture_output=True,check=True).stdout,np.uint8).reshape(-1,16)
        np.testing.assert_array_equal(values, all_values[needed-first])

    def test_broadcast_container_count_is_distinct_from_tracking_length(self):
        metadata=dict(width=1920,height=1080,nb_frames='164487',r_frame_rate='25/1',start_time='0.000000')
        self.assertEqual(validate_video_metadata(metadata),164487)
        with self.assertRaises(ValueError):
            validate_video_metadata(dict(metadata,nb_frames='164488'))

    def test_duplicates_and_source_bounds(self):
        self.assertEqual(source_centers([dict(broadcast_center=33),dict(broadcast_center=10),dict(broadcast_center=33)]),[10,33])
        n=19060; mapping={'frames':np.arange(n), 'orientation':np.zeros(n,int)}
        rows=schedule(n,mapping,[('half',11185,20000)],[dict(start=13000,end=13010)],np.array([15000]))
        self.assertTrue(rows)
        for r in rows:
            c=r['tactical_center']
            self.assertTrue(c-150>13260 or c+150<12750)
            self.assertTrue(abs(c-15000)>400)
        with self.assertRaisesRegex(ValueError,'bounds'):
            schedule(n,mapping,[('half',11185,20000)],[],[],broadcast_n=100)

    def test_mapped_interpolation_tie_and_missing_cut(self):
        n=400; mapped=np.arange(n)+1000
        mapping={'frames':mapped,'orientation':np.zeros(n,int)}
        mapping['frames'][190:210]=-1
        mapping['orientation'][190:210]=-1
        rows=[dict(tactical_center=100,broadcast_center=1100),dict(tactical_center=150,broadcast_center=1150),dict(tactical_center=225,broadcast_center=1225)]
        with tempfile.TemporaryDirectory() as tmp:
            records=[]
            for r in rows:
                c=r['broadcast_center']; frames=clip_frames(c)
                logits=np.zeros((100,31),np.float32);logits[:,1]=(frames-c)/100
                name=f'{c}.npz'
                np.savez(Path(tmp)/name,frames=frames,im_feat=logits,displ_feat=frames.astype(float),team_feat=np.zeros(100))
                records.append(dict(center=c,file=name))
            # Non-unit mapping at query verifies actual broadcast-time interpolation.
            mapping['frames'][101]=1102
            dense,sources,components=dense_actions(tmp,records,rows,mapping,n,[('half',0,n-1)])
            self.assertEqual(float(dense.sa_displacement.iloc[101]),1102.)
            self.assertEqual(sources[125],1100)  # nearest tactical center tie picks earlier.
            self.assertEqual(dense.sa_seen.iloc[200],0)
            self.assertTrue(np.isnan(dense.sa_pass.iloc[200]))
            view=feature_view(dense,components)
            self.assertEqual(len(view.columns),121)
            self.assertTrue(view.iloc[200].drop('sa_seen').isna().all())
            # Context cannot bridge unobserved queries or mapping cuts.
            self.assertAlmostEqual(view.sa_pass_future50_mean.iloc[175],dense.sa_pass.iloc[175],places=6)

    def test_components_stop_at_orientation_reversal_jump_and_half(self):
        mapping={'frames':np.array([10,11,12,13,14,15,25,26,27,26]),'orientation':np.array([0,0,0,1,1,1,1,1,1,1])}
        c=mapping_components(mapping,10,[('h1',0,7),('h2',8,9)])
        for a,b in [(2,3),(5,6),(7,8),(8,9)]:self.assertNotEqual(c[a],c[b])

    def test_full_future_features_prefix_not_only_centers(self):
        mapping={'frames':np.arange(19060),'orientation':np.zeros(19060,int)}
        blocks=[('half',11185,30000)]
        with tempfile.TemporaryDirectory() as tmp:
            records=[]
            full_rows=schedule(19060,mapping,blocks,[],[])
            for r in full_rows:
                c=r['broadcast_center']; name=f'{c}.npz'
                logits=np.zeros((100,31));logits[:,1]=np.sin(np.arange(100)/3)
                np.savez(Path(tmp)/name,frames=clip_frames(c),im_feat=logits,displ_feat=np.arange(100),team_feat=np.zeros(100))
                records.append(dict(center=c,file=name))
            views=[]
            for n in (18935,19060):
                dense,_,comp=dense_actions(tmp,records,schedule(n,mapping,blocks,[],[]),mapping,n,blocks)
                views.append(feature_view(dense,comp).iloc[11185:18685])
            pd.testing.assert_frame_equal(*views,check_exact=True)

    def test_support_counts_unique_source_clips(self):
        dense=pd.DataFrame(dict(sa_seen=np.ones(200),sa_event_max=np.tile([.1,.9,.1,.1],50),sa_entropy=np.ones(200)))
        rule=dict(public_start=0,public_stop=200,min_observed_frames=100,min_confident_local_peaks=3,peak_threshold=.3,peak_separation=25,min_unique_peak_source_clips=3,min_observed_event_max_std=.01)
        fail=support_summary(dense,np.ones(200,int),rule)
        self.assertFalse(fail['support_passed']);self.assertEqual(fail['unique_peak_source_clips'],1)
        passed=support_summary(dense,np.arange(200),rule)
        self.assertTrue(passed['support_passed'])


if __name__=='__main__':unittest.main()
