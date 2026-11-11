"""Compare retained and bounded execution without displaying private model outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from sports_action_runtime import EVIDENCE
from prepare_sports_action_observations import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',choices=['observations_initial_memory_failure','observations_output_duration_interrupted'],default='observations_initial_memory_failure')
    args=parser.parse_args()
    old=EVIDENCE/args.prior
    new=EVIDENCE/'observations'
    before=json.loads((old/'progress.json').read_text())
    after={r['center']:r for r in json.loads((new/'progress.json').read_text())}
    maximum={k:0.0 for k in ('im_feat','team_feat','displ_feat')}
    exact=True
    for row in before:
        other=after[row['center']]
        if row['input_rgb_sha256']!=other['input_rgb_sha256']:
            raise ValueError('Execution input RGB identity differs')
        for folder,record in ((old,row),(new,other)):
            if file_sha(folder/record['file'])!=record['sha256']:
                raise ValueError('Retained output hash differs')
        with np.load(old/row['file']) as a,np.load(new/other['file']) as b:
            np.testing.assert_array_equal(a['frames'],b['frames'])
            for name in maximum:
                exact=exact and np.array_equal(a[name],b[name])
                maximum[name]=max(maximum[name],float(np.max(np.abs(a[name]-b[name]))))
    result=dict(compared_clips=len(before),input_rgb_hashes_exact=True,frame_identity_exact=True,
                all_head_arrays_exact=exact,maximum_absolute_differences=maximum,
                passed=all(value<=1e-6 for value in maximum.values()))
    (EVIDENCE/('execution_parity_'+args.prior+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit('Execution parity failed; stop before source support')


if __name__=='__main__':main()
