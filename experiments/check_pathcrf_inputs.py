"""Extra valid-input witnesses and team renaming, before native model inference."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from pathcrf_observations import window
from research_temporal_dead import write_json


def main():
    folder=ROOT/'data/control_pathcrf_v1';source=folder/'observation_support'
    values=np.load(source/'windows.npy');frames=np.load(source/'frames.npy');home=np.load(source/'left_team_is_home.npy')
    chosen=np.unique(np.linspace(0,len(frames)-1,min(9,len(frames))).astype(int));checks=0
    for n in (18935,19060):
        data=load_source('tactical',n);xy=data['positions'];xy[~data['valid']]=np.nan
        for i in chosen:
            value,left,reason=window(xy,data['teams'],int(frames[i]))
            if reason!='supported' or left!=home[i]:raise ValueError('Valid prefix witness changed')
            np.testing.assert_array_equal(value,values[i])
            renamed,renamed_home,_=window(xy,1-data['teams'],int(frames[i]))
            np.testing.assert_array_equal(value,renamed)
            if renamed_home==left:raise ValueError('Team rename failed to invert ownership mapping')
            checks+=1
    write_json(folder/'valid_input_checks.json',dict(passed=True,valid_feature_prefix_witnesses=checks,team_rename_exact=True,targets_loaded=False,model_run=False))
    print('Extra native input checks pass',checks,flush=True)


if __name__=='__main__':main()
