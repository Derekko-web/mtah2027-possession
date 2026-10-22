"""Reproduce selected development predictions from the supplied match tracking.

Run from the project root: .venv/bin/python experiments/run_selected.py
For a full-match candidate afterwards: .venv/bin/python experiments/build_candidate.py
"""

from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from rich_features import build_features


def main():
    cache=ROOT/'data/rich_baseline/input_features.pkl'
    if not cache.exists():
        labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv')
        end=int(labels.loc[labels.label!='X','frame'].max())+1
        cache.parent.mkdir(exist_ok=True)
        build_features(end).to_pickle(cache)
    for arguments in [
        ['train_kinematics_baseline.py','--relative-motion'],
        ['train_kinematics_baseline.py','--relative-motion','--segment-weights'],
        ['evaluate_selected.py'],
    ]:
        subprocess.run([sys.executable,str(ROOT/'experiments'/arguments[0]),*arguments[1:]],
                       cwd=ROOT,check=True)


if __name__=='__main__':main()
