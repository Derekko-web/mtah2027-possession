"""Separately frozen bounded-gap refinement,with the original score gates."""
import argparse
import json
from pathlib import Path
import research_spurious_possessions as engine
from stoppage_gap_closure import VARIANTS,refine

ROOT=engine.ROOT
ROUND=ROOT/'data/control_spurious_gap_closure_v1'
SOURCES=engine.SOURCES+('experiments/stoppage_gap_closure.py','experiments/test_stoppage_gap_closure.py',
    'experiments/research_stoppage_gap_closure.py')
engine.ROUND=ROUND;engine.VARIANTS=VARIANTS;engine.refine=refine


def prepare():
    ROUND.mkdir(exist_ok=False)
    prior=ROOT/'data/control_spurious_possessions_v1'
    plan=json.loads((prior/'plan.json').read_text())
    plan.update(variants=VARIANTS,hypotheses=['Unseen owner flips at a stoppage require a bounded D-facing gap allowance;whole-run live continuation damages the preceding possession overlap.'],
        settings='Original combined rule,then13/25frames of each entirely unobserved flip next to original D becomeD;clip tooriginalflip;allothergatesunchanged.',
        source_sha256={p:engine.digest(ROOT/p) for p in SOURCES})
    for path in (prior/'plan.json',prior/'seed42_scores.csv'):
        plan['input_sha256'][str(path.relative_to(ROOT))]=engine.digest(path)
    engine.write(ROUND/'plan.json',plan);freeze=ROUND/'source_freeze';freeze.mkdir()
    for name in SOURCES:(freeze/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('source_sha256','input_sha256','protected_sha256'):
        for name,expected in plan[group].items():
            if engine.digest(ROOT/name)!=expected:raise ValueError('Frozen dependency changed: '+name)
    return plan


engine.verify=verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44))
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:engine.run(a.seed)
    else:p.error('Choose --prepare or --seed')
