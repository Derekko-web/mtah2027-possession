"""Separately frozen leading-gap guard; reuse unchanged scoring/selection gates."""
import argparse
import json
from pathlib import Path
import shutil
import research_home_boundaries as engine
from leading_gap_arrivals import refine,VARIANTS

ROOT=engine.ROOT
ROUND=ROOT/'data/control_leading_gap_boundaries_v1'
SOURCES=('experiments/research_leading_gap_boundaries.py','experiments/research_home_boundaries.py',
    'experiments/leading_gap_arrivals.py','experiments/source_aware_arrivals.py',
    'experiments/test_leading_gap_arrivals.py','experiments/test_source_aware_arrivals.py',
    'experiments/reproduce_home_boundary_precision.py','experiments/reproduce_boundary_precision.py',
    'experiments/possession_boundary_precision.py')
engine.ROUND=ROUND;engine.VARIANTS=VARIANTS;engine.refine=refine


def prepare():
    ROUND.mkdir(exist_ok=False)
    original=ROOT/'data/control_home_boundaries_v1'
    plan=json.loads((original/'plan.json').read_text())
    plan.update(hypotheses=['Restrict posterior recentering to short leading camera gaps,not internal/tail gaps,with later >=3owner contacts.',
        'Retain the independently supported cross-camera disagreement rule for complete intervals.'],
        variants=VARIANTS,leading_gap_rule='firstobservedoffset>0and<=posteriorradius;>=3current-owner broadcastcontacts;allotheroriginal score/target gates unchanged',
        controller_adapter='Reuse original run/score/gate/selection logic;only round,variant dictionary,inference callback and frozen verifier point to this separate plan.',
        source_sha256={p:engine.digest(ROOT/p) for p in SOURCES})
    for p in (original/'plan.json',original/'seed42_scores.csv'):
        plan['input_sha256'][str(p.relative_to(ROOT))]=engine.digest(p)
    engine.write(ROUND/'plan.json',plan);archive=ROUND/'source_freeze';archive.mkdir()
    for name in SOURCES:shutil.copy2(ROOT/name,archive/Path(name).name)


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('source_sha256','input_sha256','protected_sha256'):
        for name,sha in plan[group].items():
            if engine.digest(ROOT/name)!=sha:raise ValueError('Frozen dependency changed: '+name)
    return plan


engine.verify=verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:engine.run(a.seed)
    else:p.error('Choose --prepare or --seed')
