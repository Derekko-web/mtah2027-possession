"""Freeze the union of two separately retained one-second gap anchors."""
import argparse
import json
from pathlib import Path
import research_spurious_possessions as engine
from dual_anchor_stoppage import refine,VARIANTS

ROOT=engine.ROOT
ROUND=ROOT/'data/control_dual_anchor_stoppage_v1'
SOURCES=engine.SOURCES+('experiments/stoppage_gap_closure.py','experiments/observed_gap_stoppage.py',
    'experiments/dual_anchor_stoppage.py','experiments/test_dual_anchor_stoppage.py',
    'experiments/research_dual_anchor_stoppage.py')
engine.ROUND=ROUND;engine.refine=refine;engine.VARIANTS=VARIANTS


def prepare():
    ROUND.mkdir(exist_ok=False)
    prior=ROOT/'data/control_observed_gap_stoppage_v1'
    plan=json.loads((prior/'plan.json').read_text())
    plan.update(variants=VARIANTS,hypotheses=['Prediction-anchored and visibility-anchored one-second closures have complementary delays;their union preserves both boundaries within the original entirely unobserved flip.'],
        settings='Union of unchanged gap25 andsource_gap25 Ddecisions;unchanged original inputs,eligibility andobservedfragmentrule;allgatesunchanged.',
        source_sha256={p:engine.digest(ROOT/p) for p in SOURCES})
    for folder in ('control_spurious_gap_closure_v1','control_observed_gap_stoppage_v1'):
        for name in ('plan.json','seed42_scores.csv'):
            p=ROOT/'data'/folder/name;plan['input_sha256'][str(p.relative_to(ROOT))]=engine.digest(p)
    engine.write(ROUND/'plan.json',plan);frozen=ROUND/'source_freeze';frozen.mkdir()
    for name in SOURCES:(frozen/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('source_sha256','input_sha256','protected_sha256'):
        for name,expected in plan[group].items():
            if engine.digest(ROOT/name)!=expected:raise ValueError('Frozen dependency changed: '+name)
    return plan


engine.verify=verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:engine.run(a.seed)
    else:p.error('Choose --prepare or --seed')
