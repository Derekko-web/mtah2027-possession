"""Freeze two causal guard repairs while retaining the original failed screen."""
import argparse
import json
from pathlib import Path
import research_holdout_extras as engine
from bounded_holdout_repair import refine,VARIANTS

ROOT=engine.ROOT;ROUND=ROOT/'data/control_bounded_holdout_repair_v1'
SOURCES=engine.SOURCES+('experiments/bounded_holdout_repair.py','experiments/test_bounded_holdout_repair.py',
    'experiments/research_bounded_holdout_repair.py')
engine.ROUND=ROUND;engine.refine=refine;engine.VARIANTS=VARIANTS


def prepare():
    ROUND.mkdir(exist_ok=False);prior=ROOT/'data/control_holdout_extras_v1'
    plan=json.loads((prior/'plan.json').read_text())
    plan.update(variants=VARIANTS,hypotheses=['Preserve the live prefix when an outside-pitch core starts late in a D/live/D interval.',
        'Cross-source inward trims are intrinsically supported at decode edges and need only the one available opposing live neighbor.'],
        source_sha256={p:engine.digest(ROOT/p) for p in SOURCES})
    for name in ('plan.json','seed42_scores.csv'):
        p=prior/name;plan['input_sha256'][str(p.relative_to(ROOT))]=engine.digest(p)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));args=p.parse_args()
    if args.prepare:prepare()
    elif args.seed:engine.run(args.seed)
    else:p.error('Choose --prepare or --seed')
