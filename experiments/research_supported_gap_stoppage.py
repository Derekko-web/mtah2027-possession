"""Freeze a live-support refusal after retaining the failed seed43 overlap."""
import argparse
import json
from pathlib import Path
import research_spurious_possessions as engine
from supported_gap_stoppage import refine,VARIANTS

ROOT=engine.ROOT
ROUND=ROOT/'data/control_supported_gap_stoppage_v1'
SOURCES=engine.SOURCES+('experiments/stoppage_gap_closure.py','experiments/observed_gap_stoppage.py',
    'experiments/dual_anchor_stoppage.py','experiments/supported_gap_stoppage.py',
    'experiments/test_stoppage_gap_closure.py','experiments/test_observed_gap_stoppage.py',
    'experiments/test_dual_anchor_stoppage.py','experiments/test_supported_gap_stoppage.py',
    'experiments/research_supported_gap_stoppage.py')
engine.ROUND=ROUND;engine.refine=refine;engine.VARIANTS=VARIANTS


def prepare():
    ROUND.mkdir(exist_ok=False)
    prior=ROOT/'data/control_dual_anchor_stoppage_v1'
    plan=json.loads((prior/'plan.json').read_text())
    plan.update(variants=VARIANTS,hypotheses=['Unseen gap continuation must contribute >=8frames of its own live support;otherwise decline the closure rather than borrowing confidence from a long neighboring run.'],
        settings='Unchanged dual-anchor25;refuseunseenproposalwhenremainingliveportion<8frames;observedstoppageunchanged;allpreviousgatesunchanged.',
        limitation='All seeds now inform adaptive development after seed43 mean-overlap failure;these are paired development checks,not fresh independent statistical confirmations. Private effects unknown.',
        source_sha256={p:engine.digest(ROOT/p) for p in SOURCES})
    for name in ('plan.json','seed42_scores.csv','seed43_scores.csv','seed44_scores.csv','qualification_result.json'):
        path=prior/name;plan['input_sha256'][str(path.relative_to(ROOT))]=engine.digest(path)
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
