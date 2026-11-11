"""Select the audited zero-extra recipe and preserve preceding research bytes."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automatic_score_ledger import append_scores
from research_bounded_holdout_repair import ROUND,verify
from research_holdout_extras import digest,write
from reproduce_holdout_extras import assess
from reproduce_boundary_precision import overlap_summary
from audit_holdout_extras import FAMILIES

PACKAGE=ROOT/'data/candidate_tabpfn_holdout_repair_v1'
LEDGER=ROOT/'experiments/EXPERIMENT_LEDGER.csv'
REPORT=ROOT/'experiments/HOLDOUT_EXTRAS_REPAIR_REPORT.md'
BACKUP=ROOT/'experiments/SELECTED_MODEL_BEFORE_HOLDOUT_REPAIR_20261107.md'


def copy_new(source,target):
    with target.open('xb') as handle:handle.write(source.read_bytes())


def promote():
    plan=verify();assert not PACKAGE.exists() and not BACKUP.exists()
    assert assess(ROUND/'candidates/bounded_combined')['passed']
    audit=json.loads((ROUND/'final_audit.json').read_text());assert audit['passed']
    tests=(ROUND/'affected_tests.log').read_text();assert '\nRan 14 tests ' in tests and tests.rstrip().endswith('OK')
    diagnostic=pd.read_csv(ROUND/'boundary_diagnostics.csv')
    assert len(diagnostic)==153 and (diagnostic.iou>=diagnostic.before_iou-1e-12).all()
    folds=pd.read_csv(ROUND/'fold_scores.csv');assert len(folds)==39 and folds.passed.all()
    full=ROUND/'full_export1';manifest=json.loads((full/'manifest.json').read_text())
    assert manifest['full_public_checks_pass'] and manifest['changed_public_frames']==0
    assert manifest['candidate_sha256']==digest(full/'candidate.csv')==audit['full_csv_sha256']
    for name in ('candidate.csv','events.json','manifest.json','public_comparison.json'):
        assert (full/name).read_bytes()==(ROUND/'full_export2'/name).read_bytes()
    for group in ('source_sha256','input_sha256'):
        for name,expected in manifest[group].items():assert digest(ROOT/name)==expected,name
    for name,expected in audit['source_sha256'].items():assert digest(ROOT/name)==expected,name
    rows=[]
    for family,folder in FAMILIES:
        source=ROOT/'data'/folder/'candidates'
        for path in sorted(source.glob('*/seed*/*/oof_predictions.csv')):
            variant,seed,protocol=path.relative_to(source).parts[:3]
            name=f'{family}_{variant}_{seed}_{protocol}'
            metric=json.loads((ROUND/'official_scoring/reports'/(name+'.json')).read_text())
            table=pd.read_csv(path);overlap=overlap_summary(table.ground_truth,table.prediction)
            short=np.array([e-s<=125 for s,e,k in overlap['gt']])
            rows.append(dict(protocol=protocol,model=f'{family}_{variant}_{seed}',spellF1=metric['spell_f1_mean'],
                macroF1=metric['macro_f1'],predicted_possessions=metric['n_pred_spells'],dead_recall=metric['per_class']['D']['recall'],
                false_dead_frames=metric['confusion'][0][2]+metric['confusion'][1][2],
                short_live_matched_iou50=int(overlap['flags'][.5][short].sum())))
    assert len(rows)==30
    canonical=ROUND/'ledger_scores.csv'
    with canonical.open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    original_ledger=LEDGER.read_bytes();history=ROOT/'experiments/WORK_HISTORY.md';original_history=history.read_bytes()
    assert len(pd.read_csv(LEDGER))==1837
    sources=set(audit['source_sha256'])|set(manifest['source_sha256'])|{
        'experiments/promote_holdout_repair_candidate.py','experiments/automatic_score_ledger.py'}
    evidence=[ROUND/name for name in ('plan.json','selection.json','qualification_result.json','audit_plan.json','final_audit.json',
        'fold_scores.csv','boundary_diagnostics.csv','affected_tests.log','ledger_scores.csv')]+[REPORT]
    promotion_plan=dict(model=manifest['model'],seed=42,source_sha256={p:digest(ROOT/p) for p in sorted(sources)},
        evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in evidence},
        predecessor_sha256={p:digest(ROOT/p) for p in ('experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md',
            'experiments/EXPERIMENT_LEDGER.csv','experiments/WORK_HISTORY.md')},candidate_sha256=manifest['candidate_sha256'],
        added_dependent_score_rows=30,affected_tests=14,randomized_differential_cases=800,
        all_nine_OOF_spellF1=1.,all_nine_matched=17,all_nine_spurious=0,
        individual_true_overlaps_nonregressing=153,all_results_are_adaptive_development_checks=True)
    write(ROUND/'promotion_plan.json',promotion_plan);freeze=ROUND/'promotion_source_freeze';freeze.mkdir()
    for p in sorted(sources):copy_new(ROOT/p,freeze/Path(p).name)
    copy_new(ROOT/'experiments/SELECTED_MODEL.json',ROUND/'previous_selected_model.json')
    copy_new(ROOT/'experiments/SELECTED_MODEL.md',BACKUP)
    copy_new(LEDGER,ROUND/'ledger_before.csv');copy_new(history,ROUND/'work_history_before.md')
    PACKAGE.mkdir()
    for name in ('candidate.csv','events.json','public_comparison.json'):copy_new(full/name,PACKAGE/name)
    copy_new(full/'manifest.json',PACKAGE/'full_export_manifest.json')
    for p in evidence:copy_new(p,PACKAGE/p.name)
    copy_new(ROUND/'promotion_plan.json',PACKAGE/'promotion_plan.json')
    manifest.update(candidate_path=str((PACKAGE/'candidate.csv').relative_to(ROOT)),report_path=str(REPORT.relative_to(ROOT)),
        source_sha256=promotion_plan['source_sha256'],evidence_sha256={p.name:digest(PACKAGE/p.name) for p in evidence},
        affected_tests=14,randomized_differential_cases=800,independent_OOF_replays=30,official_score_checks=32,
        all_previous_true_matches_retained_at_all_five_IoUs=True,individual_true_overlaps_nonregressing=153,
        clipped_fold_comparisons=39,clipped_fold_regressions=0,all_nine_dead_recall_nonregressing=True,
        all_nine_false_D_count_nonincreasing=True,full_public_labels_exact=True,
        private_changes_verified=False,all_results_are_adaptive_development_checks=True)
    write(PACKAGE/'manifest.json',manifest)
    verify()
    ledger_result=append_scores(LEDGER,canonical,'round85_bounded_holdout_extras_20261107',
        'paired_development_score',str(canonical.relative_to(ROOT)))
    assert ledger_result==dict(added=30,rows_after=1867,preceding_bytes_preserved=True)
    assert LEDGER.read_bytes().startswith(original_ledger)
    (ROOT/'experiments/SELECTED_MODEL.md').write_text('''# Selected model: TabPFN with physical and continuous-control repairs

The selected recipe is `tabpfn_physical_continuity_repair_seed42`. Its 199319-row CSV is `data/candidate_tabpfn_holdout_repair_v1/candidate.csv`, SHA256 `400c86edb5ba25bf0f9c40b29a7c8d1d305b0a4d4b7fd3ebc78a0fb04efe415b`.

All nine seed/protocol OOF comparisons now have **SpellF1 1.000000**, all 17 true possessions matched and zero extras/misses at every scored IoU from 0.3 to 0.7. Seed42 main/purged/longer improves **1.000000/0.875676/0.821053 → 1.000000/1.000000/1.000000**. This is the ceiling of these development checks, not perfect boundaries or verified private performance.

All nine macroF1/accuracy/five-IoU F1 comparisons, all previous match flags, all 153 individual true overlaps and all 39 fold primary/macro checks are nonregressing. Main/purged/longer seed42 macroF1 is 0.938290/0.934610/0.955858 and accuracy 0.947067/0.944667/0.961467. Dead recall never decreases and false-D counts never increase in any labelled pair. X is exact; D may grow or shrink to repair both false live fragments and false-dead splits.

The uniform inference rules use pitch exit evidence, supported continuity across short D gaps, bounded unseen-gap allowances, contact bouts, cross-source inward trims and supported release tails. The bounded wrapper preserves live prefixes before pitch exit and permits intrinsic inward trims at decode edges. No truth, seed, protocol or target-frame exception is an inference input. No new fits or TabPFN queries are required.

All 14 tests pass, including 800 randomized independent comparisons. Independent reconstruction matches all 30 retained trial OOF files and the complete full CSV; 32 official score checks agree. Two identical full exports pass validation. Full public predictions and scores remain exactly unchanged at 0.912500/0.991058/0.991200 SpellF1/macroF1/accuracy. The full file changes 832 unlabelled frames through 43 recorded updates; those changes and their private score effects remain unverified.

The 7500 public frames and 17 true possessions have extensively informed many adaptive rounds. Reused seed/protocol checks are not a new independent match or private evaluation. Further primary improvement cannot be demonstrated on these now-saturated comparisons; exact frame labels still have errors.

See [HOLDOUT_EXTRAS_REPAIR_REPORT.md](HOLDOUT_EXTRAS_REPAIR_REPORT.md) for diagnosis, preserved failures, evidence and reproduction. The preceding selection document is preserved exactly in [SELECTED_MODEL_BEFORE_HOLDOUT_REPAIR_20261107.md](SELECTED_MODEL_BEFORE_HOLDOUT_REPAIR_20261107.md). Previous candidate packages remain unchanged. All 30 dependent score rows append with preceding ledger bytes intact.
''')
    with history.open('a') as handle:
        handle.write('''

## 7 November: purged and longer extras removed in development checks

All nine seed/protocol OOF comparisons now have SpellF1 1.0, all 17 true possessions matched and zero extras/misses at every scored IoU from 0.3 to 0.7. Seed42 main/purged/longer improves 1.000000/0.875676/0.821053 to 1.000000/1.000000/1.000000. Every primary/macro/accuracy/five-IoU score, prior match flag, all 153 individual true overlaps and 39 fold primary/macro comparisons are nonregressing. Dead recall is nondecreasing and false-D counts nonincreasing; X remains exact. Physical pitch-exit closure, moving-control D-gap bridges, bounded unseen closure, contact-bout/cross-source inward trims and supported releases repair different error causes. The initial full live-gap suppression failed; a separately frozen wrapper preserves live prefixes before outside cores and handles intrinsic control at decode edges. All 14 tests, including 800 independent randomized comparisons, pass. Independent reconstruction matches all 30 OOF files and the complete full CSV; 32 official scores agree. Both identical full exports pass validation. Full public predictions/scores remain exact; 832 unlabelled frames change through 43 recorded updates, with private effects unknown. No new fits or TabPFN queries. All seed/protocol checks remain adaptive same-match development; 1.0 saturates these checks, not private performance or exact frame labels. Selected `data/candidate_tabpfn_holdout_repair_v1/candidate.csv`, SHA256 `400c86edb5ba25bf0f9c40b29a7c8d1d305b0a4d4b7fd3ebc78a0fb04efe415b`. Prior pointer/document/candidates preserved; 30 dependent ledger rows append 1837 to 1867 with preceding bytes intact. See HOLDOUT_EXTRAS_REPAIR_REPORT.md.
''')
    pointer=dict(model=manifest['model'],candidate_path=manifest['candidate_path'],candidate_sha256=manifest['candidate_sha256'],
        manifest_path=str((PACKAGE/'manifest.json').relative_to(ROOT)))
    (ROOT/'experiments/SELECTED_MODEL.json').write_text(json.dumps(pointer,indent=2)+'\n')
    assert history.read_bytes().startswith(original_history)
    assert digest(PACKAGE/'candidate.csv')==pointer['candidate_sha256']
    assert digest(BACKUP)==promotion_plan['predecessor_sha256']['experiments/SELECTED_MODEL.md']
    for name,expected in plan['protected_sha256'].items():
        if name not in ('experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv'):
            assert digest(ROOT/name)==expected,name
    for name,expected in promotion_plan['source_sha256'].items():assert digest(ROOT/name)==expected,name
    proof=dict(passed=True,selected=pointer,ledger=ledger_result,previous_candidates_unchanged=True,
        preceding_selection_document_exact=True,preceding_history_bytes_preserved=True,
        all_nine_OOF_spellF1=1.,all_nine_matched=17,all_nine_spurious=0,
        individual_true_overlaps_nonregressing=153,affected_tests=14,
        promotion_plan_sha256=digest(ROUND/'promotion_plan.json'),manifest_sha256=digest(PACKAGE/'manifest.json'))
    write(ROUND/'promotion_result.json',proof);copy_new(ROUND/'promotion_result.json',PACKAGE/'promotion_result.json')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':promote()
