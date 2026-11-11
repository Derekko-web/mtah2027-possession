"""Preserve all preceding evidence and select the audited fragment repair."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automatic_score_ledger import append_scores
from research_supported_gap_stoppage import ROUND,verify
from research_spurious_possessions import digest,write
from reproduce_spurious_possessions import assess
from reproduce_boundary_precision import overlap_summary
from audit_stoppage_fragments import FAMILIES

PACKAGE=ROOT/'data/candidate_tabpfn_stoppage_fragments_v1'
LEDGER=ROOT/'experiments/EXPERIMENT_LEDGER.csv'
REPORT=ROOT/'experiments/SPURIOUS_POSSESSION_REPAIR_REPORT.md'
BACKUP=ROOT/'experiments/SELECTED_MODEL_BEFORE_SPURIOUS_REPAIR_20261107.md'


def copy_new(source,target):
    with target.open('xb') as handle:handle.write(source.read_bytes())


def promote():
    plan=verify();assert not PACKAGE.exists() and not BACKUP.exists()
    assert assess(ROUND/'candidates/supported_gap25')['passed']
    audit=json.loads((ROUND/'final_audit.json').read_text());assert audit['passed']
    assert '\nRan 20 tests ' in (ROUND/'affected_tests.log').read_text()
    assert (ROUND/'affected_tests.log').read_text().rstrip().endswith('OK')
    diagnostics=pd.read_csv(ROUND/'boundary_diagnostics.csv')
    assert len(diagnostics)==153 and (diagnostics.iou>=diagnostics.before_iou-1e-12).all()
    assert pd.read_csv(ROUND/'fold_scores.csv').passed.all()
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
            report=json.loads((ROUND/'official_scoring/reports'/(name+'.json')).read_text())
            table=pd.read_csv(path);overlap=overlap_summary(table.ground_truth,table.prediction)
            short=np.array([e-s<=125 for s,e,k in overlap['gt']])
            rows.append(dict(protocol=protocol,model=f'{family}_{variant}_{seed}',
                spellF1=report['spell_f1_mean'],macroF1=report['macro_f1'],predicted_possessions=report['n_pred_spells'],
                dead_recall=report['per_class']['D']['recall'],false_dead_frames=report['confusion'][0][2]+report['confusion'][1][2],
                short_live_matched_iou50=int(overlap['flags'][.5][short].sum())))
    assert len(rows)==39
    canonical=ROUND/'ledger_scores.csv'
    with canonical.open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    original_ledger=LEDGER.read_bytes();history=ROOT/'experiments/WORK_HISTORY.md';original_history=history.read_bytes()
    assert len(pd.read_csv(LEDGER))==1798
    sources=set(audit['source_sha256'])|set(manifest['source_sha256'])|{
        'experiments/promote_stoppage_fragment_candidate.py','experiments/automatic_score_ledger.py'}
    evidence=[ROUND/name for name in ('plan.json','selection.json','qualification_result.json','audit_plan.json',
        'final_audit.json','fold_scores.csv','boundary_diagnostics.csv','main_spurious_diagnostics.json',
        'affected_tests.log','ledger_scores.csv')]+[REPORT]
    promotion_plan=dict(model=manifest['model'],seed=42,
        source_sha256={name:digest(ROOT/name) for name in sorted(sources)},
        evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in evidence},
        predecessor_sha256={name:digest(ROOT/name) for name in ('experiments/SELECTED_MODEL.json',
            'experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv','experiments/WORK_HISTORY.md')},
        candidate_sha256=manifest['candidate_sha256'],added_dependent_score_rows=39,affected_tests=20,
        randomized_differential_cases=900,individual_true_overlaps_nonregressing=153,
        all_three_main_seeds_zero_spurious=True,all_seeds_are_adaptive_development_checks=True)
    write(ROUND/'promotion_plan.json',promotion_plan)
    frozen=ROUND/'promotion_source_freeze';frozen.mkdir()
    for name in sorted(sources):copy_new(ROOT/name,frozen/Path(name).name)
    copy_new(ROOT/'experiments/SELECTED_MODEL.json',ROUND/'previous_selected_model.json')
    copy_new(ROOT/'experiments/SELECTED_MODEL.md',BACKUP)
    copy_new(LEDGER,ROUND/'ledger_before.csv');copy_new(history,ROUND/'work_history_before.md')
    PACKAGE.mkdir()
    for name in ('candidate.csv','events.json','public_comparison.json'):copy_new(full/name,PACKAGE/name)
    copy_new(full/'manifest.json',PACKAGE/'full_export_manifest.json')
    for path in evidence:copy_new(path,PACKAGE/path.name)
    copy_new(ROUND/'promotion_plan.json',PACKAGE/'promotion_plan.json')
    manifest.update(candidate_path=str((PACKAGE/'candidate.csv').relative_to(ROOT)),
        report_path=str(REPORT.relative_to(ROOT)),source_sha256=promotion_plan['source_sha256'],
        evidence_sha256={p.name:digest(PACKAGE/p.name) for p in evidence},
        affected_tests=20,randomized_differential_cases=900,independent_OOF_replays=39,official_score_checks=41,
        all_previous_true_matches_retained_at_all_five_IoUs=True,individual_true_overlaps_nonregressing=153,
        clipped_fold_comparisons=39,clipped_fold_regressions=0,all_nine_no_added_false_D=True,
        all_three_main_seeds_zero_spurious=True,full_public_labels_exact=True,
        private_changes_verified=False,all_seeds_are_adaptive_development_checks=True)
    write(PACKAGE/'manifest.json',manifest)
    verify()
    result=append_scores(LEDGER,canonical,'round84_supported_stoppage_fragments_20261107',
        'paired_development_score',str(canonical.relative_to(ROOT)))
    assert result==dict(added=39,rows_after=1837,preceding_bytes_preserved=True)
    assert LEDGER.read_bytes().startswith(original_ledger)
    (ROOT/'experiments/SELECTED_MODEL.md').write_text('''# Selected model: TabPFN with supported stoppage-fragment repair

The selected recipe is `tabpfn_supported_stoppage_fragments_seed42`. Its 199319-row CSV is `data/candidate_tabpfn_stoppage_fragments_v1/candidate.csv`, SHA256 `4b1dc28231b38019644e4e943db39e5142580ba4695e7a3434f93a3e91abfeb4`.

Seed42 out-of-fold SpellF1 main/purged/longer is **1.000000/0.875676/0.821053**, compared with **0.918919/0.875676/0.800000**. All three main-validation seeds now match all 17 true possessions with zero extras at every scored IoU from 0.3 to 0.7. Main seed42 macroF1 improves 0.901583 to 0.932449 and accuracy 0.919600 to 0.942533. Other holdouts retain extras; the main ceiling is not perfect frame accuracy or verified private performance.

All nine primary/macroF1/accuracy/mean-overlap/five-IoU F1 comparisons are nonregressing. Every prior true match remains, all 153 individual true overlaps are nondecreasing, and all 39 fold primary/macro checks pass. Recent 33-/87-frame home overlaps are preserved. Existing D is retained, X is exact, and no false-D frames are added in the labelled comparisons.

The rule removes short unsupported owner flips next to existing stoppages. Entirely unseen flips use bounded visibility/prediction anchors and require eight frames of inferred live continuation; observed fragments require weak dead-head support and mostly stationary ball evidence. Both teams use the same rules; inference has no truth, protocol or target-frame input. It changes topology and may add D, as required to remove false live fragments. No new model fits or TabPFN queries are needed.

Twenty focused tests pass, including 900 randomized independent comparisons across every retained variant. Independent reconstruction verifies all 39 OOF trial outputs and the complete full CSV; 41 official score checks agree. Two identical full exports pass validation. The full file suppresses six unlabelled fragments, changing 327 frames. Public full-fit predictions and SpellF1/macroF1/accuracy remain exactly unchanged at 0.912500/0.991058/0.991200. Those 327 private changes are unverified.

The final guard followed a seed43 overlap failure, so every seed is adaptive development evidence. The 7500 public frames and 17 possessions have extensively informed the work; private effects and generalization remain unknown.

See [SPURIOUS_POSSESSION_REPAIR_REPORT.md](SPURIOUS_POSSESSION_REPAIR_REPORT.md) for methods, retained failures, evidence and reproduction. The preceding selection document is preserved exactly in [SELECTED_MODEL_BEFORE_SPURIOUS_REPAIR_20261107.md](SELECTED_MODEL_BEFORE_SPURIOUS_REPAIR_20261107.md). Prior candidates remain unchanged. All 39 dependent score rows append with preceding ledger bytes preserved.
''')
    with history.open('a') as handle:
        handle.write('''

## 7 November: supported stoppage fragments remove main spurious possessions

All three main-development seeds now have zero extras and all 17 true possessions matched at every scored IoU, giving SpellF1 1.0. Seed42 main/purged/longer improves 0.918919/0.875676/0.800000 to 1.000000/0.875676/0.821053. All nine primary/macro/accuracy/mean-overlap/five-IoU F1 comparisons, all prior match flags, all 153 individual true overlaps and 39 fold primary/macro checks are nonregressing. Current home boundary gains remain. One entirely unseen owner flip receives bounded live continuation/stoppage closure; two observed short dead-ball fragments are suppressed. Initial continuation, single-anchor and unguarded-union failures are retained. The final eight-frame support refusal followed a seed43 overlap failure, so all seeds are adaptive development checks. All 20 tests, including 900 randomized independent comparisons, pass. Independent reconstruction matches all 39 OOF files and the full CSV; 41 official score checks agree. Two identical full exports pass validation. Full public predictions remain exact; six unlabelled fragments totaling 327 frames become D, with existing D/X preserved. Private effects are unknown; remaining extras in other holdouts are not claimed fixed. No new fits or TabPFN queries. Selected `data/candidate_tabpfn_stoppage_fragments_v1/candidate.csv`, SHA256 `4b1dc28231b38019644e4e943db39e5142580ba4695e7a3434f93a3e91abfeb4`. Prior pointer/document/candidates preserved; 39 dependent ledger rows append 1798 to 1837 with preceding bytes intact. See SPURIOUS_POSSESSION_REPAIR_REPORT.md.
''')
    pointer=dict(model=manifest['model'],candidate_path=manifest['candidate_path'],
        candidate_sha256=manifest['candidate_sha256'],manifest_path=str((PACKAGE/'manifest.json').relative_to(ROOT)))
    (ROOT/'experiments/SELECTED_MODEL.json').write_text(json.dumps(pointer,indent=2)+'\n')
    assert history.read_bytes().startswith(original_history)
    assert digest(PACKAGE/'candidate.csv')==pointer['candidate_sha256']
    assert digest(BACKUP)==promotion_plan['predecessor_sha256']['experiments/SELECTED_MODEL.md']
    for name,expected in plan['protected_sha256'].items():
        if name not in ('experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv'):
            assert digest(ROOT/name)==expected,name
    for name,expected in promotion_plan['source_sha256'].items():assert digest(ROOT/name)==expected,name
    proof=dict(passed=True,selected=pointer,ledger=result,previous_candidates_unchanged=True,
        preceding_selection_document_exact=True,preceding_history_bytes_preserved=True,
        all_nine_score_and_target_checks_pass=True,all_three_main_seeds_zero_spurious=True,
        individual_true_overlaps_nonregressing=153,affected_tests=20,
        promotion_plan_sha256=digest(ROUND/'promotion_plan.json'),manifest_sha256=digest(PACKAGE/'manifest.json'))
    write(ROUND/'promotion_result.json',proof);copy_new(ROUND/'promotion_result.json',PACKAGE/'promotion_result.json')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':promote()
