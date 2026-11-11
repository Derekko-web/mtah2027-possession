"""Preserve the predecessor and select the fully audited home-arrival recipe."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automatic_score_ledger import append_scores
from research_leading_gap_boundaries import ROUND, verify
from reproduce_home_boundary_precision import assess
from reproduce_boundary_precision import overlap_summary

PACKAGE = ROOT/'data/candidate_tabpfn_home_arrivals_v1'
LEDGER = ROOT/'experiments/EXPERIMENT_LEDGER.csv'
REPORT = ROOT/'experiments/HOME_BOUNDARY_REFINEMENT_REPORT.md'
BACKUP = ROOT/'experiments/SELECTED_MODEL_BEFORE_HOME_ARRIVALS_20261107.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True)+'\n')


def copy_new(source, target):
    with target.open('xb') as handle:
        handle.write(source.read_bytes())


def promote():
    plan = verify()
    assert not PACKAGE.exists() and not BACKUP.exists()
    assert assess(ROUND/'candidates/leading_camera13')['passed']
    audit = json.loads((ROUND/'final_audit.json').read_text())
    public_check = json.loads((ROUND/'full_public_check.json').read_text())
    assert audit['passed'] and public_check['passed']
    tests = ROUND/'additional_tests.log'
    assert '\nRan 26 tests ' in tests.read_text() and tests.read_text().rstrip().endswith('OK')
    full = ROUND/'full_export1'
    manifest = json.loads((full/'manifest.json').read_text())
    assert manifest['candidate_sha256'] == audit['full_csv_sha256'] == sha(full/'candidate.csv')
    for name in ('candidate.csv', 'arrival_events.json', 'manifest.json'):
        assert (full/name).read_bytes() == (ROUND/'full_export2'/name).read_bytes()
    for name, expected in manifest['input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in audit['source_sha256'].items():
        assert sha(ROOT/name) == expected, name

    parent = pd.read_csv(ROOT/manifest['parent_candidate_path'])
    candidate = pd.read_csv(full/'candidate.csv')
    truth = pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label
    old = overlap_summary(truth, parent.label)
    new = overlap_summary(truth, candidate.label)
    assert all(new['flags'][t][old['flags'][t]].all() for t in old['flags'])
    assert new['mean_best_iou'] >= old['mean_best_iou']-1e-12

    # Retain every scored branch, including the failed broad-source screen.
    rows = []
    for family, root in (('initial', ROOT/'data/control_home_boundaries_v1/candidates'),
                         ('leading', ROUND/'candidates')):
        for path in sorted(root.glob('*/seed*/*/oof_predictions.csv')):
            variant, seed, protocol = path.relative_to(root).parts[:3]
            name = f'{family}_{variant}_{seed}_{protocol}'
            official = ROUND/'official_scoring/reports'/(name+'.json')
            metric = json.loads(official.read_text())
            table = pd.read_csv(path)
            overlap = overlap_summary(table.ground_truth, table.prediction)
            short = np.array([e-s <= 125 for s,e,k in overlap['gt']])
            rows.append(dict(protocol=protocol, model=f'{family}_{variant}_{seed}',
                spellF1=metric['spell_f1_mean'], macroF1=metric['macro_f1'],
                predicted_possessions=metric['n_pred_spells'],
                dead_recall=metric['per_class']['D']['recall'],
                false_dead_frames=metric['confusion'][0][2]+metric['confusion'][1][2],
                short_live_matched_iou50=int(overlap['flags'][.5][short].sum())))
    assert len(rows) == 30
    canonical = ROUND/'ledger_scores.csv'
    with canonical.open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    original_ledger = LEDGER.read_bytes()
    original_history = (ROOT/'experiments/WORK_HISTORY.md').read_bytes()
    assert len(pd.read_csv(LEDGER)) == 1768
    sources = set(plan['source_sha256']) | set(audit['source_sha256']) | {
        'experiments/test_home_boundary_audit.py',
        'experiments/test_possession_boundary_precision.py',
        'experiments/test_probability_guarded_boundaries.py',
        'experiments/test_guarded_boundary_audit.py',
        'experiments/promote_home_arrival_candidate.py',
        'experiments/automatic_score_ledger.py',
    }
    evidence = [ROUND/name for name in ('plan.json', 'selection.json', 'qualification_result.json',
        'final_audit.json', 'fold_scores.csv', 'boundary_diagnostics.csv', 'full_public_check.json',
        'additional_tests.log', 'ledger_scores.csv')] + [REPORT]
    promotion_plan = dict(model=manifest['model'], seed=42,
        source_sha256={p:sha(ROOT/p) for p in sorted(sources)},
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},
        predecessor_sha256={p:sha(ROOT/p) for p in (
            'experiments/SELECTED_MODEL.json', 'experiments/SELECTED_MODEL.md',
            'experiments/EXPERIMENT_LEDGER.csv', 'experiments/WORK_HISTORY.md')},
        candidate_sha256=manifest['candidate_sha256'], added_dependent_score_rows=30,
        tests=26, new_randomized_differential_cases=800,
        retained_randomized_differential_cases=900,
        full_public_prior_matches_retained_at_all_thresholds=True,
        full_public_mean_best_iou_nonregressing=True)
    write_new(ROUND/'promotion_plan.json', promotion_plan)
    freeze = ROUND/'promotion_source_freeze'; freeze.mkdir()
    for name in sorted(sources):
        copy_new(ROOT/name, freeze/Path(name).name)
    copy_new(ROOT/'experiments/SELECTED_MODEL.json', ROUND/'previous_selected_model.json')
    copy_new(ROOT/'experiments/SELECTED_MODEL.md', BACKUP)
    copy_new(LEDGER, ROUND/'ledger_before.csv')
    copy_new(ROOT/'experiments/WORK_HISTORY.md', ROUND/'work_history_before.md')

    PACKAGE.mkdir()
    for name in ('candidate.csv', 'arrival_events.json'):
        copy_new(full/name, PACKAGE/name)
    copy_new(full/'manifest.json', PACKAGE/'full_export_manifest.json')
    for path in evidence:
        copy_new(path, PACKAGE/path.name)
    copy_new(ROUND/'promotion_plan.json', PACKAGE/'promotion_plan.json')
    manifest.update(candidate_path=str((PACKAGE/'candidate.csv').relative_to(ROOT)),
        report_path=str(REPORT.relative_to(ROOT)), source_sha256=promotion_plan['source_sha256'],
        evidence_sha256={p.name:sha(PACKAGE/p.name) for p in evidence},
        affected_tests=26, new_randomized_differential_cases=800,
        all_prior_matches_retained_at_all_five_IoUs=True,
        clipped_fold_comparisons=39, clipped_fold_regressions=0,
        official_OOF_score_checks=30, independent_OOF_replays=30,
        full_public_scores=public_check['scores'],
        changed_public_frames=8, changed_unlabelled_frames=36,
        two_full_exports_byte_exact=True, official_full_validator_passed_twice=True)
    write_new(PACKAGE/'manifest.json', manifest)

    # Check the protected pre-promotion state again immediately before updates.
    verify()
    ledger_result = append_scores(LEDGER, canonical,
        'round83_source_aware_home_arrivals_20261107', 'paired_development_score',
        str(canonical.relative_to(ROOT)))
    assert ledger_result == dict(added=30, rows_after=1798, preceding_bytes_preserved=True)
    assert LEDGER.read_bytes().startswith(original_ledger)
    (ROOT/'experiments/SELECTED_MODEL.md').write_text('''# Selected model: TabPFN with source-aware possession arrivals

The selected recipe is `tabpfn_source_aware_arrivals_seed42`. Its 199319-row CSV is `data/candidate_tabpfn_home_arrivals_v1/candidate.csv`, SHA256 `7c979478291b96b884095cb7a2ad37a877d0790b7855965c48937db5d46c4876`.

Seed42 held-out SpellF1 improves **0.897297/0.854054/0.789744 → 0.918919/0.875676/0.800000** for main/purged/longer validation. All nine seed/protocol primary, macroF1, accuracy, mean best overlap and five per-IoU F1 comparisons are nonregressing. Every prior true-possession match remains at each threshold, and all 39 clipped-fold primary/macro checks pass.

The 33-frame and 87-frame main home overlaps improve **0.634615 → 0.744186** and **0.690000 → 0.711340**. All 17 main true possessions now match at every scored IoU from 0.3 to 0.7; three spurious predictions remain. The rules combine complete-source contact disagreement with posterior refinement restricted to short leading broadcast gaps supported by later owner contacts. Both teams use the same rules, with no truth, protocol or target frame input. D/X and run topology are preserved. No new model fitting or TabPFN queries are needed.

All 26 affected tests pass, including 800 new and 900 retained randomized comparisons. An independent implementation matches all 30 trial OOF outputs and the full candidate. Official scoring verifies all 30 OOF results; both identical full exports pass submission validation. The full file changes 44 frames across five events: eight public and 36 unlabelled. Its labelled full-fit public SpellF1 improves 0.9000 to 0.9125; this is distinct from held-out performance.

The public labels have extensively informed development. Boundaries remain approximate, with weak unchanged target overlaps in some holdouts. Private full-match effects remain unknown; the 36 unlabelled changes are unverified.

See [HOME_BOUNDARY_REFINEMENT_REPORT.md](HOME_BOUNDARY_REFINEMENT_REPORT.md) for evidence, limitations and reproduction. The exact preceding selection document is preserved in [SELECTED_MODEL_BEFORE_HOME_ARRIVALS_20261107.md](SELECTED_MODEL_BEFORE_HOME_ARRIVALS_20261107.md). Every previous candidate remains unchanged. Thirty dependent development score rows append with preceding ledger bytes preserved.
''')
    with (ROOT/'experiments/WORK_HISTORY.md').open('a') as handle:
        handle.write('''

## 7 November: source-aware home possession arrivals selected

The main 33-/87-frame home IoUs improve 0.634615/0.690000 to 0.744186/0.711340. Seed42 SpellF1 main/purged/longer improves 0.897297/0.854054/0.789744 to 0.918919/0.875676/0.800000. All nine paired score/target gates and 39 fold primary/macro checks are nonregressing; all prior matches at all five IoUs remain. Main now matches all 17 true possessions at every scored IoU, with three spurious predictions. Broad incomplete-source proposals failed and remain preserved; a frozen leading-gap guard and 13-frame finalist pass both confirmation seeds. All 30 trial OOF outputs and the full CSV reconstruct independently; official scoring agrees. All 26 affected tests, including 800 new and 900 retained randomized comparisons, pass. Two identical full builds pass official validation. The full CSV changes 44 frames in five events, eight public and 36 unlabelled, with D/X/topology exact; labelled full-fit public SpellF1 improves 0.9000 to 0.9125. No new fits or TabPFN queries. Private effects remain unknown and some holdout boundaries remain weak. Selected `data/candidate_tabpfn_home_arrivals_v1/candidate.csv`, SHA256 `7c979478291b96b884095cb7a2ad37a877d0790b7855965c48937db5d46c4876`. Prior pointer/document/candidates and preceding ledger bytes preserved; 30 dependent score rows append 1768 to 1798. See HOME_BOUNDARY_REFINEMENT_REPORT.md.
''')
    pointer = dict(model=manifest['model'], candidate_path=manifest['candidate_path'],
        candidate_sha256=manifest['candidate_sha256'],
        manifest_path=str((PACKAGE/'manifest.json').relative_to(ROOT)))
    (ROOT/'experiments/SELECTED_MODEL.json').write_text(json.dumps(pointer, indent=2)+'\n')

    assert (ROOT/'experiments/WORK_HISTORY.md').read_bytes().startswith(original_history)
    assert sha(PACKAGE/'candidate.csv') == pointer['candidate_sha256']
    for name, expected in plan['protected_sha256'].items():
        if name not in ('experiments/SELECTED_MODEL.json', 'experiments/SELECTED_MODEL.md',
                        'experiments/EXPERIMENT_LEDGER.csv'):
            assert sha(ROOT/name) == expected, name
    for name, expected in promotion_plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    result = dict(passed=True, selected=pointer, ledger=ledger_result,
        preceding_selection_document_exact=sha(BACKUP)==promotion_plan['predecessor_sha256']['experiments/SELECTED_MODEL.md'],
        previous_candidates_unchanged=True, preceding_history_bytes_preserved=True,
        affected_tests=26, all_nine_score_and_target_checks_pass=True,
        promotion_plan_sha256=sha(ROUND/'promotion_plan.json'),
        manifest_sha256=sha(PACKAGE/'manifest.json'))
    write_new(ROUND/'promotion_result.json', result)
    copy_new(ROUND/'promotion_result.json', PACKAGE/'promotion_result.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    promote()
