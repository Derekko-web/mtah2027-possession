"""Score frozen local vision judgments against public labels; aggregate output only."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from local_ownership import parse_ownership


def main():
    root = ROOT/'data/control_local_ownership_v1'
    inputs = audit_manifest(root/'inputs')
    inference = audit_manifest(root/'inference')
    plan_path = ROOT/'experiments/LOCAL_OWNERSHIP_PLAN.json'
    plan_sha = digest(plan_path)
    if inputs['plan_sha256'] != plan_sha or inference['plan_sha256'] != plan_sha:
        raise ValueError('Local ownership scoring plan differs')
    rows = json.loads((root/'inference/answers.json').read_text())
    samples = json.loads((root/'inputs/samples.json').read_text())
    if sorted(r['center'] for r in rows) != sorted(r['center'] for r in samples):
        raise ValueError('Local inference sample coverage differs')
    for row in rows:
        path = root/'inference'/f'answer_{row["center"]}.json'
        raw = json.loads(path.read_text())
        if digest(path) != row['raw_answer_sha256'] or parse_ownership(raw.get('response', '')) != row['prediction']:
            raise ValueError('Local answer parsing replay differs')
    gt = ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    labels = pm.read_label_csv(gt).label.to_numpy()
    truth = labels[[r['center'] for r in rows]]
    prediction = np.array([r['prediction'] for r in rows])
    if not np.isin(truth, ['H', 'A', 'D']).all():
        raise ValueError('Ownership feasibility samples outside public labels')
    counts = {team:int((truth == team).sum()) for team in ('H', 'A', 'D')}
    recall = {team:float((prediction[truth == team] == team).mean()) if counts[team] else 0.
              for team in ('H', 'A', 'D')}
    balanced = sum(recall.values())/3
    unknown = float((prediction == 'X').mean())
    checks = dict(sample_count=len(rows) >= 15, class_coverage=min(counts.values()) >= 3,
        balanced=balanced >= .75, home=recall['H'] >= .75, away=recall['A'] >= .75,
        stopped=recall['D'] >= .5, abstention=unknown <= .2)
    summary = dict(samples=len(rows), class_counts=counts, recall=recall, balanced_recall=balanced,
        accuracy=float((truth == prediction).mean()), unknown_fraction=unknown,
        predicted_counts={team:int((prediction == team).sum()) for team in ('H', 'A', 'D', 'X')},
        checks=checks, feasible=all(checks.values()), new_fits=0, new_possession_score_rows=0,
        answer_parse_replay_exact=True, plan_sha256=plan_sha, public_label_sha256=digest(gt),
        input_manifest_sha256=digest(root/'inputs/manifest.json'),
        inference_manifest_sha256=digest(root/'inference/manifest.json'),
        source_sha256=digest(Path(__file__)),
        warning='Small adaptive same-match feasibility; not independent/private accuracy or maximum proof')
    (root/'feasibility.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
