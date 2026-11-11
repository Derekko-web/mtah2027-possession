"""Evaluate one frozen sub-grid alignment proposal without media or targets."""
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from broadcast_alignment import frame_usable, map_anchors
from subgrid_alignment import (FIELDS, STRICT, equal, refine_anchors, extension, checkpoints, metadata_gate)
from alignment_coverage import inverse_schedule, query_plan, synthetic_feature_view, coverage_diagnostics
from games import GAME, half_video_blocks

ROUND = ROOT/'data/control_subgrid_alignment_v1'
PLAN = ROOT/'experiments/SUBGRID_ALIGNMENT_PLAN.json'
SOURCES = ('prepare_subgrid_alignment.py', 'subgrid_alignment.py', 'broadcast_alignment.py',
           'alignment_coverage.py', 'broadcast_action_sampling.py', 'broadcast_action_features.py',
           'sports_action_sampling.py', 'sports_action_features.py', 'provided_speed.py', '../games.py')


def validate_inputs():
    plan = json.loads(PLAN.read_text())
    if PLAN.read_bytes() != (ROUND/'protocol_freeze/plan.json').read_bytes():
        raise ValueError('Subgrid frozen plan changed')
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected: raise ValueError('Subgrid frozen input changed: '+name)
    for name, expected in json.loads((ROUND/'protocol_freeze/protected_artifacts.json').read_text()).items():
        if digest(ROOT/name) != expected: raise ValueError('Protected artifact changed')
    frozen = json.loads((ROUND/'protocol_freeze/implementation.json').read_text())
    for name, expected in frozen.items():
        if digest(ROOT/'experiments'/name) != expected: raise ValueError('Frozen execution source changed')
    # Verify inherited replay provenance as well as hashed caches.
    for base in ('control_broadcast_action_v1/prefix_replay', 'control_anchor_failures_v1/assessment'):
        record = json.loads((ROOT/'data'/base/('summary.json' if 'prefix_replay' in base else 'manifest.json')).read_text())
        for name, expected in record['source_sha256'].items():
            if digest(ROOT/'experiments'/name) != expected: raise ValueError('Inherited prefix source changed')
        for name, expected in record['cache_sha256'].items():
            if digest(ROOT/'data'/base/name) != expected: raise ValueError('Inherited prefix cache changed')
    return plan


def observations(name):
    with np.load(ROOT/f'data/control_broadcast_v1/inputs/{name}_observations.npz') as data:
        return {k:data[k] for k in ('positions', 'teams', 'valid')}


def load_npz(path):
    with np.load(path) as data: return dict(data)


def public_summary(summary):
    """Exclude whole-grid rejection counters, which depend on prefix length.

    Every public statistic, association count and checkpoint remains compared.
    Each prefix's full counters are retained and independently reconstructed by
    the auditor from its own frozen schedule, rather than equated across lengths.
    """
    return dict(summary, variants={name:{k:v for k,v in values.items() if k != 'eligibility_counts'}
                                  for name,values in summary['variants'].items()})


def assess(output):
    plan = validate_inputs(); output.mkdir(parents=True)
    hashes = {n:digest(ROOT/'experiments'/n) for n in SOURCES}
    (output/'source_at_dispatch.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    (output/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, requirements_sha256=digest(ROOT/'experiments/requirements-research.txt')), indent=2)+'\n')
    full, broadcast = observations('tactical'), observations('broadcast')
    ub = frame_usable(broadcast)[:164487]
    reserved = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    blocks = half_video_blocks(GAME)
    products, summaries = {}, {}
    for n in (18935, 19060):
        print('Subgrid actual source prefix', n, flush=True)
        tactical = {k:v[:n] if k in ('positions', 'valid') else v for k,v in full.items()}
        old_anchors = load_npz(ROOT/f'data/control_anchor_failures_v1/assessment/anchors_{n}.npz')
        baseline = map_anchors(old_anchors, tactical, broadcast, STRICT)
        cached = load_npz(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz')
        for k in baseline: equal(baseline[k], cached[k], 'original strict '+k)
        anchors, trace, candidate = refine_anchors(tactical, broadcast, old_anchors, blocks, reserved, held)
        mappings = dict(baseline=baseline, candidate=candidate, extension=extension(baseline, candidate))
        for name, values in (('anchors', anchors), ('trace', trace)):
            np.savez_compressed(output/f'{name}_{n}.npz', **values)
        queries, features, diagnostics = {}, {}, {}
        for name, mapping in mappings.items():
            np.savez_compressed(output/f'{name}_mapping_{n}.npz', **mapping)
            rows, counts = inverse_schedule(n, mapping, blocks, reserved, held, ub, 25)
            q = query_plan(rows, mapping, n)
            features[name] = synthetic_feature_view(q, mapping, n, blocks)
            if features[name].shape[1] != 121: raise ValueError('Synthetic feature schema changed')
            queries[name] = q
            (output/f'{name}_schedule_{n}.json').write_text(json.dumps(rows, indent=2)+'\n')
            np.savez_compressed(output/f'{name}_queries_{n}.npz', **q)
            features[name].to_pickle(output/f'{name}_synthetic_features_{n}.pkl')
            diagnostics[name] = coverage_diagnostics(q, queries['baseline'], rows, counts)
            diagnostics[name]['mapped_public_frames'] = int((mapping['frames'][11185:18685] >= 0).sum())
        rule = plan['opportunity_gate']
        if diagnostics['baseline']['observed_public_frames'] != rule['baseline_expected_queries'] or diagnostics['baseline']['mapped_public_frames'] != rule['baseline_expected_mapped']:
            raise ValueError('Round23 baseline not reproduced')
        original_q = load_npz(ROOT/f'data/control_alignment_coverage_v1/assessment/inverse25_{n}_queries.npz')
        for k in original_q: equal(queries['baseline'][k], original_q[k], 'retained inverse25 '+k)
        table, quality = checkpoints(mappings, tactical, broadcast, blocks, reserved, held)
        table.to_csv(output/f'checkpoints_{n}.csv', index=False)
        mask = (anchors['tactical_frames'] >= 11185) & (anchors['tactical_frames'] < 18685)
        summary = dict(variants=diagnostics, checkpoints=quality,
            public_anchors=int(mask.sum()), safe_public_anchors=int(trace['safe'][mask].sum()),
            candidate_anchor_first_failure={str(k):int(v) for k,v in zip(*np.unique(trace['reason'][mask], return_counts=True))},
            boundary_censored_refinements=int(trace['boundary_censored'][mask].sum()),
            candidate_changes_on_old_overlap=int(((candidate['frames'][11185:18685] >= 0) &
                (baseline['frames'][11185:18685] >= 0) & ((candidate['frames'][11185:18685] != baseline['frames'][11185:18685]) |
                (candidate['orientation'][11185:18685] != baseline['orientation'][11185:18685]))).sum()))
        (output/f'summary_{n}.json').write_text(json.dumps(summary, indent=2)+'\n')
        products[n] = (anchors, trace, mappings, queries, features, table)
        summaries[n] = summary
    a, b = products[18935], products[19060]
    masks = [(p[0]['tactical_frames'] >= 11135) & (p[0]['tactical_frames'] < 18735) for p in (a,b)]
    for index in (0, 1):
        for k in a[index]: equal(a[index][k][masks[0]], b[index][k][masks[1]], 'prefix anchor/trace '+k)
    for index in (2, 3):
        for name in a[index]:
            for k in a[index][name]:
                if k == 'anchor_allowed':
                    equal(a[index][name][k][masks[0]], b[index][name][k][masks[1]], 'prefix candidate mask')
                else: equal(a[index][name][k][11135:18735], b[index][name][k][11135:18735], 'prefix mapping/query '+k)
    for name in a[4]:
        if not a[4][name].iloc[11185:18685].equals(b[4][name].iloc[11185:18685]):
            raise ValueError('Full121 synthetic feature prefix parity failed')
    if not a[5].equals(b[5]) or public_summary(summaries[18935]) != public_summary(summaries[19060]):
        raise ValueError('Checkpoint or aggregate prefix parity failed')
    s = summaries[19060]
    # The replay flag is provisional until the independent auditor establishes it.
    gate = metadata_gate(s['variants']['extension'], s['checkpoints'], plan['opportunity_gate'], plan['checkpoints'], True)
    result = dict(round=25, plan_sha256=digest(PLAN), diagnostics=s, provisional_opportunity_gate=gate,
        independent_actual_prefixes=[18935,19060], original_strict_replay_all_fields=True,
        safe_original_global_coarse_replay_exact=True, candidate_trace_mapping_query_prefix_parity=True,
        full121_synthetic_feature_prefix_parity=True, prefix_context_radius=50,
        all_old_per_frame_mapping_fields_exact=True, extension_has_no_composite_anchor_mask=True,
        tactical_dependency_radius_bound=250, image_content_radius_bound=200,
        global_broadcast_search=True, actual_action_feature_parity_established=False,
        selected_unchanged=True, new_images=0, new_neural_calls=0, new_action_outputs_consumed=0,
        new_possession_targets_read=0, new_fits=0, new_score_rows=0)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    validate_inputs()
    if hashes != {n:digest(ROOT/'experiments'/n) for n in SOURCES}: raise ValueError('Execution source changed')
    manifest = dict(plan_sha256=digest(PLAN), source_sha256=hashes, input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROUND/'assessment')
    assess(parser.parse_args().output.resolve())
