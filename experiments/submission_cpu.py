"""Deterministic CPU stages for the frozen submission build."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks, source_frames


PUBLIC_GT = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
HALVES = half_video_blocks(GAME)
BLOCKS = [(start, end + 1) for _, start, end in HALVES]


def tactical(work, output):
    from broadcast_alignment import load_source
    from temporal_candidate import build_inputs

    raw, base, ordered = build_inputs(source_frames())
    for name, table in (('raw', raw), ('base', base), ('ordered', ordered)):
        table.to_pickle(output / f'{name}.pkl')
    tracked = load_source('tactical')
    positions = tracked['positions']
    positions[~tracked['valid']] = np.nan
    np.savez_compressed(output / 'encoder_input.npz', positions=positions, teams=tracked['teams'])


def reserved_windows():
    from freeze_annotation_windows import SEED, sample_windows
    from starter.load_smart_tagging import load as load_tags

    return sample_windows(load_tags(), HALVES, np.random.default_rng(SEED))


def held_frames(tactical_source, reserved):
    from ball_appearance import weak_eligibility, weak_split
    from starter.load_homography import load

    _, error, valid = load()
    eligible = weak_eligibility(tactical_source, error, valid)
    train, test, _, _ = weak_split(eligible, HALVES, reserved)
    # The second schedule uses the same exclusions plus the first schedule's
    # buffer. Both schedules are derived from geometry and automatic tags.
    frames = np.arange(len(eligible))
    inside = np.zeros(len(eligible), dtype=bool)
    for _, start, end in HALVES:
        inside[start:end + 1] = True
    eligible &= inside & (frames % 25 == 0)
    eligible[10935:18935] = False
    for window in reserved:
        eligible[max(0, window['start'] - 250):window['end'] + 251] = False
    for frame in np.r_[train, test]:
        eligible[max(0, frame - 250):frame + 251] = False
    return np.sort(np.random.default_rng(43).choice(np.flatnonzero(eligible), 60, replace=False))


def cameras(work, output):
    from broadcast_alignment import formation_anchors, load_source, map_anchors, reconcile_teams
    from broadcast_features import aligned_features
    from pair_motion import pair_feature_views
    from subgrid_alignment import STRICT, extension, refine_anchors

    tactical_source, broadcast = load_source('tactical'), load_source('broadcast')
    broadcast['teams'] = reconcile_teams(tactical_source['names'], broadcast['names'], broadcast['teams'])
    print('Aligning the original broadcast camera', flush=True)
    original = formation_anchors(tactical_source, broadcast, {'strict': STRICT},
                                 lambda i, n: print('Formation anchors', i, '/', n, flush=True))
    mapping = map_anchors(original, tactical_source, broadcast, STRICT)
    _, combined = aligned_features(tactical_source, broadcast, mapping)
    combined.to_pickle(output / 'broadcast.pkl')
    old_mapped = mapping['frames'] >= 0
    np.save(output / 'old_mapped.npy', old_mapped)

    print('Refining unmapped camera intervals', flush=True)
    reserved = reserved_windows()
    held = held_frames(tactical_source, reserved)
    _, _, refined = refine_anchors(tactical_source, broadcast, original, HALVES, reserved, held)
    _, extended = aligned_features(tactical_source, broadcast, extension(mapping, refined))
    extended.to_pickle(output / 'extended.pkl')

    print('Computing motion features and tactical contact kernels', flush=True)
    views, motion = pair_feature_views(tactical_source, broadcast, mapping, combined)
    views['broadcast_motion'].to_pickle(output / 'motion_features.pkl')
    motion.to_pickle(output / 'motion.pkl')


def partitions():
    for start, end in BLOCKS:
        for left in range(start, end, 4096):
            yield left, min(left + 4096, end)


def queries(work, output):
    from player_representation import COLUMNS
    from rich_features import swap_teams
    from sklearn.impute import SimpleImputer
    from tabpfn_full_features import full_context_frames

    ordered = pd.read_pickle(work / 'tactical/ordered.pkl')
    latent = np.load(work / 'encoder/full_latent.npy', allow_pickle=False)
    inputs = ordered.join(pd.DataFrame(latent, columns=COLUMNS, index=ordered.index))
    truth = pm.read_label_csv(PUBLIC_GT).label.to_numpy()
    centers, _ = full_context_frames(truth)
    live = inputs.iloc[centers]
    paired = pd.concat([live, swap_teams(live)], ignore_index=True)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=False)
    context = imputer.fit_transform(paired).astype(np.float32)
    target = (truth[centers] == 'A').astype(np.int64)
    np.save(output / 'context.npy', context)
    np.save(output / 'target.npy', np.r_[target, 1 - target])
    for start, end in partitions():
        batch = inputs.iloc[start:end]
        np.save(output / f'{start}-original.npy', imputer.transform(batch).astype(np.float32))
        np.save(output / f'{start}-swapped.npy', imputer.transform(swap_teams(batch)).astype(np.float32))


def probability(predict, features, width):
    """Retain the training builders' original 20,000-row inference batches."""
    result = np.empty((len(features), width))
    for start in range(0, len(features), 20000):
        end = min(start + 20000, len(features))
        result[start:end] = predict(features.iloc[start:end])
    return result


def fit_heads(work, output):
    from control_refinement import fast_columns, fit_control, predict_control
    from research_control_renewal import fit_renewal
    from stoppage_context import context_features, fit_context, load_ball, weak_samples
    from temporal_candidate import fit_models, predict_models

    raw, base, ordered = [pd.read_pickle(work / 'tactical' / f'{name}.pkl')
                          for name in ('raw', 'base', 'ordered')]
    labels = pm.read_label_csv(PUBLIC_GT).label.to_numpy()
    train = np.flatnonzero(labels != 'X')
    for name, method, features in (('current', 'ordered_xgboost_frame', ordered),
                                   ('previous', 'previous_ensemble', base)):
        models = fit_models(method, features, labels, train, 42)
        np.save(output / f'{name}.npy', probability(lambda batch: predict_models(models, batch), features, 3))

    fast_inputs = base[fast_columns(base)]
    fast = fit_control(fast_inputs, labels, train, 42)
    np.save(output / 'fast.npy', probability(lambda batch: predict_control(fast, batch), fast_inputs, 2))
    physical, restart = context_features(raw, base, load_ball(len(raw)))
    frames, target = weak_samples(raw, physical, restart, reserved_windows())
    dead = []
    for features in (physical, restart):
        model = fit_context(features, labels, train, frames, target, .5, 42)
        dead.append(probability(model.predict_proba, features, 2)[:, 1])
    np.save(output / 'heads.npy', np.column_stack(dead))

    for name, file in (('broadcast', 'broadcast'), ('motion', 'motion_features')):
        features = ordered.join(pd.read_pickle(work / 'cameras' / f'{file}.pkl'))
        model = fit_renewal(features, labels, train, 'frame', 42)
        np.save(output / f'{name}.npy', probability(lambda batch: predict_models([model], batch), features, 3))


def export(work, output):
    from bounded_holdout_repair import refine as repair_holdouts
    from broadcast_short_envelope import refine as trim_broadcast
    from leading_gap_arrivals import refine as refine_arrivals
    from new_camera_control import recover as recover_new_camera
    from probability_guarded_boundaries import guarded_refine
    from short_control_recovery import recover as recover_short_control
    from supported_gap_stoppage import refine as suppress_stoppage_fragments
    from tabpfn_short_control_inference import decode_tabpfn_short_control

    n = source_frames()
    raw = pd.read_pickle(work / 'tactical/raw.pkl')
    broadcast, extended, motion = [pd.read_pickle(work / 'cameras' / f'{name}.pkl')
                                   for name in ('broadcast', 'extended', 'motion')]
    old_mapped = np.load(work / 'cameras/old_mapped.npy', allow_pickle=False)
    eligible = np.load(work / 'encoder/full_eligible.npy', allow_pickle=False)
    arrays = [np.load(work / 'heads' / f'{name}.npy', allow_pickle=False)
              for name in ('current', 'previous', 'fast', 'heads', 'broadcast', 'motion')]
    fast, heads = arrays[2:4]
    owner = np.full((n, 2), np.nan)
    for start, end in partitions():
        value = np.load(work / 'native' / str(start) / 'owner.npy', allow_pickle=False)
        if value.shape != (end - start, 2):
            raise ValueError(f'Incomplete TabPFN partition at frame {start}')
        owner[start:end] = value

    labels, _ = decode_tabpfn_short_control(arrays, owner, eligible, broadcast, motion, raw, BLOCKS)
    labels, _ = trim_broadcast(labels, broadcast, BLOCKS, 13)
    labels, recoveries = recover_new_camera(labels, fast, raw, motion, extended, old_mapped, BLOCKS, 13)
    labels, _ = guarded_refine(labels, fast, owner, raw, extended, old_mapped, BLOCKS, recoveries, approach=25)
    labels, _ = refine_arrivals(labels, owner, raw, extended, BLOCKS, camera=True, posterior_radius=13)
    labels, _ = suppress_stoppage_fragments(labels, heads, raw, extended, BLOCKS)
    labels, _ = repair_holdouts(labels, heads, owner, fast, raw, extended, BLOCKS)
    labels, _ = recover_short_control(labels, owner, fast, eligible, raw, extended, BLOCKS)
    table = pd.DataFrame(dict(frame=np.arange(n), label=labels))
    problem = pm.validate_submission(table, n)
    if problem:
        raise ValueError(problem)
    table.to_csv(output / 'candidate.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = dict(tactical=tactical, cameras=cameras, queries=queries, heads=fit_heads, export=export)
    parser.add_argument('stage', choices=stages)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stages[args.stage](args.work, args.output)
