"""Prospective fixed probability ensemble, paired checks and saved-model replay."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation_ensemble import (
    ROOT, ROUND, PLAN, PROTOCOLS, HISTORICAL_BEST_BUFFERED,
    check_inputs, decode_ensemble,
)
from player_representation import (
    ROUND as REPRESENTATION, COLUMNS, audit_cache, digest, interpolate, write_json,
)

sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_candidate import objective_weights
from dfl_signal_audit import matching
from finalize_broadcast_closure import official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from research_temporal_dead import assess
from temporal_candidate import fit_models, predict_models


def ensemble_checks(scores, diagnostics, official, states, changes, seed):
    """Keep existing protections; test the user's current strict-score objective."""
    result = assess(scores, diagnostics, official, states, changes, seed)
    for pair in result["pairs"]:
        if pair["protocol"] == "buffered_minutes":
            pair["checks"]["strict_current_primary_gain"] = bool(pair["spellF1_delta"] > 1e-12)
            if seed == 42:
                value = scores[(scores.protocol == pair["protocol"])
                               & (scores.model == pair["model"])].spellF1.item()
                pair["checks"]["strict_historical_best_gain"] = bool(
                    value > HISTORICAL_BEST_BUFFERED + 1e-12)
    method = f"equal_pool_seed{seed}"
    eligible = [method] if all(all(p["checks"].values()) for p in result["pairs"]) else []
    result.update(passed=bool(eligible), eligible=eligible,
                  selected=method if eligible else None,
                  historical_best_buffered=HISTORICAL_BEST_BUFFERED,
                  historical_substantial_and_extra_short_required=False)
    return result


def training_record(labels, train, validation, public):
    """Construct augmentation targets using only explicitly allowed train rows."""
    if (len(np.intersect1d(train, validation)) or not np.isin(train, public).all()
            or not np.isin(validation, public).all()):
        raise ValueError("Invalid public exclusion split")
    masked = np.full_like(labels, "X")
    masked[train] = labels[train]
    target = np.asarray([{"H": 0, "A": 1, "D": 2}[v] for v in labels[train]])
    target2 = np.asarray([{"H": 0, "A": 1, "D": 2}[v] for v in masked[train]])
    y, weight = objective_weights(target, train, False)
    y2, weight2 = objective_weights(target2, train, False)
    np.testing.assert_array_equal(y, y2)
    np.testing.assert_array_equal(weight, weight2)
    return dict(train=train, validation=validation, target=y, weight=weight)


def latent_probability(seed, replay, saved, folder, protocol, folds,
                       inputs, labels, public, eligible, original):
    """Seed42 reuses exact R37 fits; new seeds use the identical frozen learner."""
    probability = np.full((len(inputs), 3), np.nan)
    new_models = 0
    for block, train, validation in folds:
        filename = f"pretrained_latent_seed{seed}_block{block}.ubj"
        path = folder / filename
        record = training_record(labels, train, validation, public)
        fitting_name = filename + ".fitting.npz"
        prior_folder = saved / protocol if replay else REPRESENTATION / "screen" / protocol
        if replay or seed == 42:
            with np.load(prior_folder / fitting_name) as prior:
                for name, value in record.items():
                    np.testing.assert_array_equal(prior[name], value)
            model = XGBClassifier()
            model.load_model(prior_folder / filename)
            shutil.copy2(prior_folder / filename, path)
        else:
            print(f"Fit pretrained latent seed{seed} {protocol} block{block}", flush=True)
            model = fit_models("ordered_xgboost_frame", inputs, labels, train, seed)[0]
            model.save_model(path)
            new_models += 1
        np.savez_compressed(folder / fitting_name, **record)
        value = predict_models([model], inputs.iloc[validation])
        loaded = XGBClassifier()
        loaded.load_model(path)
        np.testing.assert_array_equal(value, predict_models([loaded], inputs.iloc[validation]))
        probability[validation] = value
    probability[public[~eligible[public]]] = original[public[~eligible[public]]]
    np.save(folder / "pretrained_latent_probability.npy", probability[public])
    if replay or seed == 42:
        expected = np.load(prior_folder / "pretrained_latent_probability.npy")
        np.testing.assert_array_equal(probability[public], expected)
    return probability, new_models


def run(seed=42, replay=False, override=None):
    plan = check_inputs()
    audit_cache(REPRESENTATION / "inputs")
    audit_cache(REPRESENTATION / "encodings")
    audit_cache(REPRESENTATION / "screen")
    stage = "screen" if seed == 42 else f"confirmation_seed{seed}"
    saved = ROUND / stage
    if seed != 42 and not json.loads((ROUND / "screen/selection.json").read_text())["passed"]:
        raise ValueError("No eligible fixed seed42 ensemble")
    if replay:
        audit_cache(saved)
    output = override or ROUND / (stage + "_replay" if replay else stage)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    source = ROOT / "data/advanced_validation"
    raw = pd.read_pickle(source / "raw.pkl")
    ordered = pd.read_pickle(source / "ordered.pkl")
    n = len(raw)
    encoding = REPRESENTATION / "encodings"
    z, eligible = interpolate(np.load(encoding / "frames.npy"),
                              np.load(encoding / "pretrained_latent.npy"),
                              np.load(encoding / "eligible.npy"), n)
    inputs = ordered.join(pd.DataFrame(z, columns=COLUMNS, index=ordered.index))
    broadcast = pd.read_pickle(ROOT / "data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:n]
    motion = pd.read_pickle(ROOT / "data/control_pair_motion_v1/inputs/motion_info.pkl").iloc[:n]
    splits = json.loads((source / "splits.json").read_text())
    labels = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.arange(11185, 18685)
    np.testing.assert_array_equal(np.flatnonzero(labels != "X"), public)
    rows, details, segments, states, changes = [], [], [], {}, {}
    models = 0
    input_hashes = dict(plan["input_sha256"])
    for protocol in PROTOCOLS:
        folder = output / protocol
        folder.mkdir()
        folds = [(s["block"], np.asarray(s["train"]), np.asarray(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]), int(v[-1]) + 1) for _, _, v in folds]
        arrays, hashes = selected_arrays(protocol, seed, n)
        input_hashes.update(hashes)
        reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
        baseline = ROOT / "data/control_pair_motion_v1/dead_trim_confirmation" / protocol / "oof_predictions.csv"
        np.testing.assert_array_equal(reference[public], pd.read_csv(baseline)[f"motion_dead_trim_seed{seed}"])
        np.testing.assert_array_equal(
            decode_ensemble(arrays, arrays[0], broadcast, motion, raw, blocks), reference)
        probability, fitted = latent_probability(
            seed, replay, saved, folder, protocol, folds, inputs, labels,
            public, eligible, arrays[0])
        models += fitted
        prediction = decode_ensemble(arrays, probability, broadcast, motion, raw, blocks)
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public]))
        current, method = f"current_seed{seed}", f"equal_pool_seed{seed}"
        for name, values in ((current, reference), (method, prediction)):
            table[name] = values[public]
            record_predictions(output, protocol, name, public, labels, folds,
                               values[public], rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
        states[(protocol, method)] = (matching(labels[public], reference[public]),
                                      matching(labels[public], prediction[public]))
        changes[(protocol, method)] = int((prediction[public] != reference[public]).sum())
        table.to_csv(folder / "oof_predictions.csv", index=False)
    add_short_metrics(rows, segments)
    scores, diagnostics = pd.DataFrame(rows), pd.DataFrame(segments)
    scores.to_csv(output / "scores.csv", index=False)
    diagnostics.to_csv(output / "segment_diagnostics.csv", index=False)
    pd.DataFrame(details).to_csv(output / "fold_scores.csv", index=False)
    official = official_scores(output, scores)
    selection = ensemble_checks(scores, diagnostics, official, states, changes, seed)
    write_json(output / "selection.json", selection)
    if replay:
        for name in ("scores.csv", "segment_diagnostics.csv", "fold_scores.csv",
                     "official_comparison.csv", "selection.json"):
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError("Score/gate replay differs: " + name)
        for protocol in PROTOCOLS:
            name = Path(protocol) / "oof_predictions.csv"
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError("Shared inference replay differs: " + str(name))
    write_json(output / "manifest.json", dict(
        plan_sha256=digest(PLAN), new_models=models,
        reused_seed42_models=13 if seed == 42 else 0,
        manual_labels_used=False, only_public_train_indices=True,
        equal_weight=.5, identical_selected_D_and_X=True,
        unchanged_selected_identity_exact=True, checkpoint_reload_exact=True,
        input_sha256=input_hashes,
        source_sha256=json.loads((ROUND / "protocol_freeze/implementation.json").read_text()),
        cache_sha256={str(p.relative_to(output)): digest(p)
                      for p in sorted(output.rglob("*")) if p.is_file()},
    ))
    check_inputs()
    print(scores[["protocol", "model", "spellF1", "macroF1",
                  "short_live_matched_iou50", "false_dead_frames", "dead_recall"]].to_string(index=False), flush=True)
    print("Eligible:", selection["eligible"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.seed, args.replay, args.output)
