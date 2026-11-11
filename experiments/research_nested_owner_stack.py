"""Seed42 nested ownership stacking; no fitting runs before the protocol freeze."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_owner_stack import (
    ROOT, ROUND, PLAN, PROTOCOLS, PUBLIC, HISTORICAL_BEST_BUFFERED,
    inner_splits, meta_record, fit_meta, predict_meta, check_inputs, decode_owner,
)
from player_representation import (
    ROUND as REPRESENTATION, COLUMNS, audit_cache, digest, interpolate, write_json,
)
from research_representation_ensemble import latent_probability, training_record
from temporal_candidate import fit_models, predict_models
from evaluation import possession_metrics as pm
from research_ownership_graph import selected_arrays
from motion_dead_trim import decode_motion_dead_trim
from research_stoppages import record_predictions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_temporal_dead import assess
from dfl_signal_audit import matching
from finalize_broadcast_closure import official_scores


def save_record(path, record, previous=None):
    np.savez_compressed(path, **record)
    if previous is not None:
        with np.load(previous) as old:
            if set(old.files) != set(record):
                raise ValueError("Saved fitting record schema differs")
            for key, value in record.items():
                np.testing.assert_array_equal(old[key], value)


def nested_prediction(inputs, labels, outer_train, outer_validation, original,
                      representation, eligible, folder, saved, replay):
    """Every meta training prediction excludes all outer validation labels."""
    masked = np.full_like(labels, "X")
    masked[outer_train] = labels[outer_train]
    inner_probability = [np.full((len(labels), 3), np.nan) for _ in range(2)]
    covered = np.zeros(len(labels), dtype=bool)
    records, fits, reloads = [], 0, 0
    for number, train, validation in inner_splits(outer_train, outer_validation):
        records.append(dict(inner=number, train=train.tolist(), validation=validation.tolist(),
                            skipped_empty_validation=not len(validation)))
        if not len(validation):
            continue
        if set(masked[train]) != {"H", "A", "D"}:
            raise ValueError("Inner training lacks an expert target class")
        if (covered[validation].any() or np.intersect1d(train, outer_validation).size
                or np.intersect1d(validation, outer_validation).size):
            raise ValueError("Nested exclusion/coverage failed")
        covered[validation] = True
        record = training_record(masked, train, validation, PUBLIC)
        for index, view in enumerate(("original", "representation")):
            name = f"inner{number}_{view}"
            path, old = folder / (name + ".ubj"), saved / (name + ".ubj")
            save_record(folder / (name + ".fitting.npz"), record,
                        saved / (name + ".fitting.npz") if replay else None)
            if replay:
                model = XGBClassifier()
                model.load_model(old)
                shutil.copy2(old, path)
            else:
                print("Fit nested", folder.name, name, flush=True)
                model = fit_models("ordered_xgboost_frame", inputs[index], masked, train, 42)[0]
                model.save_model(path)
                fits += 1
            value = predict_models([model], inputs[index].iloc[validation])
            loaded = XGBClassifier()
            loaded.load_model(path)
            np.testing.assert_array_equal(value, predict_models([loaded], inputs[index].iloc[validation]))
            reloads += 1
            if index == 1:
                value[~eligible[validation]] = inner_probability[0][validation[~eligible[validation]]]
            inner_probability[index][validation] = value
    np.testing.assert_array_equal(np.flatnonzero(covered), np.sort(outer_train))
    for index, name in enumerate(("original", "representation")):
        value = inner_probability[index][outer_train]
        np.save(folder / (name + "_inner_probability.npy"), value)
        if replay:
            np.testing.assert_array_equal(value, np.load(saved / (name + "_inner_probability.npy")))
    record = meta_record(masked, outer_train, *[p[outer_train] for p in inner_probability])
    # Labels outside the outer pool cannot alter the record or any inner expert.
    other = meta_record(labels, outer_train, *[p[outer_train] for p in inner_probability])
    for key, value in record.items():
        np.testing.assert_array_equal(value, other[key])
    save_record(folder / "meta.fitting.npz", record, saved / "meta.fitting.npz" if replay else None)
    if replay:
        model = joblib.load(saved / "meta.joblib")
        shutil.copy2(saved / "meta.joblib", folder / "meta.joblib")
    else:
        model = fit_meta(record)
        joblib.dump(model, folder / "meta.joblib")
    owner = predict_meta(model, original[outer_validation], representation[outer_validation])
    loaded = joblib.load(folder / "meta.joblib")
    np.testing.assert_array_equal(owner, predict_meta(loaded, original[outer_validation], representation[outer_validation]))
    np.save(folder / "owner_probability.npy", owner)
    if replay:
        np.testing.assert_array_equal(owner, np.load(saved / "owner_probability.npy"))
    write_json(folder / "nested_splits.json", records)
    write_json(folder / "meta_parameters.json", dict(
        C=1., fit_intercept=False, solver="lbfgs", max_iter=2000, scaler=None,
        coefficients=model.coef_.tolist(), intercept=model.intercept_.tolist(),
        classes=model.classes_.tolist(), iterations=model.n_iter_.tolist(),
        inner_expert_fits=fits, expert_reloads=reloads,
        outer_validation_labels_used=False))
    return owner, fits, reloads


def run(replay=False, override=None):
    plan = check_inputs()
    for name in ("inputs", "encodings", "screen"):
        audit_cache(REPRESENTATION / name)
    saved = ROUND / "screen"
    if replay:
        audit_cache(saved)
    output = override or ROUND / ("screen_replay" if replay else "screen")
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    source = ROOT / "data/advanced_validation"
    raw, ordered = [pd.read_pickle(source / (name + ".pkl")) for name in ("raw", "ordered")]
    n = len(raw)
    enc = REPRESENTATION / "encodings"
    latent, eligible = interpolate(np.load(enc / "frames.npy"), np.load(enc / "pretrained_latent.npy"),
                                   np.load(enc / "eligible.npy"), n)
    inputs = (ordered, ordered.join(pd.DataFrame(latent, columns=COLUMNS, index=ordered.index)))
    broadcast = pd.read_pickle(ROOT / "data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:n]
    motion = pd.read_pickle(ROOT / "data/control_pair_motion_v1/inputs/motion_info.pkl").iloc[:n]
    labels = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels != "X"), PUBLIC)
    splits = json.loads((source / "splits.json").read_text())
    rows, details, segments, states, changes = [], [], [], {}, {}
    fits, reloads = 0, 0
    input_hashes = dict(plan["input_sha256"])
    for protocol in PROTOCOLS:
        folder = output / protocol
        folder.mkdir()
        folds = [(s["block"], np.asarray(s["train"]), np.asarray(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]), int(v[-1]) + 1) for _, _, v in folds]
        arrays, hashes = selected_arrays(protocol, 42, n)
        input_hashes.update(hashes)
        reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
        expected = pd.read_csv(ROOT / "data/control_pair_motion_v1/dead_trim_confirmation" /
                               protocol / "oof_predictions.csv").motion_dead_trim_seed42
        np.testing.assert_array_equal(reference[PUBLIC], expected)
        representation, count = latent_probability(42, replay, saved, folder, protocol, folds,
            inputs[1], labels, PUBLIC, eligible, arrays[0])
        if count:
            raise AssertionError("Seed42 outer models must be immutable reused fits")
        owner = np.full((n, 2), np.nan)
        for block, train, validation in folds:
            destination, previous = folder / f"outer{block}", saved / protocol / f"outer{block}"
            destination.mkdir()
            value, count, checked = nested_prediction(inputs, labels, train, validation,
                arrays[0], representation, eligible, destination, previous, replay)
            owner[validation] = value
            fits += count
            reloads += checked
        np.save(folder / "owner_probability.npy", owner[PUBLIC])
        if replay:
            np.testing.assert_array_equal(owner[PUBLIC], np.load(saved / protocol / "owner_probability.npy"))
        prediction = decode_owner(arrays, owner, eligible, broadcast, motion, raw, blocks)
        table = pd.DataFrame(dict(frame=PUBLIC, ground_truth=labels[PUBLIC]))
        current, method = "current_seed42", "nested_owner_seed42"
        for name, value in ((current, reference), (method, prediction)):
            table[name] = value[PUBLIC]
            record_predictions(output, protocol, name, PUBLIC, labels, folds,
                               value[PUBLIC], rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
        states[(protocol, method)] = (matching(labels[PUBLIC], reference[PUBLIC]),
                                      matching(labels[PUBLIC], prediction[PUBLIC]))
        changes[(protocol, method)] = int((prediction[PUBLIC] != reference[PUBLIC]).sum())
        table.to_csv(folder / "oof_predictions.csv", index=False)
    add_short_metrics(rows, segments)
    scores, diagnostics = pd.DataFrame(rows), pd.DataFrame(segments)
    scores.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(details).to_csv(output / "fold_scores.csv", index=False)
    diagnostics.to_csv(output / "segment_diagnostics.csv", index=False)
    official = official_scores(output, scores)
    selection = assess(scores, diagnostics, official, states, changes, 42)
    for pair in selection["pairs"]:
        if pair["protocol"] == "buffered_minutes":
            value = scores[(scores.protocol == pair["protocol"]) & (scores.model == pair["model"])].spellF1.item()
            pair["checks"]["strict_historical_best_gain"] = bool(value > HISTORICAL_BEST_BUFFERED + 1e-12)
    method = "nested_owner_seed42"
    pairs = [p for p in selection["pairs"] if p["model"] == method]
    passed = len(pairs) == 3 and all(all(p["checks"].values()) for p in pairs)
    selection.update(passed=passed, eligible=[method] if passed else [], selected=method if passed else None,
                     historical_substantial_and_extra_short_required=False)
    write_json(output / "selection.json", selection)
    if replay:
        for name in ("scores.csv", "fold_scores.csv", "segment_diagnostics.csv", "official_comparison.csv", "selection.json"):
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError("Score/gate replay differs: " + name)
        for protocol in PROTOCOLS:
            name = Path(protocol) / "oof_predictions.csv"
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError("Prediction replay differs")
    write_json(output / "manifest.json", dict(
        plan_sha256=digest(PLAN), seed=42, new_inner_expert_models=fits,
        inner_expert_reloads=reloads, meta_models=13, outer_R37_models_reused=13,
        manual_labels_used=False, outer_validation_labels_used=False, scaler=None,
        input_sha256=input_hashes,
        source_sha256=json.loads((ROUND / "protocol_freeze/implementation.json").read_text()),
        cache_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}))
    check_inputs()
    print(scores[["protocol", "model", "spellF1", "macroF1", "short_live_matched_iou50"]].to_string(index=False), flush=True)
    print("Eligible:", selection["eligible"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.replay, args.output)
