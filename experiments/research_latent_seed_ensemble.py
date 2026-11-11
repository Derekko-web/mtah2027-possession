"""Prospective three-seed latent pool; members are never individually selected."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_seed_ensemble import (
    ROOT, ROUND, PLAN, PROTOCOLS, HISTORICAL_BEST_BUFFERED,
    member_seeds, pool_probabilities, check_inputs,
)
from player_representation import (
    ROUND as REPRESENTATION, COLUMNS, audit_cache, digest, interpolate, write_json,
)
from research_representation_ensemble import training_record
from research_player_representation import decode as decode_latent
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


def member_probability(member_seed, protocol, folds, inputs, labels, public,
                       eligible, original, folder, previous, replay):
    probability = np.full((len(inputs), 3), np.nan)
    fits = 0
    folder.mkdir()
    if member_seed == 42 and not replay:
        prior = REPRESENTATION / "screen" / protocol
    else:
        prior = previous
    for block, train, validation in folds:
        record = training_record(labels, train, validation, public)
        masked = np.full_like(labels, "X")
        masked[train] = labels[train]
        other = training_record(masked, train, validation, public)
        for key, value in record.items():
            np.testing.assert_array_equal(value, other[key])
        name = f"pretrained_latent_seed{member_seed}_block{block}.ubj"
        fitting = name + ".fitting.npz"
        np.savez_compressed(folder / fitting, **record)
        if replay or member_seed == 42:
            with np.load(prior / fitting) as saved:
                for key, value in record.items():
                    np.testing.assert_array_equal(saved[key], value)
            model = XGBClassifier()
            model.load_model(prior / name)
            shutil.copy2(prior / name, folder / name)
        else:
            print(f"Fit member{member_seed} {protocol} block{block}", flush=True)
            model = fit_models("ordered_xgboost_frame", inputs, masked, train, member_seed)[0]
            model.save_model(folder / name)
            fits += 1
        value = predict_models([model], inputs.iloc[validation])
        loaded = XGBClassifier()
        loaded.load_model(folder / name)
        np.testing.assert_array_equal(value, predict_models([loaded], inputs.iloc[validation]))
        probability[validation] = value
    probability[public[~eligible[public]]] = original[public[~eligible[public]]]
    np.save(folder / "probability.npy", probability[public])
    if replay:
        np.testing.assert_array_equal(probability[public], np.load(previous / "probability.npy"))
    elif member_seed == 42:
        np.testing.assert_array_equal(probability[public],
                                      np.load(prior / "pretrained_latent_probability.npy"))
    return probability[public], fits


def run(seed=42, replay=False, override=None):
    plan = check_inputs()
    for name in ("inputs", "encodings", "screen"):
        audit_cache(REPRESENTATION / name)
    seeds = member_seeds(seed)
    stage = "screen" if seed == 42 else f"confirmation_seed{seed}"
    saved = ROUND / stage
    if seed != 42 and not json.loads((ROUND / "screen/selection.json").read_text())["passed"]:
        raise ValueError("No eligible fixed seed42 pool")
    if replay:
        audit_cache(saved)
    output = override or ROUND / (stage + "_replay" if replay else stage)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    source = ROOT / "data/advanced_validation"
    raw, ordered = [pd.read_pickle(source / (name + ".pkl")) for name in ("raw", "ordered")]
    n = len(raw)
    enc = REPRESENTATION / "encodings"
    z, eligible = interpolate(np.load(enc / "frames.npy"), np.load(enc / "pretrained_latent.npy"),
                              np.load(enc / "eligible.npy"), n)
    inputs = ordered.join(pd.DataFrame(z, columns=COLUMNS, index=ordered.index))
    broadcast = pd.read_pickle(ROOT / "data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:n]
    motion = pd.read_pickle(ROOT / "data/control_pair_motion_v1/inputs/motion_info.pkl").iloc[:n]
    splits = json.loads((source / "splits.json").read_text())
    labels = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.arange(11185, 18685)
    np.testing.assert_array_equal(np.flatnonzero(labels != "X"), public)
    rows, details, segments, states, changes = [], [], [], {}, {}
    fits = 0
    hashes = dict(plan["input_sha256"])
    for protocol in PROTOCOLS:
        folder = output / protocol
        folder.mkdir()
        folds = [(s["block"], np.asarray(s["train"]), np.asarray(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]), int(v[-1]) + 1) for _, _, v in folds]
        arrays, identities = selected_arrays(protocol, seed, n)
        hashes.update(identities)
        reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
        expected = pd.read_csv(ROOT / "data/control_pair_motion_v1/dead_trim_confirmation" /
                               protocol / "oof_predictions.csv")[f"motion_dead_trim_seed{seed}"]
        np.testing.assert_array_equal(reference[public], expected)
        members = []
        for member in seeds:
            name = f"member{member}"
            value, count = member_probability(member, protocol, folds, inputs, labels, public,
                eligible, arrays[0], folder / name, saved / protocol / name, replay)
            members.append(value)
            fits += count
        pooled = pool_probabilities(members)
        # Preserve the reference byte-for-byte where the representation is absent.
        pooled[~eligible[public]] = arrays[0][public[~eligible[public]]]
        np.save(folder / "pooled_probability.npy", pooled)
        if replay:
            np.testing.assert_array_equal(pooled, np.load(saved / protocol / "pooled_probability.npy"))
        probability = np.full((n, 3), np.nan)
        probability[public] = pooled
        prediction = decode_latent(arrays, probability, broadcast, motion, raw, blocks)
        np.testing.assert_array_equal(prediction == "D", reference == "D")
        np.testing.assert_array_equal(prediction == "X", reference == "X")
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public]))
        current, method = f"current_seed{seed}", f"latent_seed_pool_seed{seed}"
        for name, value in ((current, reference), (method, prediction)):
            table[name] = value[public]
            record_predictions(output, protocol, name, public, labels, folds,
                               value[public], rows, details, segments)
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
    selection = assess(scores, diagnostics, official, states, changes, seed)
    for pair in selection["pairs"]:
        if seed == 42 and pair["protocol"] == "buffered_minutes":
            value = scores[(scores.protocol == pair["protocol"]) &
                           (scores.model == pair["model"])].spellF1.item()
            pair["checks"]["strict_historical_best_gain"] = bool(value > HISTORICAL_BEST_BUFFERED + 1e-12)
    method = f"latent_seed_pool_seed{seed}"
    passing = len(selection["pairs"]) == 3 and all(
        all(pair["checks"].values()) for pair in selection["pairs"])
    selection.update(passed=passing, eligible=[method] if passing else [],
                     selected=method if passing else None, member_seeds=list(seeds),
                     historical_substantial_and_extra_short_required=False)
    write_json(output / "selection.json", selection)
    if replay:
        for name in ("scores.csv", "fold_scores.csv", "segment_diagnostics.csv",
                     "official_comparison.csv", "selection.json"):
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError("Score/gate replay differs: " + name)
        for protocol in PROTOCOLS:
            name = Path(protocol) / "oof_predictions.csv"
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError("Prediction replay differs")
    write_json(output / "manifest.json", dict(
        plan_sha256=digest(PLAN), main_seed=seed, member_seeds=list(seeds),
        new_models=fits, checkpoint_reloads=39, reused_R37_models=13 if seed == 42 else 0,
        manual_labels_used=False, individual_members_scored_or_selected=False,
        targets_only_explicit_public_training=True, identical_reference_D_X=True,
        input_sha256=hashes,
        source_sha256=json.loads((ROUND / "protocol_freeze/implementation.json").read_text()),
        cache_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}))
    check_inputs()
    print(scores[["protocol", "model", "spellF1", "macroF1", "short_live_matched_iou50"]].to_string(index=False), flush=True)
    print("Eligible:", selection["eligible"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.seed, args.replay, args.output)
