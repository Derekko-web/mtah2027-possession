"""Fixed live-only control/representation heads with paired validation and replay."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorized_latent import (
    ROOT, ROUND, PLAN, PROTOCOLS, VIEWS, HISTORICAL_BEST_BUFFERED,
    check_inputs, training_record, fit_owner, predict_owner, decode_owner,
)
from player_representation import (
    ROUND as REPRESENTATION, COLUMNS, audit_cache, digest, interpolate, write_json,
)

sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from dfl_signal_audit import matching
from finalize_broadcast_closure import official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from research_temporal_dead import assess


def selection_checks(scores, diagnostics, official, states, changes, seed):
    result = assess(scores, diagnostics, official, states, changes, seed)
    for pair in result["pairs"]:
        if seed == 42 and pair["protocol"] == "buffered_minutes":
            value = scores[(scores.protocol == pair["protocol"])
                           & (scores.model == pair["model"])].spellF1.item()
            pair["checks"]["strict_historical_best_gain"] = bool(value > HISTORICAL_BEST_BUFFERED + 1e-12)
    eligible = [name for name in scores.model.unique()
                if name != f"current_seed{seed}" and
                all(all(pair["checks"].values()) for pair in result["pairs"] if pair["model"] == name)]
    def rank(name):
        pairs = [p for p in result["pairs"] if p["model"] == name]
        return (-next(p["spellF1_delta"] for p in pairs if p["protocol"] == "buffered_minutes"),
                -min(p["spellF1_delta"] for p in pairs), sum(p["changed_frames"] for p in pairs),
                0 if name.startswith("control") else 1)
    eligible.sort(key=rank)
    result.update(passed=bool(eligible), eligible=eligible, selected=eligible[0] if eligible else None,
                  historical_best_buffered=HISTORICAL_BEST_BUFFERED,
                  historical_substantial_and_extra_short_required=False)
    return result


def run(seed=42, replay=False, override=None):
    plan = check_inputs()
    audit_cache(REPRESENTATION / "inputs")
    audit_cache(REPRESENTATION / "encodings")
    views = list(VIEWS)
    if seed != 42:
        selection = json.loads((ROUND / "screen/selection.json").read_text())
        if not selection["passed"]:
            raise ValueError("No eligible seed42 factorized head")
        views = [selection["selected"].rsplit("_seed", 1)[0]]
    stage = "screen" if seed == 42 else f"confirmation_seed{seed}"
    saved = ROUND / stage
    if replay:
        audit_cache(saved)
    output = override or ROUND / (stage + "_replay" if replay else stage)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(PLAN.read_bytes())
    source = ROOT / "data/advanced_validation"
    raw, ordered = [pd.read_pickle(source / (name + ".pkl")) for name in ("raw", "ordered")]
    n = len(raw)
    encoding = REPRESENTATION / "encodings"
    z, eligible = interpolate(np.load(encoding / "frames.npy"),
                              np.load(encoding / "pretrained_latent.npy"),
                              np.load(encoding / "eligible.npy"), n)
    inputs = dict(control_owner=ordered,
                  pretrained_owner=ordered.join(pd.DataFrame(z, columns=COLUMNS, index=ordered.index)))
    support = dict(control_owner=np.ones(n, bool), pretrained_owner=eligible)
    broadcast = pd.read_pickle(ROOT / "data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:n]
    motion = pd.read_pickle(ROOT / "data/control_pair_motion_v1/inputs/motion_info.pkl").iloc[:n]
    splits = json.loads((source / "splits.json").read_text())
    labels = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.arange(11185, 18685)
    np.testing.assert_array_equal(np.flatnonzero(labels != "X"), public)
    rows, details, segments, states, changes = [], [], [], {}, {}
    models, reloads = 0, 0
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
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public]))
        current = f"current_seed{seed}"
        predictions = {current: reference}
        for view in views:
            probability = np.full((n, 2), np.nan)
            for block, train, validation in folds:
                if not np.isin(train, public).all() or not np.isin(validation, public).all():
                    raise ValueError("Training/validation outside public indices")
                record = training_record(labels, train, validation)
                masked = np.full_like(labels, "X")
                masked[train] = labels[train]
                masked_record = training_record(masked, train, validation)
                for key, value in record.items():
                    np.testing.assert_array_equal(value, masked_record[key])
                name = f"{view}_seed{seed}_block{block}.cbm"
                np.savez_compressed(folder / (name + ".fitting.npz"), **record)
                if replay:
                    with np.load(saved / protocol / (name + ".fitting.npz")) as prior:
                        for key, value in record.items():
                            np.testing.assert_array_equal(prior[key], value)
                    model = CatBoostClassifier()
                    model.load_model(saved / protocol / name)
                    shutil.copy2(saved / protocol / name, folder / name)
                else:
                    print(f"Fit {view} seed{seed} {protocol} block{block}", flush=True)
                    model = fit_owner(inputs[view], labels, train, validation, seed)
                    model.save_model(folder / name)
                    models += 1
                value = predict_owner(model, inputs[view].iloc[validation])
                loaded = CatBoostClassifier()
                loaded.load_model(folder / name)
                np.testing.assert_array_equal(value, predict_owner(loaded, inputs[view].iloc[validation]))
                reloads += 1
                probability[validation] = value
            np.save(folder / (view + "_probability.npy"), probability[public])
            if replay:
                np.testing.assert_array_equal(probability[public], np.load(saved / protocol / (view + "_probability.npy")))
            predictions[f"{view}_seed{seed}"] = decode_owner(
                arrays, probability, support[view], broadcast, motion, raw, blocks)
        for name, prediction in predictions.items():
            table[name] = prediction[public]
            record_predictions(output, protocol, name, public, labels, folds,
                               prediction[public], rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
            if name != current:
                states[(protocol, name)] = (matching(labels[public], reference[public]),
                                           matching(labels[public], prediction[public]))
                changes[(protocol, name)] = int((prediction[public] != reference[public]).sum())
        table.to_csv(folder / "oof_predictions.csv", index=False)
    add_short_metrics(rows, segments)
    scores, diagnostics = pd.DataFrame(rows), pd.DataFrame(segments)
    scores.to_csv(output / "scores.csv", index=False)
    diagnostics.to_csv(output / "segment_diagnostics.csv", index=False)
    pd.DataFrame(details).to_csv(output / "fold_scores.csv", index=False)
    official = official_scores(output, scores)
    selection = selection_checks(scores, diagnostics, official, states, changes, seed)
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
        plan_sha256=digest(PLAN), new_models=models, checkpoint_reloads=reloads,
        manual_labels_used=False, only_public_train_indices=True, only_HA_training_targets=True,
        identical_selected_D_and_X=True, unchanged_selected_identity_exact=True,
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
