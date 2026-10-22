"""Consolidate aggregate historical artifacts and the fixed continuation round.

The saved CSV retains the historical aggregate evidence alongside the source.
Rebuilding its historical rows requires preserved local experiment outputs;
absence is an error rather than silently dropping failed experiments.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_FOLDERS = (
    "contact_control", "equal_model_ensemble", "equal_objective_ensemble",
    "factorized_baseline", "kinematics_baseline", "markov_decoder", "nested_baseline",
    "physics_context", "physics_context_momentum", "physics_context_persistent",
    "relative_motion_baseline", "relative_motion_baseline_segment_weights",
    "rich_baseline", "tree_ensembles",
)
EXPANDED_FOLDERS = (
    "stoppage_research", "audio_research", "tempo_research", "sequence_research",
    "camera_research", "monotone_research", "external_teacher_research", "phase_control_research",
    "stacked_dead_research", "duration_research", "pooled_stoppage_research", "episode_stoppage_research",
    "temporal_control_research", "temporal_fusion_research", "temporal_decoding_research",
    "boundary_training_research", "context_ablation_research", "catboost_research",
    "ball_dropout_research", "objective_research", "advanced_validation",
    "control_refinement", "supported_control_research", "control_blend_research",
    "control_consensus_research", "control_confirmation",
    "consensus_boundary_research",
    "control_boundary_v1/screen", "control_boundary_v1/confirmation",
    "control_renewal_v1/ownership_screen", "control_renewal_v1/transition_screen",
    "control_renewal_v1/stoppage_screen", "control_renewal_v1/hysteresis_screen",
    "control_renewal_v1/neural_screen", "control_renewal_v1/island_screen",
    "control_renewal_v1/recovery_confirmation",
    "control_precision_v1/screen", "control_precision_v1/weak_ownership_screen",
    "control_dynamics_v1/screen",
    "control_calibration_v1/screen",
    "control_broadcast_v1/screen", "control_broadcast_v1/source_screen",
    "control_broadcast_v1/intersection_screen", "control_broadcast_v1/closure_confirmation",
    "control_jersey_v1/screen",
    "control_pair_motion_v1/screen", "control_pair_motion_v1/symmetric_screen",
    "control_pair_motion_v1/dead_trim_confirmation",
    "control_ownership_graph_v1/screen",
    "control_ownership_graph_v1/conditional_screen",
)


def main():
    rows = []
    for folder in HISTORICAL_FOLDERS:
        source = ROOT / "data" / folder / "oof_scores.csv"
        scores = pd.read_csv(source)
        folds = pd.read_csv(source.with_name("fold_scores.csv"))
        for score in scores.to_dict("records"):
            name = score["model"]
            detail = folds[folds.model == name]
            if folder == "rich_baseline" and name.endswith(("_raw", "_smoothed")):
                family, decoder = name.rsplit("_", 1)
                detail = folds[(folds.model == family) & (folds.decoder == decoder)]
            minute_col = "validation_minute" if "validation_minute" in folds else "minute"
            row = dict(round="historical_handoff", record_type="historical_score",
                       protocol="buffered_minutes", model=name,
                       stitched_spellF1=score["spellF1"], macroF1=score["macroF1"],
                       predicted_possessions=score["possessions"],
                       source=str(source.relative_to(ROOT)),
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            if len(detail):
                if sorted(detail[minute_col].tolist()) != [1, 2, 3, 4, 5]:
                    raise ValueError(f"Incomplete fold evidence: {folder} {name}")
                row["mean_minute_spellF1"] = detail.spellF1.mean()
                row.update({f"minute_{int(item[minute_col])}": item["spellF1"]
                            for item in detail.to_dict("records")})
            rows.append(row)
    if len(rows) != 65:
        raise ValueError("Historical score inventory changed; investigate before rewriting the ledger")
    current = ROOT / "data/continuation_20261021"
    scores = pd.read_csv(current / "scores.csv")
    folds = pd.read_csv(current / "fold_scores.csv")
    for score in scores.to_dict("records"):
        protocol, name = score["protocol"], score["model"]
        source = current / "scores.csv"
        detail = folds[(folds.protocol == protocol) & (folds.model == name)]
        official = json.loads((current / protocol / "official/reports" / f"{name}.json").read_text())
        for column, key in (("spellF1", "spell_f1_mean"), ("macroF1", "macro_f1")):
            if abs(score[column]-official[key]) > 1e-12:
                raise ValueError("Official scorer disagrees with continuation metrics")
        row = dict(round="continuation_20261021",
                   record_type="reproduction_or_stress_reference" if name == "selected" else "new_alternative",
                   protocol=protocol, model=name,
                   stitched_spellF1=score["spellF1"], macroF1=score["macroF1"],
                   predicted_possessions=score["predicted_possessions"],
                   mean_minute_spellF1=score["mean_minute_spellF1"],
                   dead_recall=score["dead_recall"], false_dead_frames=score["false_dead_frames"],
                   missed_stoppages=score["missed_stoppages"],
                   official_bootstrap90_low=official["spell_bootstrap_low"],
                   official_bootstrap90_high=official["spell_bootstrap_high"],
                   source=str(source.relative_to(ROOT)),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        row.update({f"minute_{item['minute']}": item["spellF1"] for item in detail.to_dict("records")})
        rows.append(row)
    for folder in EXPANDED_FOLDERS:
        source = ROOT/"data"/folder/"scores.csv"
        scores = pd.read_csv(source)
        folds = pd.read_csv(source.with_name("fold_scores.csv"))
        for score in scores.to_dict("records"):
            protocol, name = score["protocol"], score["model"]
            detail = folds[(folds.protocol == protocol) & (folds.model == name)]
            expected = [1, 2, 3] if protocol == "buffered_100_seconds" else [1, 2, 3, 4, 5]
            if sorted(detail.minute.tolist()) != expected:
                raise ValueError(f"Incomplete expanded-round evidence: {folder} {protocol} {name}")
            row = dict(round=folder, record_type="adaptive_development_or_stress_check", protocol=protocol,
                       model=name, stitched_spellF1=score["spellF1"], macroF1=score["macroF1"],
                       predicted_possessions=score["predicted_possessions"],
                       mean_minute_spellF1=score["mean_minute_spellF1"],
                       dead_recall=score["dead_recall"], false_dead_frames=score["false_dead_frames"],
                       missed_stoppages=score["missed_stoppages"],
                       source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            for metric in ("short_live_mean_recall", "short_live_mostly_recovered",
                           "short_live_mean_best_iou", "short_live_matched_iou50"):
                if metric in score:
                    row[metric] = score[metric]
            row.update({f"block_{int(item['minute'])}": item["spellF1"] for item in detail.to_dict("records")})
            rows.append(row)
    # Early scratch screens are retained explicitly, including the fusion rows
    # later reproduced verbatim by research_sequence.py. They are not extra
    # independent evidence and do not have the same complete per-block artifacts.
    for folder in ("stoppage_fusion", "tempo_prior"):
        source = ROOT/"data"/folder/"screen_scores.csv"
        for score in pd.read_csv(source).to_dict("records"):
            rows.append(dict(round=folder, record_type="preliminary_scratch_screen",
                             protocol=score.get("protocol", "buffered_minutes"), model=score["model"],
                             stitched_spellF1=score["spellF1"], macroF1=score["macroF1"],
                             predicted_possessions=score["predicted_possessions"],
                             mean_minute_spellF1=score["mean_minute_spellF1"],
                             dead_recall=score["dead_recall"], false_dead_frames=score["false_dead_frames"],
                             missed_stoppages=score["missed_stoppages"], source=str(source.relative_to(ROOT)),
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    target = ROOT / "experiments/EXPERIMENT_LEDGER.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f"Saved {target}: {len(rows)} method/protocol/seed evaluations; these are not independent experiments.")


if __name__ == "__main__":
    main()
