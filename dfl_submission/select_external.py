"""Select among predeclared candidates using only external holdout reports."""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission.skillcorner_trial import dfl_average, skill_average
from dfl_submission.train import HOLDOUT, write_json


def select(robust_folder, skill_folder, cross_report, fusion_report=None):
    robust = json.loads((robust_folder / "external_validation.json").read_text())
    skill = json.loads((skill_folder / "external_validation.json").read_text())
    cross = json.loads(cross_report.read_text())
    skill_scores, dfl_scores = skill["skillcorner_holdout"], skill["dfl_holdout"]
    robust_dfl = {match: dict(control=robust["matches"][match]["control"],
                             selected=robust["matches"][match]["robust"]) for match in HOLDOUT}
    if robust["holdout_passed"] and cross["recipe"] != json.loads((robust_folder / "recipe.json").read_text())["recipe"]:
        raise ValueError("Cross-check used a different robustness recipe")

    def candidate(dfl, other, key, passed):
        dfl_score = float(np.mean([dfl_average(dfl[match], key) for match in HOLDOUT]))
        skill_score = skill_average(other, key)
        return dict(dfl_spell_f1=dfl_score, skillcorner_spell_f1=skill_score,
                    combined_spell_f1=(dfl_score + skill_score) / 2, passed=passed)

    control = candidate(dfl_scores, skill_scores, "control", True)
    other_control = candidate(robust_dfl, cross["matches"], "control", True)
    for metric in ("dfl_spell_f1", "skillcorner_spell_f1"):
        if abs(control[metric] - other_control[metric]) > 1e-8:
            raise ValueError(f"External controls do not agree on {metric}")
    candidates = {
        "dfl_control": control,
        "dfl_robust": candidate(robust_dfl, cross["matches"], "selected", robust["holdout_passed"]),
        "skillcorner_blend": candidate(dfl_scores, skill_scores, "selected", skill["holdout_passed"]),
    }
    if fusion_report is not None:
        fusion = json.loads(fusion_report.read_text())
        dfl, other = fusion["dfl_holdout"], fusion["skillcorner_holdout"]
        fusion_control = candidate(dfl, other, "control", True)
        for metric in ("dfl_spell_f1", "skillcorner_spell_f1"):
            if abs(candidates["dfl_robust"][metric] - fusion_control[metric]) > 1e-8:
                raise ValueError(f"Fusion control does not reproduce the richer DFL score on {metric}")
        candidates["dfl_skillcorner_fusion"] = candidate(dfl, other, "selected", fusion["holdout_passed"])
    for result in candidates.values():
        result["passed"] = result["passed"] and all(result[key] >= control[key] - .02
            for key in ("dfl_spell_f1", "skillcorner_spell_f1"))
    selected = max((name for name, result in candidates.items() if result["passed"]),
                   key=lambda name: candidates[name]["combined_spell_f1"])
    return dict(selected=selected, candidates=candidates, method=(
        "Equal weight per dataset; DFL weights clean 1/3 and masked 2/3. "
        "Candidate must pass its own gate and stay within .02 of control on each dataset. "
        "These external holdouts are used to choose the final model family."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl-robust", type=Path, required=True)
    parser.add_argument("--skillcorner-blend", type=Path, required=True)
    parser.add_argument("--cross-report", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.dfl_robust, args.skillcorner_blend, args.cross_report, args.fusion_report)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
