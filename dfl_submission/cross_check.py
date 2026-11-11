"""Check the DFL robustness candidate on independent SkillCorner matches."""

import argparse
import json
from pathlib import Path

from dfl_submission import skillcorner
from dfl_submission.robust import CONTROL, evaluate, predict, verified_models
from dfl_submission.skillcorner_trial import skill_halves
from dfl_submission.train import write_json
from evaluation import possession_metrics as pm


def compare(folder, skill_data, output):
    control_recipe = json.loads((CONTROL / "recipe.json").read_text())["recipe"]
    frozen = json.loads((folder / "recipe.json").read_text())
    control = verified_models(CONTROL, frozen["control_models"])
    robust = verified_models(folder, frozen["models"])
    reference = dict(control_recipe, owner_weight=0, dead_weight=0)
    manifest = json.loads((skill_data / "manifest.json").read_text())
    if manifest["revision"] != skillcorner.REVISION:
        raise ValueError("SkillCorner source revision differs")
    scores = {}
    for match in skillcorner.HOLDOUT:
        observations = []
        for i, half in enumerate(skill_halves(skill_data, match), 1):
            name = f"{match}_{i}.npz"
            if skillcorner.identity(skill_data / name) != tuple(manifest["parsed"][name]):
                raise ValueError(f"SkillCorner parsed half identity differs: {name}")
            labels = pm.label_codes(half["labels"])
            probabilities = predict(control, robust, half["positions"], half["ball"], half["teams"], half["pitch"])
            observations.append((labels, probabilities))
        predictions = {"observed": observations}
        scores[str(match)] = {"control": evaluate(predictions, reference),
                              "selected": evaluate(predictions, frozen["recipe"])}
        print(f"SkillCorner cross-check {match}: {scores[str(match)]}", flush=True)
    write_json(output, {"matches": scores, "recipe": frozen["recipe"], "skillcorner_revision": skillcorner.REVISION})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--skillcorner", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    compare(args.model_dir, args.skillcorner, args.out)
