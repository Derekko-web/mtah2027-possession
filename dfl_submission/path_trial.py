"""Select and validate an alternative temporal decoder using external DFL only."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission.build import write_csv
from dfl_submission.decoder import decode
from dfl_submission.path_decoder import decode_path
from dfl_submission.robust import CONTROL, verified_models
from dfl_submission.train import DEVELOPMENT, HOLDOUT, SOURCE_MANIFEST, TRAIN, match_predictions, write_json
from dfl_transfer_data import SECTIONS, digest
from evaluation import possession_metrics as pm


def cache(path, hashes):
    if digest(path) != hashes[path.name]:
        raise ValueError(f"External cache identity differs: {path}")
    return np.load(path, allow_pickle=False)


def cached_predictions(transfer, stoppage, model_hashes, match):
    owner = transfer / "external_trial"
    dead = stoppage / "external_trial"
    labels = transfer / "assessment_A"
    for folder, name in ((owner, "ownership.ubj"), (dead, "dead.ubj")):
        if digest(folder / "teacher_seed42.ubj") != model_hashes[name]:
            raise ValueError("External predictions came from a different checkpoint")
    owner_hashes, dead_hashes, label_hashes = (
        json.loads((folder / "manifest.json").read_text())["cache_sha256"]
        for folder in (owner, dead, labels)
    )
    views = {"clean": [], "masked": []}
    for half in SECTIONS:
        name = f"DFL-MAT-J03{match}_{half}"
        with cache(labels / f"{name}.npz", label_hashes) as parsed:
            truth = pm.label_codes(parsed["labels"])
        for view, halves in views.items():
            home_p = cache(owner / f"{name}_{view}_teacher.npy", owner_hashes)
            dead_p = cache(dead / f"{name}_{view}_dead.npy", dead_hashes)
            halves.append((truth, home_p, dead_p, None))
    return views


def score(predictions, decoder, recipe):
    result = {}
    for view, halves in predictions.items():
        first, second = halves
        truth = np.r_[first[0], pm.X, second[0]]
        labels = np.r_[decoder(first[1], first[2], **recipe), pm.X, decoder(second[1], second[2], **recipe)]
        metrics = pm.spell_scores(truth, labels, pm.valid_runs(truth))
        cm = pm.scored_confusion(pm.confusion(truth, labels))
        result[view] = {"spell_f1": metrics["mean_f1"], "macro_f1": pm.macro_f1(cm), "spells": metrics["n_pred"]}
    return result


def rank(metrics):
    return tuple((metrics["clean"][k] + 2 * metrics["masked"][k]) / 3 for k in ("spell_f1", "macro_f1"))


def tune(predictions, frozen, output):
    output.mkdir(parents=True, exist_ok=False)
    reference = frozen["recipe"]
    dev = predictions(DEVELOPMENT)
    trials = []
    for cutoff, dead_cost, owner_cost, radius in product((.4, .5, .6), (4, 10, 22), (4, 10, 22), (0, 20)):
        recipe = dict(dead_cutoff=cutoff, dead_cost=dead_cost, owner_cost=owner_cost, radius=radius)
        trials.append(dict(recipe=recipe, metrics=score(dev, decode_path, recipe)))
    chosen = max(trials, key=lambda trial: rank(trial["metrics"]))
    write_json(output / "recipe.json", dict(models=frozen["models"], recipe=chosen["recipe"]))
    result = {DEVELOPMENT: {"control": score(dev, decode, reference), "selected": chosen["metrics"]}}
    for match in HOLDOUT:
        held = predictions(match)
        result[match] = {"control": score(held, decode, reference), "selected": score(held, decode_path, chosen["recipe"])}
    gains = {m: rank(s["selected"])[0] - rank(s["control"])[0] for m, s in result.items()}
    passed = all(gain > 0 for gain in gains.values()) and all(
        result[match]["selected"][view]["spell_f1"] >= result[match]["control"][view]["spell_f1"] - .02
        for match in HOLDOUT for view in ("clean", "masked"))
    write_json(output / "external_validation.json", {
        "holdout_passed": passed, "matches": result, "development_search": trials, "weighted_gains": gains,
        "gate": "Positive weighted SpellF1 gain on all three evaluation matches; no holdout view loses more than .02",
        "train": TRAIN, "dataset_manifest_sha256": digest(SOURCE_MANIFEST),
    })
    print(f"Path decoder comparison passed: {passed}; {chosen['recipe']}; weighted gains {gains}", flush=True)


def build(folder, output):
    from dfl_submission import models as shared
    from dfl_submission.build import tactical_halves

    report = json.loads((folder / "external_validation.json").read_text())
    if report["holdout_passed"] is not True:
        raise ValueError("Path decoder failed its external comparison")
    frozen = json.loads((folder / "recipe.json").read_text())
    models = verified_models(CONTROL, frozen["models"])
    labels = tactical_halves(lambda positions, ball, teams, pitch: decode_path(
        *shared.predict(*models, positions, ball, teams)[:2], **frozen["recipe"]))
    write_csv(output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fitting = commands.add_parser("tune")
    fitting.add_argument("--dfl-raw", type=Path)
    fitting.add_argument("--external-cache", nargs=2, type=Path, metavar=("OWNERSHIP", "STOPPAGE"))
    fitting.add_argument("--out", required=True, type=Path)
    inference = commands.add_parser("build")
    inference.add_argument("--model-dir", required=True, type=Path)
    inference.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "tune":
        frozen = json.loads((CONTROL / "recipe.json").read_text())
        if args.external_cache:
            tune(lambda match: cached_predictions(*args.external_cache, frozen["models"], match), frozen, args.out)
        elif args.dfl_raw:
            models = verified_models(CONTROL, frozen["models"])
            source = json.loads(SOURCE_MANIFEST.read_text())
            tune(lambda match: match_predictions(args.dfl_raw, source, match, models), frozen, args.out)
        else:
            fitting.error("provide --dfl-raw or --external-cache")
    else:
        build(args.model_dir, args.out)
