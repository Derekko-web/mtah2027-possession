"""User-directed spellF1 decision layer; all original diagnostics stay intact."""
import hashlib
import json
from pathlib import Path

import tabpfn_fixed_confirmation as native

ROOT, ROUND = native.ROOT, native.ROUND
POLICY = ROUND / "ranking_policy"
PLAN = native.PLAN
SOURCES = native.SOURCES


def digest(path):
    return native.digest(path)


def read(path):
    return json.loads(Path(path).read_text())


def identity():
    return {name: digest(ROOT / name) for name in SOURCES}


def verify_policy():
    native.verify()
    if read(ROUND/"user_ranking_direction.json").get("user_message")!="spellf1 is the main score that determines rankings.":
        raise ValueError("The explicit spellF1-first direction changed")


def rank_rows(rows, seed):
    expected = set(native.PROTOCOLS)
    method, reference = f"tabpfn_short_control_seed{seed}", f"current_seed{seed}"
    index = {(r["protocol"], r["model"]): r for r in rows}
    pairs = []
    for protocol in native.PROTOCOLS:
        old, new = index[protocol, reference], index[protocol, method]
        before, after = float(old["spellF1"]), float(new["spellF1"])
        checks = dict(primary_non_decreasing=after >= before - 1e-12)
        if protocol == "buffered_minutes":
            checks["main_gain_at_least_0_01"] = after >= before + .01 - 1e-12
            if seed == 42:
                checks["saved_main_reproduced"] = abs(after - .7771428571428571) < 1e-12
        pairs.append(dict(protocol=protocol, seed=seed, model=method, spellF1=after,
            reference_spellF1=before, spellF1_delta=after-before, checks=checks))
    passed = (len(pairs) == 3 and {r["protocol"] for r in pairs} == expected
              and all(all(r["checks"].values()) for r in pairs))
    return dict(passed=passed, selected=method if passed else None, eligible=[method] if passed else [],
        seed=seed, pairs=pairs, performance_policy="user_directed_official_spellF1",
        original_secondary_quality_checks_remain_reported=True,
        replacement_authorized=False)


def assess_stage(stage, folder, seed, replay=False):
    import csv
    verify_policy()
    native.audit(folder)
    rows = list(csv.DictReader((folder / "scores.csv").open()))
    result = rank_rows(rows, seed)
    result.update(stage=stage, replay=replay, scores_sha256=digest(folder / "scores.csv"),
        source_manifest_sha256=digest(folder / "manifest.json"),
        strict_original_selection_sha256=digest(folder / "selection.json"),
        policy_plan_sha256=digest(PLAN), user_direction_sha256=digest(ROUND / "user_ranking_direction.json"))
    name = stage + ("_replay" if replay else "") + "_selection.json"
    path = POLICY / name
    with path.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def ranking_selections():
    return [read(POLICY / (stage + "_selection.json"))
            for stage in ("screen", "confirmation_seed43", "confirmation_seed44")]


def all_nine(selections):
    if len(selections) != 3:
        return False
    for seed, item in zip((42,43,44), selections):
        if item.get("passed") is not True or item.get("seed") != seed:
            return False
        expected = rank_rows([dict(protocol=p["protocol"], model=p["model"], spellF1=p["spellF1"])
                              for p in item["pairs"]] +
                             [dict(protocol=p["protocol"], model=f"current_seed{seed}", spellF1=p["reference_spellF1"])
                              for p in item["pairs"]], seed)
        if not expected["passed"] or expected["pairs"] != item["pairs"]:
            return False
    return True


def require_seed42():
    verify_policy()
    item = read(POLICY / "screen_selection.json")
    folder = ROUND / "screen"
    native.audit(folder)
    if (item.get("passed") is not True or item["scores_sha256"] != digest(folder / "scores.csv")
            or item["source_manifest_sha256"] != digest(folder / "manifest.json")):
        raise ValueError("Complete seed42 ranking-first evidence must pass before confirmations")


def require_confirmations():
    verify_policy()
    items = ranking_selections()
    for stage, item in zip(("screen", "confirmation_seed43", "confirmation_seed44"), items):
        native.audit(ROUND / stage)
        if (item["scores_sha256"] != digest(ROUND / stage / "scores.csv")
                or item["source_manifest_sha256"] != digest(ROUND / stage / "manifest.json")):
            raise ValueError("Ranking decision is not bound to its complete validation stage")
    if not all_nine(items):
        raise ValueError("All nine matched seed/protocol spellF1 comparisons must pass")
