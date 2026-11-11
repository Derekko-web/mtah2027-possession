"""Execution-only repair for the frozen WASB native prefix-parity witness.

The cached ball source already applies the producer's pitch/validity cleanup.
The original witness compared it with an uncleaned prefix. This wrapper applies
the identical cleanup to that witness while preserving all frozen downstream
sampling, inference, readout, and opportunity gates.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from player_representation import ROOT, digest, write_json
import wasb_native as native

ORIGINAL = ROOT / "data/control_wasb_native_v1"
DEFAULT = ROOT / "data/control_wasb_native_repair_v1"
PUBLIC = slice(11185, 18685)


def cleaned_ball(ball, valid, pitch):
    """Exact prepare_dfl_dead_native.observations validity/bounds convention."""
    ball = np.asarray(ball).copy()
    ball[~np.asarray(valid, bool)] = np.nan
    pitch = np.asarray(pitch)
    usable = (np.isfinite(ball).all(axis=1) & (ball >= -5).all(axis=1)
              & (ball <= pitch + 5).all(axis=1))
    ball[~usable] = np.nan
    return ball


def diagnosis():
    """Prove the cleanup difference on both actual prefixes without targets."""
    from broadcast_alignment import load_source
    from games import GAME, GAMES
    pitch = np.asarray(GAMES[GAME]["pitch"])
    with np.load(ROOT / "data/control_dfl_dead_v1/native_inputs/native_observations.npz") as source:
        cached_ball, cached_positions, cached_teams = source["ball"], source["positions"], source["teams"]
    reports = []
    for length in (18935, 19060):
        source = load_source("tactical", length)
        positions = source["positions"].copy()
        positions[~source["valid"]] = np.nan
        np.testing.assert_array_equal(positions[PUBLIC], cached_positions[PUBLIC])
        np.testing.assert_array_equal(source["teams"], cached_teams)
        raw = source["ball"]
        clean = cleaned_ball(raw, source["valid"], pitch)
        np.testing.assert_array_equal(clean[PUBLIC], cached_ball[PUBLIC])
        both = np.isfinite(raw[PUBLIC]).all(1) & np.isfinite(cached_ball[PUBLIC]).all(1)
        np.testing.assert_array_equal(raw[PUBLIC][both], cached_ball[PUBLIC][both])
        changed = np.flatnonzero(np.isnan(raw[PUBLIC]).any(1) != np.isnan(cached_ball[PUBLIC]).any(1)) + PUBLIC.start
        reports.append(dict(
            prefix=length, raw_cached_nan_disagreement_rows=len(changed),
            invalid_homography_rows=int((~source["valid"][changed]).sum()),
            outside_pitch_rows=int(((raw[changed] < -5).any(1) | (raw[changed] > pitch + 5).any(1)).sum()),
            shared_finite_values_exact=True, cleaned_cached_exact=True,
            positions_and_teams_exact=True,
        ))
    return dict(targets_loaded=False, changed_sampling=False, changed_cached_ball=False, reports=reports)


def synthetic():
    value = np.array([[1., 2.], [120., 2.], [1., 90.], [-6., 4.],
                      [np.nan, 5.], [10., 20.], [-5., 73.]])
    valid = np.array([True, True, True, True, True, False, True])
    expected = value.copy()
    expected[1:6] = np.nan
    np.testing.assert_array_equal(cleaned_ball(value, valid, [105., 68.]), expected)
    np.testing.assert_array_equal(value[1], [120., 2.])
    np.testing.assert_array_equal(cleaned_ball(cleaned_ball(value, valid, [105., 68.]), valid, [105., 68.]), expected)
    return dict(passed=True, synthetic_cleanup_boundary_and_idempotence=True, source_unchanged=True)


def bind(output):
    """Keep the original failure/freeze intact and bind this added wrapper."""
    native.FOLDER = ORIGINAL
    native.validate()
    output.mkdir(parents=True, exist_ok=True)
    freeze = output / "protocol_freeze"
    if not freeze.exists():
        shutil.copytree(ORIGINAL / "protocol_freeze", freeze)
    for old in (ORIGINAL / "protocol_freeze").iterdir():
        if old.is_file() and (freeze / old.name).read_bytes() != old.read_bytes():
            raise ValueError("Original frozen record differs: " + old.name)
    names = ("experiments/wasb_native.py", "experiments/prepare_dfl_dead_native.py",
             "experiments/wasb_native_repair.py", "data/control_wasb_native_v1/prepare.log")
    binding = dict(
        source_sha256={name: digest(ROOT / name) for name in names},
        unchanged_plan_sha256=digest(native.PLAN),
        original_failure_preserved=True, original_inputs_preserved=True,
        repair="Apply existing ball-source cleanup only to preparation parity witness; cached tracked ball, sampling, neural inference, contact readout and gates unchanged",
    )
    path = freeze / "execution_repair_binding.json"
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError("Execution repair binding changed")
    else:
        write_json(path, binding)
    native.FOLDER = output
    native.validate()


def main(phase, output):
    if output.resolve() == ORIGINAL.resolve():
        raise ValueError("Use a separate folder to preserve the failed preparation")
    bind(output)
    if phase in ("diagnose", "prepare"):
        write_json(output / "synthetic_repair.json", synthetic())
        write_json(output / "preparation_diagnosis.json", diagnosis())
    if phase == "diagnose":
        print("Both actual prefixes equal the immutable cleaned source; no targets read", flush=True)
        return
    if phase == "prepare":
        import broadcast_alignment
        from games import GAME, GAMES
        original_load = broadcast_alignment.load_source
        def canonical_load(source, n_frames=None):
            result = original_load(source, n_frames)
            if source == "tactical":
                result["ball"] = cleaned_ball(result["ball"], result["valid"], GAMES[GAME]["pitch"])
            return result
        broadcast_alignment.load_source = canonical_load
        try:
            native.prepare()
        finally:
            broadcast_alignment.load_source = original_load
    else:
        getattr(native, phase)()
    if phase == "infer":
        summary = json.loads((output / "outputs/summary.json").read_text())
        checks = dict(cpu_memory=summary["cpu_peak_GiB"] <= 2.5,
                      mps_memory=summary["MPS_driver_GiB"] <= 2.,
                      execution_target=summary["elapsed_seconds"] <= 900.)
        write_json(output / "resource_checks.json", dict(passed=all(checks.values()), checks=checks))
        if not all(checks.values()):
            raise ValueError("Frozen native inference resource bounds failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("diagnose", "prepare", "infer", "contacts", "audit"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    main(args.phase, args.output)
