"""Reproduce the expanded round's tracking/audio/external-data experiments.

Prerequisites: requirements-research.txt, ffmpeg, run_selected.py and
run_continuation.py. External Metrica sample data is fetched at a pinned commit;
competition assets are read locally. The selected-candidate reproduction does
not need this broad search; use validate_temporal_candidate.py instead.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    ("research_stoppages.py",), ("research_stoppages.py", "--audio"),
    ("research_stoppages.py", "--tempo"), ("research_monotone.py",),
    ("research_sequence.py",), ("research_camera.py",),
    ("external_stoppages.py",), ("research_external_teacher.py",),
    ("research_phase_control.py",), ("research_stacked_dead.py",),
    ("research_sequence.py", "--stoppage-penalty"),
    ("research_pooled_stoppages.py",), ("research_pooled_stoppages.py", "--whole-episodes"),
    ("research_temporal_control.py",), ("research_temporal_fusion.py",),
    ("research_temporal_decoding.py",), ("research_boundary_training.py",),
    ("research_context_ablation.py",), ("research_catboost.py",),
    ("research_ball_dropout.py",), ("research_objectives.py",),
)


def main():
    for name, *args in COMMANDS:
        print("Running", name, *args, flush=True)
        subprocess.run([sys.executable, str(ROOT/"experiments"/name), *args], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
