"""Fast official-score regression check from retained public OOF predictions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(folder, column="tabpfn_context_seed42"):
    import pandas as pd
    from evaluation import possession_metrics as pm
    rows = []
    for protocol in ("buffered_minutes", "buffered_100_seconds"):
        table = pd.read_csv(folder / protocol / "oof_predictions.csv")
        truth = pm.label_codes(table.ground_truth)
        values = {}
        for model in ("current_seed42", column):
            values[model] = pm.evaluate(truth, pm.label_codes(table[model]), bootstrap_n=0)["spell_f1_mean"]
        rows.append(dict(protocol=protocol, **values))
    main, longer = rows
    checks = dict(main_at_least_saved_score=main[column] >= .7771428571428571 - 1e-12,
                  longer_at_least_current=longer[column] >= longer["current_seed42"] - 1e-12)
    return dict(passed=all(checks.values()), checks=checks, scores=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "data/control_tabpfn_extended_v1/screen")
    parser.add_argument("--column", default="tabpfn_context_seed42")
    args = parser.parse_args()
    result = check(args.folder, args.column)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
