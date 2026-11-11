"""Aggregate use of new features and conservative-label identity; no frame dumps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT/'data/control_spatial_shape_v1'
    manifest = json.loads((root/'screen/manifest.json').read_text())
    rows = []
    for head in ('shape_joint', 'shape_reduced'):
        columns = manifest['feature_columns'][head]
        shape = np.array(['shape_' in column for column in columns])
        gains = []
        for name in manifest['model_sha256']:
            if Path(name).name.startswith(head+'_'):
                model = XGBClassifier()
                model.load_model(root/'screen'/name)
                gains.append(float(model.feature_importances_[shape].sum()))
        if len(gains) != 13:
            raise AssertionError('Incomplete spatial model diagnostic')
        for protocol in ('buffered_minutes', 'purged_spells', 'buffered_100_seconds'):
            table = pd.read_csv(root/'screen'/protocol/'oof_predictions.csv')
            for mode in ('replace_current', 'live_replace', 'dead_trim'):
                prediction = table[head+'_'+mode+'_seed42']
                rows.append(dict(head=head, protocol=protocol, mode=mode,
                    changed_frames=int((prediction != table.current_seed42).sum()),
                    shape_mean_normalized_gain=float(np.mean(gains))))
    pd.DataFrame(rows).to_csv(root/'aggregate_diagnostics.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
