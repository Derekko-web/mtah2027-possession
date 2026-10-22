"""Cache native directional interactions; verify actual padded-prefix parity."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from ownership_graph import graph_views
from prepare_broadcast_alignment import sliced


def main():
    root = ROOT/'data/control_ownership_graph_v1'
    output = root/'inputs'
    output.mkdir(exist_ok=False)
    source = ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(source)
    plan = ROOT/'experiments/OWNERSHIP_GRAPH_PLAN.json'
    (output/'plan.json').write_bytes(plan.read_bytes())
    tactical, broadcast = [dict(np.load(source/(name+'_observations.npz'))) for name in ('tactical', 'broadcast')]
    mapping = dict(np.load(source/'strict_mapping.npz'))
    print('Computing local native directional interactions; no targets or media', flush=True)
    features, info = graph_views(tactical, broadcast, mapping)
    n = 18935
    short_mapping = {key:value[:n] if len(value) == len(tactical['positions']) else value for key, value in mapping.items()}
    short_features, short_info = graph_views(sliced(tactical, n), broadcast, short_mapping)
    pd.testing.assert_frame_equal(features.iloc[11185:18685], short_features.iloc[11185:18685])
    pd.testing.assert_frame_equal(info.iloc[11185:18685], short_info.iloc[11185:18685])
    features.to_pickle(output/'graph_inputs.pkl')
    info.to_pickle(output/'graph_info.pkl')
    diagnostics = dict(feature_columns=len(features.columns), tactical_observed=float(info.tg_seen.iloc[11185:18685].mean()),
                       broadcast_observed=float(info.bg_seen.iloc[11185:18685].mean()))
    names = ('ownership_graph.py', 'prepare_ownership_graph.py', 'pair_motion.py')
    manifest = dict(plan_sha256=digest(plan), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256={'data/control_broadcast_v1/inputs/manifest.json':digest(source/'manifest.json')},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        actual_padded_prefix_parity=True, targets_not_read=True, diagnostics=diagnostics)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == '__main__':
    main()
