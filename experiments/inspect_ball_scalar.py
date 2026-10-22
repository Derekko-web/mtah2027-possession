"""Inspect the supplied third ball field locally without exposing input excerpts."""

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import require, source_frames


def ball_scalar_spec(path):
    tree = ET.parse(path)
    refs = [e.attrib.get('channelId') for e in tree.iter() if e.tag.rsplit('}', 1)[-1] == 'BallChannelRef']
    if refs != ['x', 'y', 's']:
        raise ValueError('Expected exactly three declared ball channels in x/y/s order')
    channel = next(e for e in tree.iter() if e.tag.rsplit('}', 1)[-1] == 'Channel' and e.attrib.get('id') == 's')
    unit = re.sub(r'[^a-z/]', '', channel.findtext('Unit', '').lower())
    if unit in ('km/h', 'kph', 'kmph', 'kmh', 'kilometersperhour', 'kilometresperhour'):
        name, factor = 'kilometers_per_hour', 1/3.6
    elif unit in ('m/s', 'mps', 'meters/second', 'metres/second', 'meterspersecond', 'metrespersecond'):
        name, factor = 'meters_per_second', 1.
    else:
        raise ValueError('Supplied ball scalar has an unrecognized declared speed unit')
    return dict(declared_xy_s_order=True, declared_speed=True, declared_unit_class=name,
                raw_to_declared_meters_per_second=factor, calibration_not_independently_verified=True)


def load_ball_fields(source='tactical', limit=None):
    meta, raw = require(['atd_meta', 'atd_raw'], source=source)
    spec = ball_scalar_spec(meta)
    n = source_frames(source) if limit is None else int(limit)
    values = np.full((n, 3), np.nan)
    count = 0
    with open(raw) as handle:
        for i, line in enumerate(handle):
            if i >= n:
                break
            frame, _, ball = line.rstrip().split(':')
            row = np.fromstring(ball, sep=',')
            if int(frame) != i+1 or row.shape != (3,):
                raise ValueError('Supplied ball record format or frame order changed')
            values[i] = row
            count += 1
    if count != n:
        raise ValueError('Supplied ball records ended early')
    return values, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['tactical', 'broadcast'], default='tactical')
    args = parser.parse_args()
    values, spec = load_ball_fields(args.source)
    xy = np.isfinite(values[:, :2]).all(axis=1)
    scalar = np.isfinite(values[:, 2])
    result = dict(source=args.source, **spec, rows=len(values),
        scalar_finite=int(scalar.sum()), scalar_and_xy=int((scalar & xy).sum()),
        scalar_without_xy=int((scalar & ~xy).sum()), xy_without_scalar=int((~scalar & xy).sum()),
        neither=int((~scalar & ~xy).sum()),
        raw_scalar_quantiles=np.quantile(values[scalar, 2], [0, .25, .5, .75, .95, .99, 1]).tolist())
    output = ROOT/'data/control_ball_location_v1'
    (output/(args.source+'_scalar_schema.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
