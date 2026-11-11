"""Fixed local ownership probe helpers; competition media never leaves loopback."""
import json
import re
import urllib.request
import numpy as np

ENDPOINT = 'http://127.0.0.1:11434'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Local inference redirect forbidden')


def local_json(route, body=None):
    if route not in ('/api/tags', '/api/version', '/api/generate'):
        raise ValueError('Local inference route forbidden')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(ENDPOINT+route,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    with opener.open(request, timeout=600) as response:
        return json.load(response)


def require_model(plan):
    models = local_json('/api/tags')['models']
    matches = [m for m in models if m['name'] == plan['model'] and m['digest'] == plan['model_digest']]
    if len(matches) != 1 or local_json('/api/version')['version'] != plan['runtime_version']:
        raise ValueError('Pinned local model/runtime unavailable')


def parse_ownership(text):
    value = str(text).strip().lower()
    match = re.fullmatch(r'(home|away|stopped|unclear)[.!]?', value)
    return {'home':'H', 'away':'A', 'stopped':'D', 'unclear':'X'}[match[1]] if match else 'X'


def allowed_frame(frame, reserved, held, exclude_public=False):
    if exclude_public and 11185-250 <= frame < 18685+250:
        return False
    if any(w['start']-250 <= frame <= w['end']+250 for w in reserved):
        return False
    return not np.any(np.abs(np.asarray(held)-frame) <= 250)


def sample_centers(valid, blocks, reserved, held):
    selected = []
    for center in range(11310, 18685, 250):
        frames = [center-12, center, center+12]
        if not any(start <= frames[0] and frames[-1] <= end for _, start, end in blocks):
            continue
        if all(valid[f] and allowed_frame(f, reserved, held) for f in frames):
            selected.append(center)
    return selected
