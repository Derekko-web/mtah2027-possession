"""Append verified score rows while preserving every preceding ledger byte."""
import csv
import hashlib
from io import StringIO


def append_score_rows(ledger_path, scores_path, round_name, source_identity):
    original = ledger_path.read_bytes()
    reader = csv.DictReader(StringIO(original.decode()))
    columns = reader.fieldnames
    existing = list(reader)
    keys = ('round', 'protocol', 'model', 'source')
    indexed = {tuple(row[k] for k in keys):row for row in existing}
    if len(indexed) != len(existing):
        raise ValueError('Ledger already contains duplicate identities')
    source_sha = hashlib.sha256(scores_path.read_bytes()).hexdigest()
    additions = []
    with scores_path.open() as handle:
        for score in csv.DictReader(handle):
            row = {name:score.get(name, '') for name in columns}
            row.update(round=round_name, record_type='prospective_detector_feature_screen',
                       source=source_identity, source_sha256=source_sha,
                       stitched_spellF1=score['spellF1'])
            key = tuple(row[k] for k in keys)
            if key in indexed:
                if indexed[key] != row:
                    raise ValueError('An existing detector ledger row differs')
            else:
                additions.append(row)
                indexed[key] = row
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writerows(additions)
    if additions:
        with ledger_path.open('ab') as handle:
            if original and not original.endswith(b'\n'):
                handle.write(b'\n')
            handle.write(buffer.getvalue().encode())
    if not ledger_path.read_bytes().startswith(original):
        raise AssertionError('Ledger append changed preceding bytes')
    return dict(added=len(additions), rows_after=len(existing)+len(additions), preceding_bytes_preserved=True)
