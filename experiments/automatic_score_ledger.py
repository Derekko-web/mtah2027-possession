"""Append automatic trial scores with an explicit record type and byte preservation."""
import csv
import hashlib
from io import StringIO


def append_scores(ledger_path,scores_path,round_name,record_type,source_identity):
    original=ledger_path.read_bytes();reader=csv.DictReader(StringIO(original.decode()))
    columns=reader.fieldnames;existing=list(reader);keys=('round','protocol','model','source')
    indexed={tuple(row[k] for k in keys):row for row in existing}
    if len(indexed)!=len(existing):raise ValueError('Existing ledger identities are not unique')
    source_sha=hashlib.sha256(scores_path.read_bytes()).hexdigest();additions=[]
    with scores_path.open() as handle:
        for score in csv.DictReader(handle):
            row={name:score.get(name,'') for name in columns}
            row.update(round=round_name,record_type=record_type,source=source_identity,source_sha256=source_sha,stitched_spellF1=score['spellF1'])
            key=tuple(row[k] for k in keys)
            if key in indexed:
                if indexed[key]!=row:raise ValueError('Existing score identity differs')
            else:additions.append(row);indexed[key]=row
    buffer=StringIO();csv.DictWriter(buffer,fieldnames=columns,lineterminator='\n').writerows(additions)
    if additions:
        with ledger_path.open('ab') as handle:
            if original and not original.endswith(b'\n'):handle.write(b'\n')
            handle.write(buffer.getvalue().encode())
    if not ledger_path.read_bytes().startswith(original):raise AssertionError('Ledger preceding bytes changed')
    return dict(added=len(additions),rows_after=len(existing)+len(additions),preceding_bytes_preserved=True)
