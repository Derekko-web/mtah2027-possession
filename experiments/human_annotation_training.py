"""Reviewed-subset supervision; human targets never become inference inputs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from build_candidate import PARAMS, objective_weights
from rich_features import swap_teams
from xgboost_baseline import CLASSES


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def reviewed_rows(document,window,freeze_hash,clip_hash):
    """Require complete independent review; exclude explicit uncertain frames."""
    if window['split']!='train' or document.get('split')!='train':raise ValueError('Audit supervision is sealed')
    if document.get('status')!='reviewed':raise ValueError('Only reviewed windows are eligible')
    annotator=document.get('annotator','').strip();reviewer=document.get('reviewer','').strip()
    if not annotator or not reviewer or annotator.casefold()==reviewer.casefold():raise ValueError('Distinct reviewer required')
    expected=dict(window_id=window['window_id'],freeze_sha256=freeze_hash,clip_sha256=clip_hash,
                  source_start=window['start'],source_end=window['end'],fps=25,label_source='human_video')
    if any(document.get(key)!=value for key,value in expected.items()):raise ValueError('Annotation provenance differs')
    coverage=np.zeros(window['end']-window['start']+1,int);rows=[]
    for segment in document.get('segments',[]):
        start,end,label=segment.get('start'),segment.get('end'),segment.get('label')
        if type(start) is not int or type(end) is not int or not 0<=start<=end<len(coverage) or label not in ('H','A','D','U'):
            raise ValueError('Invalid annotation range')
        coverage[start:end+1]+=1
        if label=='U':
            if not str(segment.get('note','')).strip():raise ValueError('Uncertainty needs a reason')
        else:rows.extend((window['start']+i,label,window['window_id']) for i in range(start,end+1))
    if not np.all(coverage==1):raise ValueError('Every frame needs exactly one reviewed label')
    return rows


def fitting_arrays(labels,train,human_frames,human_labels,validation,mass=.5):
    """Public weights stay exact; human class mass totals half public mass."""
    train=np.asarray(train,int);human_frames=np.asarray(human_frames,int);validation=np.asarray(validation,int)
    human_labels=np.asarray(human_labels)
    if len(train)!=len(np.unique(train)) or len(human_frames)!=len(np.unique(human_frames)):raise ValueError('Duplicate training rows')
    if np.intersect1d(train,human_frames).size:raise ValueError('Public/human rows overlap')
    all_frames=np.r_[train,human_frames]
    if len(validation) and np.any((all_frames>=validation.min()-250)&(all_frames<=validation.max()+250)):
        raise ValueError('Validation exclusion buffer violated')
    if set(labels[train])!=set(CLASSES) or not len(human_frames) or len(human_frames)!=len(human_labels) or not np.isin(human_labels,CLASSES).all():
        raise ValueError('Unsupported fitting labels')
    public_target=np.array([np.flatnonzero(CLASSES==label)[0] for label in labels[train]])
    human_target=np.array([np.flatnonzero(CLASSES==label)[0] for label in human_labels])
    yp,wp=objective_weights(public_target,train,False)
    yh,wh=objective_weights(human_target,human_frames,False)
    if set(yh)!=set(range(3)):raise ValueError('Human source must support all H/A/D classes after symmetry')
    wh=wh*(float(mass)*wp.sum()/wh.sum())
    return dict(public_frames=train,human_frames=human_frames,target=np.r_[yp,yh],weight=np.r_[wp,wh])


def fit_human(inputs,arrays,seed=42):
    public=inputs.iloc[arrays['public_frames']];human=inputs.iloc[arrays['human_frames']]
    augmented=pd.concat([public,swap_teams(public),human,swap_teams(human)],ignore_index=True)
    model=XGBClassifier(**dict(PARAMS,max_depth=3,n_estimators=160,random_state=seed))
    model.fit(augmented,arrays['target'],sample_weight=arrays['weight'])
    return model


def append_scores(ledger,scores,round_name):
    """Idempotent append without changing any prior byte or score identity."""
    original=ledger.read_bytes();reader=csv.DictReader(io.StringIO(original.decode()))
    fields=reader.fieldnames;old=list(reader);keys=('round','protocol','model','source')
    index={tuple(row[k] for k in keys):row for row in old};source=str(scores.relative_to(ledger.parents[1]));sha=digest(scores)
    if len(index)!=len(old):raise ValueError('Duplicate existing ledger identities')
    additions=[]
    with scores.open() as handle:
        for value in csv.DictReader(handle):
            row={key:value.get(key,'') for key in fields}
            row.update(round=round_name,record_type='reviewed_human_annotation_pilot',source=source,source_sha256=sha,stitched_spellF1=value['spellF1'])
            key=tuple(row[k] for k in keys)
            if key in index:
                if index[key]!=row:raise ValueError('Previously recorded pilot score differs')
            else:additions.append(row);index[key]=row
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');writer.writerows(additions)
    if additions:
        with ledger.open('ab') as handle:
            if original and not original.endswith(b'\n'):handle.write(b'\n')
            handle.write(buffer.getvalue().encode())
    if not ledger.read_bytes().startswith(original):raise AssertionError('Ledger prefix changed')
    return dict(added=len(additions),rows_after=len(index),prior_bytes_preserved=True)
