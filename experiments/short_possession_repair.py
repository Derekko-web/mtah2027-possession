"""Train-only duration-balanced ownership and bounded short-interval proposals."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from control_refinement import contact_support, smooth
from rich_features import swap_teams

HEADS = {'sqrt_spell': .5, 'equal_spell': 1.}


def features(base, raw):
    if len(base) != len(raw) or base.shape[1] != 166:
        raise ValueError('Expected original102 plus fixed64 representation')
    extra = {}
    for team in ('h__', 'a__'):
        for name in ('distance_1', 'nearest_speed', 'control_share'):
            value = raw[team+name]
            for lag in (-13, -8, -3, 3, 8, 13):
                extra[team+'short_'+name+f'_lag_{lag}'] = value.shift(lag)
    for name in ('ball_speed', 'ball_displacement_0.2s', 'ball_displacement_1s'):
        for lag in (-13, -8, -3, 3, 8, 13):
            extra['short_'+name+f'_lag_{lag}'] = raw[name].shift(lag)
    return base.join(pd.DataFrame(extra, index=base.index)).astype(np.float32)


def training_record(labels, train, power):
    labels, train = np.asarray(labels), np.asarray(train)
    if power not in HEADS.values() or train.ndim != 1 or not len(train) or not np.all(np.diff(train)>0):
        raise ValueError('Invalid duration-balanced training inputs')
    if not np.isin(labels[train], ['H', 'A', 'D']).all():
        raise ValueError('Training may use only provided H/A/D labels')
    live = train[np.isin(labels[train], ['H', 'A'])]
    target = (labels[live] == 'A').astype(np.int64)
    if len(np.unique(target)) != 2:
        raise ValueError('Both live teams are required')
    group = np.cumsum(np.r_[True, (np.diff(live)!=1) | (target[1:]!=target[:-1])])-1
    lengths = np.bincount(group)[group]
    weights = np.maximum(lengths, 8.).astype(float)**(-power)
    weights /= weights.mean()
    return dict(train=live, target=np.r_[target, 1-target], weight=np.r_[weights, weights],
                group=group, training_spell_lengths=lengths)


def fit(inputs, labels, train, head, seed):
    record = training_record(labels, train, HEADS[head])
    x = inputs.iloc[record['train']]
    paired = pd.concat([x, swap_teams(x)], ignore_index=True)
    model = CatBoostClassifier(loss_function='Logloss', depth=4, iterations=300,
        learning_rate=.04, l2_leaf_reg=10, random_seed=seed, thread_count=2,
        allow_writing_files=False, verbose=False, nan_mode='Min', boosting_type='Ordered')
    model.fit(paired, record['target'], sample_weight=record['weight'])
    return model, record


def predict(model, inputs):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Invalid ownership class order')
    return (model.predict_proba(inputs) + model.predict_proba(swap_teams(inputs))[:, ::-1])/2


def spans(states):
    edges = np.r_[0, np.flatnonzero(states[1:] != states[:-1])+1, len(states)]
    return [(int(a), int(b), str(states[a])) for a, b in zip(edges[:-1], edges[1:])]


def repair(reference, probability, tabpfn_owner, raw, blocks, mode='insert_refine'):
    """Uniform proposal rule; no targets, absolute frame rules, or protocol input.

    Proposals use 11-frame-smoothed model probabilities, 8..125-frame runs,
    mean confidence >=.65, and physical contact or independent TabPFN support.
    Existing short islands can be relocated by at most25frames per endpoint.
    All gates read unchanged inputs; overlapping edits resolve symmetrically.
    """
    reference = np.asarray(reference)
    probability, owner = np.asarray(probability), np.asarray(tabpfn_owner)
    n = len(reference)
    if (reference.ndim != 1 or probability.shape != (n, 2) or owner.shape != (n, 2)
            or len(raw)!=n or not np.isin(reference, ['H','A','D','X']).all()
            or mode not in ('insert', 'insert_refine', 'refine')):
        raise ValueError('Invalid aligned ownership repair inputs')
    support = contact_support(raw)
    result = reference.copy()
    covered = np.zeros(n, bool)
    events = []
    for left, right in blocks:
        if not 0 <= left < right <= n or covered[left:right].any():
            raise ValueError('Invalid or overlapping decode blocks')
        covered[left:right] = True
        p = probability[left:right]
        o = owner[left:right]
        if (not np.isfinite(p).all() or not np.isfinite(o).all() or (p<0).any()
                or not np.allclose(p.sum(1),1) or not np.allclose(o.sum(1),1)):
            raise ValueError('Incomplete binary ownership evidence')
        p = smooth(p, 5)
        ref = reference[left:right]
        proposals = spans(np.where(p[:,0] > p[:,1], 'H', np.where(p[:,1]>p[:,0], 'A', ref)))
        original = spans(ref)
        votes = np.zeros((right-left, 2), float)
        for a, b, team in proposals:
            if team not in ('H','A') or not 8 <= b-a <= 125 or a==0 or b==right-left:
                continue
            k = 0 if team=='H' else 1
            if p[a:b,k].mean() < .65:
                continue
            if support[left+a:left+b,k].sum() < 3 and o[a:b,k].mean() <= .5:
                continue
            other = 'A' if team=='H' else 'H'
            candidates = [(x,y) for x,y,t in original if t==team and 8<=y-x<=125
                          and min(y,b)>max(x,a) and abs(x-a)<=25 and abs(y-b)<=25]
            kind = None
            if candidates and mode in ('insert_refine','refine'):
                x,y = max(candidates, key=lambda xy: min(xy[1],b)-max(xy[0],a))
                begin,end = min(a,x),max(b,y)
                if not ((ref[max(0,begin-13):begin]==other).mean()>.5
                        and (ref[end:min(len(ref),end+13)]==other).mean()>.5):
                    continue
                kind = 'refine'
            elif not candidates and mode in ('insert','insert_refine'):
                begin,end = a,b
                if ((ref[a:b]==team).mean()>=.5 or
                        (ref[max(0,a-13):a]==other).mean()<=.5 or
                        (ref[b:min(len(ref),b+13)]==other).mean()<=.5):
                    continue
                kind = 'insert'
            if kind is None or not np.isin(ref[begin:end], ['H','A']).all():
                continue
            target = np.full(end-begin, other, dtype='<U1')
            target[a-begin:b-begin] = team
            confidence = float(p[a:b,k].mean())
            for target_team, column in (('H',0), ('A',1)):
                keep = (target==target_team) & (ref[begin:end]!=target_team)
                votes[begin:end,column][keep] = np.maximum(votes[begin:end,column][keep], confidence)
            events.append(dict(start=left+a,end_exclusive=left+b,team=team,kind=kind,
                               confidence=confidence,edited_start=left+begin,edited_end_exclusive=left+end))
        local = result[left:right]
        local[votes[:,0]>votes[:,1]] = 'H'
        local[votes[:,1]>votes[:,0]] = 'A'
    for state in ('D','X'):
        np.testing.assert_array_equal(result==state,reference==state)
    return result, events
