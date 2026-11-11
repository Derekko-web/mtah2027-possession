"""Team formation and surrounding-space predictors without track identities."""
import numpy as np
import pandas as pd
from provided_speed import bounded_statistics

FIELDS = ['count', 'scale', 'eigen_ratio', 'logarea', 'pair_distance', 'opponent_center_distance',
          'center_ball_distance', 'normalized_center_ball_distance', 'mahalanobis',
          'projection', 'along_spread', 'across_spread']
FIELDS += [f'density{r}' for r in (2, 5, 10, 20)]
FIELDS += [f'ring{r}' for r in (5, 10, 20)]
SUMMARY = ['logarea', 'normalized_center_ball_distance', 'ring10']


def spatial_core(positions, ball, teams, valid, blocks):
    n = len(ball)
    result = {prefix+'shape_'+field:np.full(n, np.nan) for prefix in ('h__', 'a__') for field in FIELDS}
    result['shape_ball_seen'] = (np.isfinite(ball).all(axis=1) & valid).astype(float)
    result['shape_ball_speed5'] = np.full(n, np.nan)
    angle = np.arange(8)*np.pi/4
    bearings = np.c_[np.cos(angle), np.sin(angle)]
    for left, right in blocks:
        left, right = max(0, left), min(n, right)
        for frame in range(left, right):
            if not valid[frame]:
                continue
            xy = [positions[frame, teams == team] for team in (0, 1)]
            xy = [p[np.isfinite(p).all(axis=1)] for p in xy]
            for team, prefix in enumerate(('h__', 'a__')):
                key = prefix+'shape_'
                own, other = xy[team], xy[1-team]
                result[key+'count'][frame] = len(own)
                if len(own) < 5:
                    continue
                center = own.mean(axis=0)
                delta = own-center
                covariance = delta.T@delta/len(own)+np.eye(2)
                eigen = np.linalg.eigvalsh(covariance)
                scale = np.sqrt(eigen.sum())
                result[key+'scale'][frame] = scale
                result[key+'eigen_ratio'][frame] = eigen[0]/eigen[1]
                result[key+'logarea'][frame] = np.log(eigen).sum()/2
                pair = np.linalg.norm(own[:, None]-own[None], axis=2)
                result[key+'pair_distance'][frame] = pair[np.triu_indices(len(own), 1)].mean()
                if len(other) >= 5:
                    result[key+'opponent_center_distance'][frame] = np.linalg.norm(other-center, axis=1).min()
                if not result['shape_ball_seen'][frame]:
                    continue
                offset = ball[frame]-center
                distance = np.linalg.norm(own-ball[frame], axis=1)
                result[key+'center_ball_distance'][frame] = np.linalg.norm(offset)
                result[key+'normalized_center_ball_distance'][frame] = np.linalg.norm(offset)/scale
                result[key+'mahalanobis'][frame] = np.sqrt(offset@np.linalg.solve(covariance, offset))
                for radius in (2, 5, 10, 20):
                    result[key+f'density{radius}'][frame] = np.exp(-.5*(distance/radius)**2).mean()
                if len(other) < 5:
                    continue
                axis = other.mean(axis=0)-center
                length = np.linalg.norm(axis)
                if length > 1e-6:
                    axis /= length
                    across = np.array([-axis[1], axis[0]])
                    result[key+'projection'][frame] = offset@axis
                    result[key+'along_spread'][frame] = np.sqrt(axis@covariance@axis)
                    result[key+'across_spread'][frame] = np.sqrt(across@covariance@across)
                for radius in (5, 10, 20):
                    points = ball[frame]+radius*bearings
                    own_distance = np.linalg.norm(points[:, None]-own[None], axis=2).min(axis=1)
                    other_distance = np.linalg.norm(points[:, None]-other[None], axis=2).min(axis=1)
                    margin = np.clip((other_distance-own_distance)/2, -40, 40)
                    result[key+f'ring{radius}'][frame] = (1/(1+np.exp(-margin))).mean()
        for frame in range(left+5, right):
            if valid[frame-5:frame+1].all() and np.isfinite(ball[[frame-5, frame]]).all():
                speed = np.linalg.norm(ball[frame]-ball[frame-5])*5
                if speed <= 80:
                    result['shape_ball_speed5'][frame] = speed
    return pd.DataFrame(result).astype(np.float32)


def shape_features(positions, ball, teams, valid, blocks):
    core = spatial_core(positions, ball, teams, valid, blocks)
    extra = {}
    for prefix in ('h__', 'a__'):
        for field in SUMMARY:
            column = prefix+'shape_'+field
            for radius in (25, 75):
                for direction in ('past', 'future'):
                    mean, std = np.full(len(core), np.nan), np.full(len(core), np.nan)
                    for left, right in blocks:
                        left, right = max(0, left), min(len(core), right)
                        if left < right:
                            values = bounded_statistics(core[column].iloc[left:right], radius, direction)
                            mean[left:right], std[left:right] = values[:2]
                    extra[column+f'_{direction}{radius}_mean'] = mean
                    extra[column+f'_{direction}{radius}_std'] = std
    return core.join(pd.DataFrame(extra)).astype(np.float32)
