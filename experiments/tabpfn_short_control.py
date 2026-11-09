"""Protect independently supported short control signals in a TabPFN hybrid."""
import numpy as np


def restore_supported_islands(reference, alternative, support, blocks):
    """Retain short reference control when independent contacts support it.

    All eligibility reads the unchanged inputs. The rule uses no truth labels,
    frame-specific exceptions, protocol names or fitted thresholds.
    """
    reference, alternative = np.asarray(reference), np.asarray(alternative)
    support = np.asarray(support)
    if (reference.ndim != 1 or alternative.shape != reference.shape
            or support.shape != (len(reference), 2) or not np.isfinite(support).all()
            or not np.isin(reference, ["H", "A", "D", "X"]).all()
            or not np.isin(alternative, ["H", "A", "D", "X"]).all()):
        raise ValueError("Aligned reference, alternative and two-team support are required")
    for label in ("D", "X"):
        if not np.array_equal(reference == label, alternative == label):
            raise ValueError("The upstream D/X masks must already be identical")
    result, events = alternative.copy(), []
    covered = np.zeros(len(reference), bool)
    for left, right in blocks:
        if not 0 <= left < right <= len(reference) or covered[left:right].any():
            raise ValueError("Invalid or overlapping decode blocks")
        covered[left:right] = True
        local = reference[left:right]
        edges = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1, len(local)]
        for begin, finish in zip(edges[:-1], edges[1:]):
            team = local[begin]
            if team not in ("H", "A") or not 8 <= finish-begin <= 250:
                continue
            start, end = left+int(begin), left+int(finish)
            if np.mean(alternative[start:end] == team) >= .5:
                continue
            column = 0 if team == "H" else 1
            if np.count_nonzero(support[start:end, column]) < 3:
                continue
            other = "A" if team == "H" else "H"
            flanks = [v for v in (reference[max(left,start-13):start],
                                  reference[end:min(right,end+13)]) if len(v)]
            if not flanks or any(np.mean(v == other) <= .5 for v in flanks):
                continue
            result[start:end] = team
            events.append(dict(start=start, end_exclusive=end, team=str(team),
                supported_frames=int(np.count_nonzero(support[start:end,column])),
                prior_agreement=float(np.mean(alternative[start:end] == team))))
    return result, events
