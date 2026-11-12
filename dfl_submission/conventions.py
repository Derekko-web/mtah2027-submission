"""Re-express external possession targets under the challenge's controlled-possession definition."""

import numpy as np

from evaluation import possession_metrics as pm


def runs(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [[int(start), int(stop), labels[start]] for start, stop in zip(edges[:-1], edges[1:])]


def live_segments(labels):
    live = np.isin(labels, (pm.H, pm.A)).astype(np.int8)
    edges = np.diff(np.r_[0, live, 0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def collapse_brief_changes(labels, seconds, fps=25):
    """Merge each A-B-A whose interior B lasts under `seconds` into A, shortest first.

    Work happens inside maximal live segments, so dead balls, abstentions and the possessor at
    a restart are never changed.
    """
    labels = np.asarray(labels)
    result = labels.copy()
    limit = seconds * fps
    for start, stop in live_segments(labels):
        spells = runs(labels[start:stop])
        while True:
            brief = [i for i in range(1, len(spells) - 1)
                     if spells[i - 1][2] == spells[i + 1][2] and spells[i][1] - spells[i][0] < limit]
            if not brief:
                break
            i = min(brief, key=lambda j: (spells[j][1] - spells[j][0], j))
            spells[i - 1][1] = spells[i + 1][1]
            del spells[i:i + 2]
        for first, last, label in spells:
            result[start + first:start + last] = label
    return result


def remove_brief_stoppages(labels, frames=25):
    """Remove every stoppage shorter than `frames` with a team on both sides (DECISIONS.md D25).

    The same team on both sides takes the stoppage's frames; otherwise the earlier team takes the
    first floor(n / 2) of its n frames and the later team the rest. Stoppages at the edge of the
    sequence or next to an X frame, and stoppages of `frames` or more, are untouched.
    """
    labels = np.asarray(labels)
    result = labels.copy()
    spells = runs(labels)
    for i in range(1, len(spells) - 1):
        start, stop, label = spells[i]
        before, after = spells[i - 1][2], spells[i + 1][2]
        if label != pm.D or stop - start >= frames or before not in (pm.H, pm.A) or after not in (pm.H, pm.A):
            continue
        middle = start + (stop - start) // 2
        result[start:middle] = before
        result[middle:stop] = after
    return result
