import math
from typing import Union

import numpy as np
import pandas as pd

from .type_alias import *


# ==========================
# data loading
# ==========================


def load_hits(path: str) -> pd.DataFrame:
    """
    Load hits from a csv file, set the index to the `hit_id` and compute the `r` (radius) column.
    :param path: the path to the file (the suffix `-hits.csv` is optional). Example: `/path/to/dir/event000001000`
    :return: the dataframe of hits
    """
    if not path.endswith('.csv'): path = f'{path}-hits.csv'
    hits = pd.read_csv(path).set_index('hit_id', drop=False)
    hits.index.rename('idx', inplace=True)
    hits['r'] = np.linalg.norm(hits[['x', 'y']].values.T, axis=0)
    return hits


def load_truth(path: str, hits: pd.DataFrame = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List]]:
    """
    Load truth from a csv file, set the index to the `hit_id` and optionally reconstruct the real tracks.
    :param path: the path to the file (the suffix `-truth.csv` is optional). Example: `/path/to/dir/event000001000`
    :param hits: if set, the real tracks are computed and returned as well
    :return: either a dataframe,
        or a dataframe and a list of tracks (represented as a list of hit ids ordered by radius, ascending)
    """
    if not path.endswith('.csv'): path = f'{path}-truth.csv'
    truth = pd.read_csv(path).set_index('hit_id', drop=False)
    truth.index.rename('idx', inplace=True)
    if hits is not None:
        return truth, recreate_tracks(hits, truth)
    else:
        return truth


# ==========================
# segments/tracks conversion
# ==========================

def track_to_xplets(track: TXplet, x=2) -> List[TXplet]:
    """
    Convert a track to a list of xplets. precondition: x <= length of the track.
    :param x: the size of the resulting xplets. Default to 2 (doublet)
    :return: a list of xplets of size x
    """
    return [track[n:n + x] for n in range(len(track) - x + 1)]


def tracks_to_xplets(tracks: List[TXplet], x=2) -> List[TXplet]:
    """Convert a list of tracks to a list of xplets (flattened). """
    return [track[n:n + x] for track in tracks for n in range(len(track) - x + 1)]


def recreate_tracks(hits: pd.DataFrame, df: pd.DataFrame) -> List[TXplet]:
    """
    Recreate tracks from the truth.

    :param hits: the dataframe of hits
    :param df: a dataframe indexed by hit_id with the column `particle_id`
    :return: a list of tracks, each track encoded as a list of hit_id ordered by increasing radius
    """
    tracks = []
    if 'r' not in hits: hits['r'] = np.linalg.norm(hits[['x', 'y']].values.T)
    for particle_id, df in df.groupby('particle_id'):
        if particle_id == 0: continue
        track = hits.loc[df.hit_id.values].sort_values('r').hit_id.values
        tracks.append(track)
    return tracks


def truth_to_xplets(hits, df, x=2) -> List[TXplet]:
    """
    Generate the list of real xplets from the truth.
    :param hits: the dataframe of hits
    :param df: a dataframe indexed by hit_id with the column `particle_id`
    :return: a list of xplets. xplets are encoded as a list of hit_id ordered by increasing radius.
    """
    return tracks_to_xplets(recreate_tracks(hits, df), x=x)


# ==========================
# math
# ==========================

def angle_between(v1, v2):
    """Compute the angle between two vectors."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_diff(angle1, angle2):
    """Compute the absolute difference between to angles in radiant. The result is between [0, π]. """
    delta_angle = abs(angle2 - angle1)
    return delta_angle if delta_angle <= np.pi else 2 * np.pi - delta_angle


def curvature(p0, p1, p2):
    """Compute the `Menger curvature <https://en.wikipedia.org/wiki/Menger_curvature>`_ between three
    3D points. """
    # cf. Menger: curvature = 1/R = 4*triangleArea/(sideLength1*sideLength2*sideLength3)
    # Adapted from https://stackoverflow.com/questions/41144224/calculate-curvature-for-3-points-x-y
    (x0, y0), (x1, y1), (x2, y2) = (p0, p1, p2)
    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x0, y2 - y0
    twice_area = dx1 * dy2 - dy1 * dx2
    len0 = math.hypot(dx1, dy1)  # p0.distance(p1);
    len1 = math.hypot(x2 - x1, y2 - y1)  # len1 = p1.distance(p2);
    len2 = math.hypot(dx2, dy2)  # p2.distance(p0);
    return 2 * twice_area / (len0 * len1 * len2)


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).

    Taken from: https://stackoverflow.com/a/50974391
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        # TODO: this is probably not smart as it would crash the code downstream.
        # if the points are aligned, compute the distance between the line
        # and the origin instead
        return (None, np.inf)

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return ((cx, cy), radius)


# ==========================
# pandas/numpy utils
# ==========================

def intersect_rows(a, b):
    """Return the intersection between a and b, row-wise."""
    if isinstance(a, np.ndarray): a = a.tolist()
    if isinstance(b, np.ndarray): b = b.tolist()
    # return rows that are common to one another
    tmp = np.array(a + b)
    res, cnts = np.unique(tmp, return_counts=True, axis=0)
    return res[cnts > 1]


def diff_rows(a, b):
    """
    Compute both intersection and difference between a and b. A and b must be matrices with the same number of
    columns.
    :param a: an matrix-like object
    :param b: another matrix-like object with the same number of columns as a
    :return: three matrices, as a python List of List:
        * rows only in a,
        * rows only in b,
        * rows in a and b
    """
    if isinstance(a, np.ndarray): a = a.tolist()
    if isinstance(b, np.ndarray): b = b.tolist()
    # return rows that are common to one another
    tmp = np.array(a + b)
    res, idx, cnt = np.unique(tmp, return_counts=True, return_index=True, axis=0)
    return (
        res[(cnt == 1) & (idx < len(a)), :].tolist(),  # only in a
        res[(cnt == 1) & (idx >= len(a)), :].tolist(),  # only in b
        res[cnt > 1].tolist()  # in a and b
    )


def pd_read_csv_array(csv_rows: List[str], **kwargs):
    """Create a pandas dataframe from a list of rows in csv format. The first row is the header. Additional parameters
    will be passed to `pd.read_csv`."""
    import pandas as pd
    # create a pseudo-file with the results as CSV
    from io import StringIO
    csv = StringIO()
    csv.write('\n'.join(csv_rows))
    csv.seek(0)
    # get the results into a pandas dataframe
    return pd.read_csv(csv, **kwargs)


# ==========================
# other
# ==========================

def merge_dicts(default, overrides):
    """
    Merge ``override`` into ``default`` recursively.
    As the names suggest, if an entry is defined in both, the value in ``overrides`` takes precedence.
    """
    import collections
    for k, v in overrides.items():
        if isinstance(v, collections.Mapping):
            default[k] = merge_dicts(default.get(k, {}), v)
        else:
            default[k] = v
    return default


def transform_qubo(Q, bias_weight, conflict_strength, bw_marker=10, cs_marker=20):
    Q2 = dict()
    for k, v in Q.items():
        if v == bw_marker:
            Q2[k] = bias_weight
        elif v == cs_marker:
            Q2[k] = conflict_strength
        else:
            Q2[k] = v
    return Q2
