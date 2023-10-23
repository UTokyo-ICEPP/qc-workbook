"""
This module defines classes to reassemble tracks from subtracks.

:py:class:~`TrackRecreater` is very generic and can handle any kind of subtracks.
:py:class:~`TrackRecreaterD` is especially made for handling :py:class:`hepqpr.qallse.qallse.Qallse` outputs, that is
a list of doublets that can potentially contain duplicates and/or conflicting doublets.
"""

import logging
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrackRecreater:
    """
    A generic class for recreating tracks from a list of subtracks. Subtracks are encoded as a list of hit ids.
    This implementation will only work if tracks are disjoint, i.e. a hit can only be part of one track.
    """

    def __init__(self):
        # Dictionary of subtracks indexed by their last hit
        self._ends = dict()
        # Dictionary of subtracks indexed by their first hit
        self._starts = dict()

    @property
    def final_tracks(self):
        """The list of final tracks found on the last :py:meth:~`recreate` call."""
        return list(self._starts.values())

    def recreate(self, subtracks: Union[List, np.ndarray, pd.DataFrame]) -> List:
        """
        Recreate tracks from subtracks. Subtracks can be of arbitrary length.

        Preconditions:
        * a hit belongs only to one track
        * there are no conflicting subtracks
          (example: `1-2-3` is in conflict with `1-4-5` and `0-1-2`, but not with `10-0-1`)

        Postconditions:
        * all hits appearing in subtracks are in a track
        * no more "merge" is possible, i.e.
            - if a track starts at h1, no track ends at h1
            - if a track ends at h2, no track starts at h2

        :param subtracks: a list of subtracks
        :return: a list of tracks
        """

        # make two passes, since in the first pass the merge depends
        # on the order of processing
        merges = 1
        # we are using '+' to concatenate lists, and if the doublets are numpy array, '+' actually
        # adds cell by cell... so ensure we deal with real lists !
        tracks = subtracks.tolist() if hasattr(subtracks, 'tolist') else subtracks
        # iterations stops when no new merge can be performed
        iterations = 0
        while merges > 0:
            self._ends = dict()
            self._starts = dict()
            tracks, merges = self._recreate(tracks)
            iterations += 1
        return tracks

    def _recreate(self, subtracks) -> Tuple[List, bool]:
        # do one pass of the recreation algorithm.
        merges = 0
        for subtrack in subtracks:
            if subtrack[-1] in self._ends or subtrack[0] in self._starts:
                logger.warning(f'conflicting subtrack added {subtrack}')
            if subtrack[-1] in self._starts:
                new_xplet = subtrack[:-1] + self._starts[subtrack[-1]]
                self._remove(self._starts[subtrack[-1]])
                self._add(new_xplet)
                merges += 1
            elif subtrack[0] in self._ends:
                new_xplet = self._ends[subtrack[0]] + subtrack[1:]
                self._remove(self._ends[subtrack[0]])
                self._add(new_xplet)
                merges += 1
            else:
                self._add(subtrack)
        return self.final_tracks, merges

    def _remove(self, subtrack):
        if subtrack[0] in self._starts: del self._starts[subtrack[0]]
        if subtrack[-1] in self._ends: del self._ends[subtrack[-1]]

    def _add(self, subtrack):
        self._starts[subtrack[0]] = subtrack
        self._ends[subtrack[-1]] = subtrack


class TrackRecreaterD(TrackRecreater):
    """
    Track recreator handling doublets only. Compared to :py:class:~`TrackRecreater`, it handles
    duplicates and *tries* to resolve conflicts, if any.

    .. note::

        Conflicts here are defined as doublets either starting or ending at the same hit. Currently,
        conflicts are resolved on the basis of the *track length*: a conflict is added only if it results in
        a longer track. A better implementation would take the physical characteristics of the resulting tracks
        and not only their length.

    """

    def __init__(self):
        super().__init__()

        #: List of conflicts found during the last call to :py:meth:~`recreate`
        self.conflicts = []

    def process_results(self, doublets, resolve_conflicts=True, min_hits_per_track=5) -> Tuple[List, List]:
        """
        Recreate tracks and handle duplicates from a set of doublets.
        :param doublets: a set of doublets, with possible duplicates and conflicts
        :param resolve_conflicts: if set, the conflicts will be processed in a second pass
            and some will be added to the final solution
        :param min_hits_per_track: if > 0, tracks with less than min_hits_per_track hits will be discarded from the results
        :return: a list of tracks, a list of doublets with duplicates removed and conflicts resolved
        """
        from .utils import tracks_to_xplets

        final_tracks = self.recreate(doublets, resolve_conflicts)
        if min_hits_per_track > 0:
            final_tracks = [f for f in final_tracks if len(f) >= min_hits_per_track]
        final_doublets = tracks_to_xplets(final_tracks, x=2)

        return final_tracks, final_doublets

    def recreate(self, doublets: Union[List, np.ndarray, pd.DataFrame], resolve_conflicts=True):
        dblets, conflicts = self.find_conflicts(doublets)
        self.conflicts = conflicts.values.tolist()

        logger.info(f'Found {len(self.conflicts)} conflicting doublets')
        super().recreate(dblets.values)

        if resolve_conflicts and len(self.conflicts) > 0:
            n_resolved = self._resolve_conflicts(self.conflicts)
            logger.info(f'Added {n_resolved} conflicting doublets')

        return self.final_tracks

    @classmethod
    def find_conflicts(cls, doublets: Union[pd.DataFrame, List, np.array]) -> [pd.DataFrame, pd.DataFrame]:
        """
        Remove duplicates and extract conflicts from a list of doublets.

        :param doublets: the doublets
        :return: a dataframe of doublets devoid of duplicates or conflicts and a dataframe with all the conflicts
        """
        df = doublets if isinstance(doublets, pd.DataFrame) else \
            pd.DataFrame(doublets, columns=['start', 'end'])
        # remove exact duplicates
        df.drop_duplicates(inplace=True)
        # find conflicts, i.e. doublets either starting or ending at the same hit
        conflicts = df[df.duplicated('start', keep=False) | df.duplicated('end', keep=False)]
        return df.drop(conflicts.index), conflicts

    def _resolve_conflicts(self, conflicts) -> int:
        resolved = []

        while len(conflicts) > 0:

            best_candidate, best_score, sum_score = 0, 0, 0
            for c in conflicts:
                # compute the score based on the resulting track length if added.
                # This has to be recomputed each time, since adding a doublet to the solution
                # may change the landscape.
                score = 0  # TODO: use another score that looks at the shape of the tracks
                if c[0] in self._ends: score += len(self._ends[c[0]])
                if c[1] in self._starts: score += len(self._starts[c[1]])
                sum_score += score
                if score > best_score:
                    best_score, best_candidate = score, c

            if sum_score == 0:  # all remaining conflicts are lonely doublets. Stop.
                break

            # add the best candidate to the solution
            resolved.append(best_candidate)
            self._recreate([best_candidate])
            # remove conflicts that can no longer be added
            conflicts = [c for c in conflicts if c[0] not in self._starts and c[1] not in self._ends]

        if len(resolved):
            logger.debug(f'Conflicts added: {", ".join(map(str, resolved))}.')
        return len(resolved)
