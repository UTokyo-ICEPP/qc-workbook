from typing import List, Optional, Union

import numpy as np
import pandas as pd
from trackml.score import score_event

from .type_alias import TQubo, TDimodSample, TXplet, XpletType, TDoublet
from .utils import truth_to_xplets, track_to_xplets, diff_rows


class DataWrapper:
    """
    Wraps a hits and a truth file and exposes useful functions to compute scores, check xplet validity and more.
    """

    def __init__(self, hits: pd.DataFrame, truth: pd.DataFrame):
        """
        Create a wrapper. Hits and Truth should match the TrackML challenge schema.
        See `TrackML data <https://www.kaggle.com/c/trackml-particle-identification/data>`_ on Kaggle for more info.

        Note that indexes and all will be handled here, so you can just use `pd.read_csv` to load the
        files.
        """
        self.hits = hits
        self.truth = truth

        # add proper indexing
        for df in [self.hits, self.truth]:
            df['idx'] = df.hit_id.values
            df.set_index('idx', inplace=True)

        # add radius information
        hits['r'] = np.linalg.norm(hits[['x', 'y']].values.T, axis=0)

        # keep a lookup of real doublets: '{hit_id_1}_{hit_id_2}' -> [hit_id_1, hit_id_2]
        df = hits.join(truth, lsuffix='_')
        self._doublets = truth_to_xplets(hits, df[df.weight > 0], x=2)
        self._unfocused = truth_to_xplets(hits, df[df.weight == 0], x=2)

        self._lookup = dict(
            [(self._get_dkey(*d), XpletType.REAL) for d in self._doublets] +
            [(self._get_dkey(*d), XpletType.REAL_UNFOCUSED) for d in self._unfocused]
        )

    def _get_dkey(self, h1, h2):
        return f'{h1}_{h2}'

    def get_unfocused_doublets(self) -> List[TDoublet]:
        return self._unfocused

    def get_real_doublets(self, with_unfocused=False) -> List[TDoublet]:
        """Return the list of real doublets"""
        if with_unfocused:
            return self._doublets + self._unfocused
        return self._doublets

    # ==== doublets and subtrack checking

    def is_real_doublet(self, doublet: TDoublet) -> XpletType:
        """Test whether a doublet is real, i.e. part of a real track."""
        key = self._get_dkey(*doublet)
        return self._lookup.get(key, XpletType.FAKE)

    def is_real_xplet(self, xplet: TXplet) -> XpletType:
        """Test whether an xplet is real, i.e. a sub-track of a real track."""
        doublets = track_to_xplets(xplet, x=2)
        if len(doublets) == 0:
            raise Exception(f'Got a subtrack with no doublets in it "{xplet}"')

        xplet_type = set(self.is_real_doublet(s) for s in doublets)
        return XpletType.FAKE if len(xplet_type) > 1 else xplet_type.pop()

    # =============== QUBO and energy checking

    def sample_qubo(self, Q: TQubo) -> TDimodSample:
        """
        Compute the ideal solution for a given QUBO. Here, ideal means correct, but I doesn't guarantee that
        the energy is minimal.
        """
        sample = dict()
        for (k1, k2), v in Q.items():
            if k1 == k2:
                subtrack = list(map(int, k1.split('_')))
                sample[k1] = int(self.is_real_xplet(subtrack) != XpletType.FAKE)
        return sample

    def compute_energy(self, Q: TQubo, sample: Optional[TDimodSample] = None) -> float:
        """Compute the energy of a given sample. If sample is None, the ideal sample is used (see :py:meth:~`sample_qubo`). """
        if sample is None:
            sample = self.sample_qubo(Q)
        en = 0
        for (k1, k2), v in Q.items():
            if sample[k1] != 0 and sample[k2] != 0:
                en += v
        return en

    # =============== scoring

    def get_score_numbers(self, doublets: Union[List, np.array, pd.DataFrame]) -> [float, float, float]:
        """
        :param doublets: a set of doublets
        :return: the number of real, fake and missing doublets
        """
        if isinstance(doublets, pd.DataFrame): doublets = doublets.values
        doublets_found, _, unfocused_found = diff_rows(doublets, self._unfocused)
        missing, fakes, real = diff_rows(self._doublets, doublets_found)
        return len(real), len(fakes), len(missing)

    def compute_score(self, doublets: Union[List, np.array, pd.DataFrame]) -> [float, float, List[List]]:
        """
        Precision and recall are defined as follow:
        * precision (purity): how many doublets are correct ? `len(real ∈ doublets) / len(doublets)`
        * recall (efficiency): how well does the solution covers the truth ? `len(real ∈ doublets) / len(truth)`

        :param doublets: a set of doublets
        :return: the precision, the recall and the list of missing doublets. p and r are between 0 and 1.
        """
        if isinstance(doublets, pd.DataFrame): doublets = doublets.values
        doublets_found, _, unfocused_found = diff_rows(doublets, self._unfocused)
        missing, fakes, real = diff_rows(self._doublets, doublets_found)
        return len(real) / len(doublets_found), \
               len(real) / len(self._doublets), \
               missing

    def add_missing_doublets(self, doublets: Union[np.array, pd.DataFrame]) -> pd.DataFrame:
        """
        :param doublets: a list of doublets
        :return: a list of doublets with 100% recall
        """
        if isinstance(doublets, pd.DataFrame):
            doublets = doublets.values

        ip, ir, missing = self.compute_score(doublets)
        print(f'got {len(doublets)} doublets.')
        print(f'  Input precision (%): {ip * 100:.4f}, recall (%): {ir * 100:.4f}')

        if len(missing) == 0:
            # nothing to do
            return doublets
        else:
            ret = pd.DataFrame(np.vstack((doublets, missing)), columns=['start', 'end'])
            p, _, _ = self.compute_score(ret.values)
            print(f'    New precision (%): {p * 100:.4f}')
            return ret

    def compute_trackml_score(self, final_tracks: List[TXplet], submission=None) -> float:
        """
        :param final_tracks: a list of xplets representing tracks
        :param submission: (optional) a TrackML submission, see :py:meth:~`create_submission`
        :return: the trackml score (between 0 and 1)
        """
        if submission is None:
            submission = self.create_submission(final_tracks)
        return score_event(self.truth, submission)

    def create_submission(self, tracks: List[TXplet], event_id=1000) -> pd.DataFrame:
        """Encode a solution into a dataframe following the structure of a trackml submission."""
        hit_ids = self.hits.hit_id.values
        n_rows = len(hit_ids)
        sub_data = np.column_stack(([event_id] * n_rows, hit_ids, np.zeros(n_rows)))
        submission = pd.DataFrame(
            data=sub_data, columns=["event_id", "hit_id", "track_id"], index=hit_ids, dtype=int)
        for idx, track in enumerate(tracks):
            submission.loc[track, 'track_id'] = idx + 1
        return submission

    # =============== class utils

    @classmethod
    def from_path(cls, path):
        """
        Create a DataWrapper by reading the hits and the truth from a path.
        :path: the path + event id, in the format `/path/to/directory/eventXXXXX`
        """
        path = path.replace('-hits.csv', '')
        return cls(hits=pd.read_csv(path + '-hits.csv'), truth=pd.read_csv(path + '-truth.csv'))
