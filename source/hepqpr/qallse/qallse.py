import pandas as pd

from .data_structures import *
from .qallse_base import ConfigBase, QallseBase
from .utils import pd_read_csv_array


class Config(ConfigBase):
    cheat = False

    # === Hard cut

    #: Doublets can miss at most (max_layer_span - 1) layers.
    #: Note that triplets and quadruplets also have this limitation, i.e.
    #: any xplet will at most miss (max_layer_span - 1) layers
    max_layer_span = 2

    #: Maximum radius of curvature for a triplet. The curvature is computed using
    #: the *Mengel curvature*.
    tplet_max_curv = 5E-3
    #: Maximum (absolute) difference between the angles in the R-Z plane of the two doublets forming
    #: the triplet. The angles are defined as arctan(dz/dr).
    tplet_max_drz = 0.2

    #: Maximum difference between the radius of curvature of the two triplets forming the quadruplet.
    qplet_max_dcurv = 5E-4
    #: Maximum strength of a quadruplet. This cut is really efficient, but the actual value depends
    #: highly on the strength function parameters (see below)
    qplet_max_strength = -0.2

    #: Linear bias weight associated to triplets in the QUBO.
    qubo_bias_weight = 0
    #: Quadratic coupling strength associated to two conflicting triplets in the QUBO.
    #: Set it to 1 (other things being equal) to avoid conflicts.
    qubo_conflict_strength = 1

    # === strength computation

    #: Factor of the numerator in the strength formula. Should be negative.
    num_multiplier = -1
    #: Ponderation between the curvature (X-Y plane) and the delta angle (R-Z plane) in the numerator.
    #: Should be a percentage (0 <= `xy_relative_strength` <= 1).
    xy_relative_strength = 0.5
    #: Exponent of the curvature (X-Y plane) in the strength formula. Should be >= 0.
    xy_power = 1
    #: Exponent of the delta angle (R-Z plane) in the strength formula. Should be >= 0.
    rz_power = 1
    #: Exponent of the "layer miss" in the strength formula (denominator). Should be >= 0.
    volayer_power = 2
    #: Clipping bounds of the strength. If defined, strength values outside those bounds will take the
    #: value of the bound.
    strength_bounds = None


class Config1GeV(Config):
    tplet_max_curv = 8E-4  # (vs 5E-3)
    tplet_max_drz = 0.1  # (vs0.2)
    qplet_max_dcurv = 1E-4  # (vs4E-4)


class Qallse(QallseBase):
    config: Config  # for proper autocompletion in PyCharm

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hard_cuts_stats = ['type,hid,reason,arg1,arg2']

    def _get_base_config(self):
        return Config1GeV() # TODO

    def get_build_stats(self) -> pd.DataFrame:
        """Return a dataframe, each row corresponding to a real xplet that has been dropped during preprocessing."""
        assert len(self.hard_cuts_stats) >= 1  # ensure it has headers
        return pd_read_csv_array(self.hard_cuts_stats)

    def build_model(self, *args, **kwargs):
        super().build_model(*args, **kwargs)
        # add stats information to the logs
        self.log_build_stats()

    def log_build_stats(self):
        """ Log information about real doublets/triplets/quadruplets dropped during model building"""
        stats = self.get_build_stats()
        if stats.shape[0] > 0:
            self.logger.info(f'Dropped {len(stats)} real structures during preprocessing')
            if len(stats) <= 10:
                self.logger.debug('\n' + stats.to_string())
            details = 'Dropped type:reason:count => '
            for (typ, reason), df in stats.groupby(['type', 'reason']):
                details += f'{typ}:{reason}:{len(df)} '
            self.logger.info(details)

    # --------------- early cuts

    def _is_invalid_doublet(self, dblet: Doublet) -> bool:
        # Apply hard cuts on doublets.
        # Currently, doublets are only dropped if they miss more than one layer.

        v1, v2 = dblet.h1.volayer, dblet.h2.volayer
        ret = v1 >= v2 or v2 > v1 + self.config.max_layer_span
        if ret and self.dataw.is_real_doublet(dblet.hit_ids()) == XpletType.REAL:
            self.hard_cuts_stats.append(f'dblet,{dblet},volayer,{v1},{v2}')
            return not self.config.cheat
        return ret

    def _is_invalid_triplet(self, tplet: Triplet) -> bool:
        # Apply hard cuts on triplets.
        # Currently, we look at three information:
        # * the number of layer miss
        # * the radius of the curvature formed by the three hits (cut on GeV)
        # * how well are the two doublets aligned in the R-Z plane

        is_real = self.dataw.is_real_xplet(tplet.hit_ids()) == XpletType.REAL

        # layer skips
        volayer_skip = tplet.hits[-1].volayer - tplet.hits[0].volayer
        if volayer_skip > self.config.max_layer_span + 1:
            if is_real:
                self.hard_cuts_stats.append(f'tplet,{tplet},volayer,{volayer_skip},')
                return not self.config.cheat
            return True
        # radius of curvature formed by the three hits
        if abs(tplet.curvature) > self.config.tplet_max_curv:
            if is_real:
                self.hard_cuts_stats.append(f'tplet,{tplet},curv,{tplet.curvature},')
                return not self.config.cheat
            return True
        # angle between the two doublets in the rz plane
        if tplet.drz > self.config.tplet_max_drz:
            if is_real:
                self.hard_cuts_stats.append(f'tplet,{tplet},drz,{tplet.drz},')
                return not self.config.cheat
            return True
        return False

    def _is_invalid_quadruplet(self, qplet: Quadruplet) -> bool:
        # Apply hard cuts on quadruplets.
        # Currently, we discard directly any potential quadruplet between triplets that don't have
        # a very similar curvature in the X-Y plane. Then, we compute the coupling strength (combining
        # layer miss, R-Z plane delta angles and curvature) and apply a cut on it.
        is_real = self.dataw.is_real_xplet(qplet.hit_ids()) == XpletType.REAL

        # delta delta curvature between the two triplets
        ret = qplet.delta_curvature > self.config.qplet_max_dcurv
        if ret and is_real:
            self.hard_cuts_stats.append(f'qplet,{qplet},dcurv,{qplet.delta_curvature},')
            return not self.config.cheat

        # strength of the quadruplet
        qplet.strength = self._compute_strength(qplet)
        ret = qplet.strength > self.config.qplet_max_strength
        if ret and is_real:
            self.hard_cuts_stats.append(f'qplet,{qplet},strength,{qplet.strength},')
            return not self.config.cheat
        return ret

    # --------------- qubo weights

    def _compute_weight(self, tplet: Triplet) -> float:
        # Just return a constant for now.
        # In the future, it would be interesting to try to measure a-priori how interesting a triplet is
        # (using for example the number of quadruplets it is part of) and encode this information into a
        # variable weight.
        return self.config.qubo_bias_weight

    def _compute_strength(self, qplet: Quadruplet) -> float:
        # Combine information about the layer miss, the alignment in the R-Z plane and the curvature in the X-Y plane.
        # The strength is negative, its range depending on the configuration (default: 1 >= strength >= max_strength)

        if qplet.strength != 0:  # avoid computing twice
            return qplet.strength

        # normalised difference of curvature between the two triplets
        xy_strength = 1 - ((qplet.delta_curvature / self.config.qplet_max_dcurv) ** self.config.xy_power)

        # normalised [maximum] angle in the R-Z plane
        max_drz = max(qplet.t1.drz, qplet.t2.drz)
        rz_strength = 1 - ((max_drz / self.config.tplet_max_drz) ** self.config.rz_power)

        # numerator: combine both X-Y and R-Z plane information
        numerator = self.config.num_multiplier * (
                self.config.xy_relative_strength * xy_strength +
                (1 - self.config.xy_relative_strength) * rz_strength
        )

        # denominator: shrink the strength proportional to the number of layer miss
        exceeding_volayer_span = qplet.volayer_span - len(qplet.hits) + 1
        denominator = (1 + exceeding_volayer_span) ** self.config.volayer_power

        strength = numerator / denominator

        # clip the strength if needed
        if self.config.strength_bounds is not None:
            strength = np.clip(strength, *self.config.strength_bounds)

        return strength

    def _compute_conflict_strength(self, t1: Triplet, t2: Triplet) -> float:
        # Just return a constant for now.
        # Careful: if too low, the number of remaining conflicts in the QUBO solution will explode.
        # If too high, qbsolv can behave strangely: the execution time augments significantly while the
        # scores drop slowly.
        return self.config.qubo_conflict_strength
