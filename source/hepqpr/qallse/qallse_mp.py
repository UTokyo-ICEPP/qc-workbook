import time

from .data_structures import *
from .qallse import Qallse, Config1GeV
from .qallse_base import QallseBase


class MpConfig(Config1GeV):
    #: To be kept, a quadruplet needs to be part of at least one chain of
    #: min_qplet_path or more (including itself). Hence, setting this parameter
    #: to one has no effect, and setting it to 3 makes tracks of < 6 hits
    #: likely to not appear in the QUBO.
    min_qplet_path = 2


class QallseMp(Qallse):
    """ Same as Qallse, but also applies a *max path* cut in order to discard isolated quadruplets."""
    config: MpConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_base_config(self):
        return MpConfig()

    def _find_max_path(self, qplet: Quadruplet, direction=0):
        # Use recursion to compute the max path, i.e. the longest chain of qplets
        # this qplet is part of. The return value is always >= 1.
        # Use direction = 0 to initiate the recursion, then direction = -1 is
        # "inner", direction = 1 is "outer".
        inner_length = 0
        outer_length = 0

        if direction <= 0 and len(qplet.t1.inner):
            ls = (self._find_max_path(q, direction=-1) for q in qplet.t1.inner)
            inner_length = max(ls) if len(qplet.t1.inner) > 1 else next(ls)
        if direction >= 0 and len(qplet.t2.outer):
            ls = (self._find_max_path(q, direction=1) for q in qplet.t2.outer)
            outer_length = max(ls) if len(qplet.t2.outer) > 1 else next(ls)

        return 1 + inner_length + outer_length


    def build_model(self, *args, **kwargs):
        # create the model as usual
        QallseBase.build_model(self, *args, **kwargs)

        # filter quadruplets
        start_time = time.process_time()
        dropped = self._filter_quadruplets()
        exec_time = time.process_time() - start_time

        self.logger.info(
            f'MaxPath done in {exec_time:.2f}s. '
            f'doublets: {len(self.qubo_doublets)}, triplets: {len(self.qubo_triplets)}, ' +
            f'quadruplets: {len(self.quadruplets)} (dropped {dropped})')
        self.log_build_stats()

    def _filter_quadruplets(self) -> int:
        # Here, we compute the max path for each quadruplet and keep only the ones that
        # belong to long tracks (see :py:attr:`~MpConfig.min_qplet_path`).
        # Only triplets part of the kept quadruplet will appear in the QUBO
        # (see :py:meth:`hepqr.qallse.qallse_base.QallseBase._register_qubo_quadruplet`)
        filtered_qplets = []

        for qplet in self.quadruplets:
            qplet.max_path = self._find_max_path(qplet)
            if qplet.max_path >= self.config.min_qplet_path:
                # keep qplet and register the structures it is made of
                filtered_qplets.append(qplet)
                self._register_qubo_quadruplet(qplet)
            elif self.dataw.is_real_xplet(qplet.hit_ids()) == XpletType.REAL:
                # we are dropping a real qplet here, log it !
                self.hard_cuts_stats.append(f'qplet,{qplet},max_path,{qplet.max_path},')

        dropped = len(self.quadruplets) - len(filtered_qplets)
        self.quadruplets = filtered_qplets

        return dropped

    def _create_quadruplets(self, register_qubo=False):
        # don't register quadruplet now, do it in a second pass
        super()._create_quadruplets(False)
