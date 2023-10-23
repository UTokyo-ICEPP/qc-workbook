"""
This module contains the definition of all the data structures used by our model, :py:class:`hepqpr.qallse.qallse.Qallse`
as well as some useful type alias used throughout the project.
"""

import math
from typing import Set, Iterable

import numpy as np

from .type_alias import *
from .utils import curvature, angle_diff


class Volayer:
    """
    Support the encoding of a hit's `volume_id` and `layer_id` into one single number that can be used for
    ordering and distance calculation. Note that it only works for TrackML data limited to the barrel region.
    """

    #: Define the mapping of `volume_id` and `layer_id` into one number (the index in the list)
    ordering = [(8, 2), (8, 4), (8, 6), (8, 8), (13, 2), (13, 4), (13, 6), (13, 8), (17, 2), (17, 4)]

    @classmethod
    def get_index(cls, volayer: Tuple[int, int]) -> int:
        """Convert a couple `volume_id`, `layer_id` into a number (see :py:attr:`~ordering`)."""
        return cls.ordering.index(tuple(volayer))

    @classmethod
    def difference(cls, volayer1, volayer2) -> int:
        """Return the distance between two volayers."""
        return cls.ordering.index(volayer2) - cls.ordering.index(volayer1)


class Xplet:
    """
    Base class for doublets, triplets and quadruplets.
    An xplet is an ordered list of hits (ordered by radius).

    It contains lists of inner and outer xplets (with one more hit) and sets of "kept" inner and outer xplets,
    i.e. xplets actually used when generating the qubo. Those lists and sets are populated during model building
    (see :py:meth:`hepqpr.qallse.Qallse.build_model`).
    """

    def __init__(self, hits, inout_cls=None):
        """
        Create an xplet. Preconditions:
        * hits are all different
        * hits are ordered in increasing radius in the X-Y plane
        """
        self.hits = hits

        if inout_cls is not None:
            self.inner: List[inout_cls] = []
            self.outer: List[inout_cls] = []
            self.inner_kept: Set[inout_cls] = set()
            self.outer_kept: Set[inout_cls] = set()

    def hit_ids(self) -> TXplet:
        """Convert this xplet into a list of hit ids."""
        return [h.hit_id for h in self.hits]

    @classmethod
    def name_to_hit_ids(cls, str):
        """Convert a string representation of an xplet into a list of hit ids (see :py:meth:~`__str__`)."""
        return [int(h) for h in str.split('_')]

    @classmethod
    def hit_ids_to_name(cls, hits):
        """Inverse of :py:meth:~`name_to_hit_ids`."""
        return '_'.join(map(str, hits))

    def __str__(self):
        """Return a string made of hit ids joined by an underscore. This can be used in the QUBO as an identifier."""
        return '_'.join(map(str, self.hits))

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        d = dict(name=str(self), hits=self.hit_ids())
        for k, v in self.__dict__.items():
            if k == 'hits' or k.startswith('inner') or k.startswith('outer'): continue
            if isinstance(v, Xplet): v = str(v)
            d[k] = v
        return d


class Hit(Xplet):
    """One hit."""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        super().__init__([self], Doublet)

        #: The hit id
        self.hit_id: int = int(self.hit_id)
        #: The volayer
        self.volayer: int = Volayer.get_index((int(self.volume_id), int(self.layer_id)))

        #: The coordinates in the X-Y plane, i.e. `(x,y)`
        self.coord_2d: Tuple[float, float] = np.array([self.x, self.y])
        #: The coordinates, i.e. `(x,y,z)`
        self.coord_3d: Tuple[float, float, float] = np.array([self.x, self.y, self.z])

        # TODO: remove if QallseCs is discarded from the project
        # test: second order conflicts
        self.inner_tplets: List[Triplet] = []
        self.outer_tplets: List[Triplet] = []

    def __str__(self):
        return str(self.hit_id)  # to avoid recursion


class Doublet(Xplet):
    """A doublet is composed of two hits."""

    def __init__(self, hit_start: Hit, hit_end: Hit):
        """
        Create a doublet.
        """
        assert hit_start != hit_end
        assert hit_start.r <= hit_end.r

        super().__init__([hit_start, hit_end], Triplet)
        #: The hits composing this doublet
        self.h1, self.h2 = self.hits
        #: The delta r of the doublet
        self.dr = hit_end.r - hit_start.r
        #: The delta z of the doublet
        self.dz = hit_end.z - hit_start.z

        #: The angle in the R-Z plane between this doublet and the R axis.
        self.rz_angle = math.atan2(self.dz, self.dr)
        #: The 2D vector of this doublet in the X-Y plane, i.e. `(∆x,∆y)`
        self.coord_2d = hit_end.coord_2d - hit_start.coord_2d
        #: The 3D vector of this doublet, i.e. `(∆x,∆y,∆z)`
        self.coord_3d = hit_end.coord_3d - hit_start.coord_3d


class Triplet(Xplet):
    """A triplet is composed of two doublets, where the first ends at the start of the other."""

    def __init__(self, d1: Doublet, d2: Doublet):
        """
        Create a triplet. Preconditions:
        * `d1` ends where `d2` starts: `d1.hits[-1] == d2.hits[0]`
        """
        super().__init__([d1.h1, d2.h1, d2.h2], Quadruplet)
        assert d1.hits[-1] == d2.hits[0]
        assert d1.h1.r < d2.h2.r

        self.d1: Doublet = d1
        self.d2: Doublet = d2

        #: Radius of curvature, see `Menger curvature <https://en.wikipedia.org/wiki/Menger_curvature>`_.
        self.curvature = curvature(*[h.coord_2d for h in self.hits])
        #: Difference between the doublet's rz angles (see :py:attr:~`Doublet.rz_angle`)
        self.drz = angle_diff(d1.rz_angle, d2.rz_angle)
        #: Sign of the `drz` difference
        self.drz_sign = 1 if abs(d1.rz_angle + self.drz - d2.rz_angle) < 1e-3 else -1
        #: QUBO weight, assigned later
        self.weight = .0

    def doublets(self) -> List[Doublet]:
        """Return the ordered list of doublets composing this triplet."""
        return [self.d1, self.d2]


class Quadruplet(Xplet):
    """A quadruplet is composed of two triplets having two hits (or one doublet) in common."""

    def __init__(self, t1: Triplet, t2: Triplet):
        """
        Create a quadruplet. Preconditions:
        * `t1` and `t2` share two hits/one doublet: `t1.hits[-2:] == t2.hits[:2]` and `t1.d2 == t2.d1`
        """
        assert t1.d2 == t2.d1
        super().__init__(t1.hits + [h for h in t2.hits if h not in t1.hits])

        self.t1: Triplet = t1
        self.t2: Triplet = t2

        #: Absolute difference between the two triplets' curvatures
        self.delta_curvature = abs(self.t1.curvature - self.t2.curvature)
        #: Number of layers this quadruplet spans across
        #: If no layer skip, this is equal to `len(self.hits) - 1`.
        self.volayer_span = self.hits[-1].volayer - self.hits[0].volayer
        #: QUBO coupling strength between the two triplets. Should be negative to encourage
        #: the two triplets to be kept together.
        self.strength = .0

    def doublets(self) -> List[Doublet]:
        """Return the ordered list of doublets composing this triplet."""
        return self.t1.doublets() + [self.t2.d2]
