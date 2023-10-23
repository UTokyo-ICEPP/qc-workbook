"""
Utilities to dump the built model to disk. Useful for benchmarks, debugging, etc.

Example usage:

.. code::

    from hepqpr.qallse import QallseD0, DataWrapper, dumper

    # build model
    model = QallseD0(DataWrapper.from_path('/data/path/eventx'))
    model.build_model()

    # EITHER create and dump QUBO+xplets
    dumper.dump_model(model)

    # OR do each step separately. Here, ensure you call dump_qubo first !
    Q = dumper.dump_qubo(model, c_marker='conflict', w_marker='linear')
    xplets = dumper.xplets_to_serializable_dict(model)
    dumper.dump_xplets(xplets, format='json') # use json, so you can view the actual format

"""
import json
import pickle
from contextlib import contextmanager
from json import JSONEncoder
from os.path import join as path_join

from .data_structures import *
from .qallse_base import QallseBase


# ---- custom Json encoder to handle special types

class _XpletsJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Xplet):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# ---- default argument values

_default_opath = '.'
_default_prefix = ''


@contextmanager
def use_markers(model, w_marker=None, c_marker='c'):
    """
    Temporarily modifies the _compute_* methods of the model to insert placeholder
    values instead of coefficients in the QUBO. Note that the original methods are still
    called, because they can have side-effects (example: computing new xplet properties).

    Usage:

        with use_markers(model) as altered_model:
            Q = altered_model.to_qubo()
    :param model: an implementation of :py:class:`hepqpr.qallse.QallseBase`
    :param w_marker: the placeholder used for linear weights. Set it to None to use the original weight.
    :param c_marker: the placeholder used for conflict strengths. Set it to None to use the original weight. Default to 'c'.
    :return: an altered model
    """
    old_cw, old_cc = None, None
    # change the model functions during qubo
    if w_marker is not None:
        old_cw = model._compute_weight

        def new_cv(*args, **kwargs):
            old_cw(*args, **kwargs)
            return w_marker

        model._compute_weight = new_cv

    if c_marker is not None:
        old_cc = model._compute_conflict_strength

        def new_cc(*args, **kwargs):
            old_cc(*args, **kwargs)
            return c_marker

        model._compute_conflict_strength = new_cc

    yield model

    if old_cw is not None: model._compute_weight = old_cw
    if old_cc is not None: model._compute_conflict_strength = old_cc


def xplets_to_serializable_dict(model):
    """
    Create a dictionary of doublets, triplets and quadruplets left after model building (i.e. used in the QUBO).
    The dictionary has the form `str(xplet) -> dict(xplet)`, where `dict(xplet)` contains all the attributes
    of the xplet, except the lists of inner and outer connections. Note that attributes of type Xplet are transformed
    into string.

    .. warning::
        This only works after model building (i.e. call to :py:meth:`hepqpr.qallse.QallseBase.build_model`).
        Also, some implementations might modify the xplets during qubo building, so it is better to call
        `model.to_qubo` beforehand.

    :param model: an implementation of :py:class:`hepqpr.qallse.QallseBase`
    :return: a dict without cyclic references.
    """
    xplets = []
    for xs in [model.qubo_doublets, model.qubo_triplets, model.quadruplets]:
        xplets += [(str(x), x.to_dict()) for x in xs]
    return dict(xplets)


def dump_qubo(model, output_path=_default_opath, prefix=_default_prefix, **markers):
    """
    Pickle a QUBO using specific markers. See also :py:meth:`use_markers`. The default filename is
    `qubo.pickle`.

    :param model: an implementation of :py:class:`~hepqpr.qallse.QallseBase`
    :param output_path: the output directory
    :param prefix: a prefix to use in the filename
    :param markers: see :py:meth:`use_markers`
    :return: the generated QUBO
    """
    with use_markers(model, **markers) as altered_model:
        Q = altered_model.to_qubo()
        with open(path_join(output_path, prefix + 'qubo.pickle'), 'wb') as f:
            pickle.dump(Q, f)
    return Q

def dump_xplets(obj, output_path=_default_opath, prefix=_default_prefix,
                format='pickle', **lib_kwargs):
    """
    Save the output of :py:meth:`xplets_to_serializable_dict` to disk.

    :param obj: the dict of xplet or a Qallse model
    :param output_path: the output directory
    :param prefix: a prefix to use in the filename
    :param format: either `pickle` or `json`
    :param lib_kwargs: extra arguments to pass to json/pickle.
    """
    if isinstance(obj, QallseBase):
        obj = xplets_to_serializable_dict(obj)

    fname = path_join(output_path, f'{prefix}xplets.{format}')
    if format == 'pickle':
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, **lib_kwargs)
    elif format == 'json':
        with open(fname, 'w') as f:
            json.dump(obj, f, cls=_XpletsJsonEncoder, **lib_kwargs)
    else:
        raise Exception(f'Unknown format: {format}')


def dump_model(model, output_path=_default_opath, prefix=_default_prefix,
               xplets_kwargs=None, qubo_kwargs=None):
    """
    Calls :py:meth:`dump_qubo` and :py:meth:`dump_xplets`.
    """
    kwargs = qubo_kwargs or dict()
    Q = dump_qubo(model, output_path, prefix, **kwargs)
    kwargs = xplets_kwargs or dict()
    dump_xplets(model, output_path, prefix, **kwargs)
    return Q
