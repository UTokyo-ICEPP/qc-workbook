"""
Wrapper in order to get timing information on each subQUBO sampling using D-Wave & qbsolv.

Timings available (see https://docs.dwavesys.com/docs/latest/c_timing_1.html):

    1) QPU time, as returned by the DWaveSampler (response.info['timing'] field)
    2) internet latency + service time, thanks to the additional information added by the SAPI client

To record the QPU time, we just need to intercept the sample_qubo calls and store the info field somewhere.
Qbsolv accepts a function as a solver argument, which is great: we can then have complete control over the
D-Wave solver calls (see `dimod_callback` in `qbsolv/python/dwave_qbsolv/qbsolv_binding.pyx` ).

To record the other information, we need to patch the `_result_to_response_hook` function in
<`dwave_sampler.py` https://github.com/dwavesystems/dwave-system/blob/master/dwave/system/samplers/dwave_sampler.py>.
so that it adds all the relevant information to the `info` field.

Inspirations:

* For 1) https://github.com/dwavesystems/qbsolv/issues/134#issuecomment-433037625
    -  qpu_timing_example.py https://gist.github.com/dexter2206/916e407e4ce88475ea93e20d2516f78f
* For 2) https://docs.dwavesys.com/docs/latest/c_timing_5.html#timing-data-returned-by-the-dwave-cloud-client

Example usage:

>>> from dwave.system.samplers import DWaveSampler
>>> from dwave.system.composites import EmbeddingComposite
>>> dwave_solver = EmbeddingComposite(DWaveSampler(config_file='path/to/dwave.conf', permissive_ssl=True))
>>> from dwave_qbsolv import QBSolv
>>> with solver_with_timing(dwave_solver, num_reads=10) as (solver, records):
>>>     response = QBSolv().sample_qubo(Q, solver=solver)
>>> total_qpu_time = sum([r['timing']['total_real_time'] for r in records]) # in microseconds
>>> print('total QPU time (s): ', total_qpu_time * 1E-6)
>>> total_service_time = sum([(r['time_solved'] - r['time_received']).total_seconds() for r in records])
>>> print('total service time (s): ', total_service_time)
>>> total_time = sum([(r['time_resolved'] - r['time_created']).total_seconds() for r in records])
>>> print('total time (s): ', total_time)
"""

from contextlib import contextmanager
import dimod

_INTERESTING_COMPUTATION_KEYS = [
    'clock_diff',  # difference in seconds between the client-server UTC clocks
    'time_created',  # client-side: request created
    'time_received',  # server-side: request received
    'time_solved',  # server-side: response sent
    'time_resolved'  # client-side: response received
]


class TimingRecord(dict):
    """Use this wrapper to simplify the handling of times."""
    @property
    def qpu_time(self):
        return self['timing']['total_real_time'] * 1E-6  # in microseconds

    @property
    def service_time(self):
        return (self['time_solved'] - self['time_received']).total_seconds()

    @property
    def total_time(self):
        return (self['time_resolved'] - self['time_created']).total_seconds()

    @property
    def internet_latency(self):
        return self.total_time - self.service_time


def _result_to_response_hook_patch(variables, vartype):
    """see https://github.com/dwavesystems/dwave-system/blob/master/dwave/system/samplers/dwave_sampler.py"""

    def _hook(computation):
        result = computation.result()
        # get the samples. The future will return all spins so filter for the ones in variables
        samples = [[sample[v] for v in variables] for sample in result.get('solutions')]
        # the only two data vectors we're interested in are energies and num_occurrences
        vectors = {'energy': result['energies']}
        if 'num_occurrences' in result:
            vectors['num_occurrences'] = result['num_occurrences']
        # PATCH: record all interesting timing information
        info = {}
        for attr in _INTERESTING_COMPUTATION_KEYS:
            info[attr] = getattr(computation, attr, None)
        if 'timing' in result:
            info['timing'] = result['timing']
        return dimod.Response.from_samples(samples, vectors, info, vartype, variable_labels=variables)

    return _hook


@contextmanager
def solver_with_timing(sampler: dimod.Sampler, **solver_kwargs):
    """
    Wrap a given solver for use with qbsolv, recording timing information.
    For it to work, the sampler should be (or have a child of type) dwave.system.samplers.DWaveSampler.
    *Note*: If the sampler is None, (None, []) is returned, so that you can use the same code in simulation mode.

    :param sampler: the dimod sampler using D-Wave
    :param solver_kwargs: extra arguments to pass to the sample_qubo method
    :return: the new solver, a reference to the records array
    """
    if sampler is None:
        # qbsolv is running in classical mode
        # just return default values, so no error is raised.
        yield None, []
        return

    import dwave.system.samplers.dwave_sampler as spl
    original_hook = spl._result_to_response_hook
    spl._result_to_response_hook = _result_to_response_hook_patch

    records = []

    try:
        def dimod_callback(Q, best_state):
            result = sampler.sample_qubo(Q, **solver_kwargs)
            sample = next(result.samples())
            for key, value in sample.items():
                best_state[key] = value
            result.info['q_size'] = len(Q)
            records.append(result.info)
            return best_state

        # return the "solver" to use for qbsolv and a reference to the records array
        yield dimod_callback, records

    finally:
        # restore the "normal" methods
        spl._result_to_response_hook = original_hook
