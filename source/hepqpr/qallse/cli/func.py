import logging
import sys

from hepqpr.qallse import *

logger = logging.getLogger(__name__)


# ======= utils

def init_logging(level=logging.INFO, stream=sys.stderr):
    logging.basicConfig(
        stream=stream,
        format='%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')
    logging.getLogger('hepqpr').setLevel(level)


from contextlib import contextmanager
import time


@contextmanager
def time_this():
    time_info = [
        time.process_time(),
        time.perf_counter(),
    ]

    yield time_info

    time_info[0] = time.process_time() - time_info[0]
    time_info[1] = time.perf_counter() - time_info[1]


# ======= model building

def build_model(path, model, add_missing):
    doublets = pd.read_csv(path + '-doublets.csv')

    # prepare doublets
    if add_missing:
        print('Cheat on, adding missing doublets.')
        doublets = model.dataw.add_missing_doublets(doublets)
    else:
        p, r, ms = model.dataw.compute_score(doublets)
        print(f'INPUT -- precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')

    # build the qubo
    model.build_model(doublets=doublets)


# ======= sampling

def solve_neal(Q, seed=None, **kwargs):
    from neal import SimulatedAnnealingSampler
    # generate seed for logging purpose
    if seed is None:
        import random
        seed = random.randint(0, 1 << 31)
    # run neal
    start_time = time.process_time()
    response = SimulatedAnnealingSampler().sample_qubo(Q, seed=seed, **kwargs)
    exec_time = time.process_time() - start_time
    logger.info(f'QUBO of size {len(Q)} sampled in {exec_time:.2f}s (NEAL, seed={seed}).')
    return response


def solve_qbsolv(Q, logfile=None, seed=None, **kwargs):
    from hepqpr.qallse.other.stdout_redirect import capture_stdout
    from dwave_qbsolv import QBSolv
    # generate seed for logging purpose
    if seed is None:
        import random
        seed = random.randint(0, 1 << 31)
    # run qbsolv
    logger.debug('Running qbsolv with extra arguments: %s', kwargs)
    start_time = time.process_time()
    if logfile is not None:
        logger.debug(
            f'Writting qbsolv output to {logfile}. If you see an output in stdout, run "export PYTHONUNBUFFERED=1".')
        with capture_stdout(logfile):
            response = QBSolv().sample_qubo(Q, seed=seed, **kwargs)
    else:
        response = QBSolv().sample_qubo(Q, seed=seed, **kwargs)
    exec_time = time.process_time() - start_time
    logger.info(f'QUBO of size {len(Q)} sampled in {exec_time:.2f}s (QBSOLV, seed={seed}).')
    return response


def solve_dwave(Q, conf_file, **kwargs):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    from dwave.system import EmbeddingComposite
    from dwave.system import DWaveSampler
    sampler = DWaveSampler(config_file=conf_file, permissive_ssl=True)
    solver = EmbeddingComposite(sampler)
    logger.info(f'Using {sampler.solver} as the sub-QUBO solver.')
    if 'num_reads' not in kwargs: kwargs['num_reads'] = 10
    if 'num_repeats' not in kwargs: kwargs['num_repeats'] = 10
    return solve_qbsolv(Q, solver=solver, **kwargs)


# ======= results

def process_response(response):
    sample = next(response.samples())
    final_triplets = [Triplet.name_to_hit_ids(k) for k, v in sample.items() if v == 1]
    all_doublets = tracks_to_xplets(final_triplets)
    final_tracks, final_doublets = TrackRecreaterD().process_results(all_doublets)

    return final_doublets, final_tracks


def print_stats(dw, response, Q=None):
    final_doublets, final_tracks = process_response(response)

    en0 = 0 if Q is None else dw.compute_energy(Q)
    en = response.record.energy[0]
    print(f'SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})')
    occs = response.record.num_occurrences
    print(f'          best sample occurrence: {occs[0]}/{occs.sum()}')

    p, r, ms = dw.compute_score(final_doublets)
    print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}')
    trackml_score = dw.compute_trackml_score(final_tracks)
    print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')

    return final_doublets, final_tracks
