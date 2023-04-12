import os
import re

import click

from .config import *
from .doublet_making import doublet_making
from .storage import *
from .topology import DetectorModel


def generate_doublets(*args, **kwargs) -> pd.DataFrame:
    seeding_results = run_seeding(*args, **kwargs)
    doublets = structures_to_doublets(*seeding_results)
    doublets_df = pd.DataFrame(doublets, columns=['start', 'end']).drop_duplicates()
    return doublets_df


def run_seeding(hits_path=None, hits=None, config_cls=HptSeedingConfig):
    det = DetectorModel.buildModel_TrackML()
    n_layers = len(det.layers)

    hits = pd.read_csv(hits_path, index_col=False) if hits is None else hits.copy()
    hits = hits.iloc[np.where(np.in1d(hits['volume_id'], [8, 13, 17]))]

    config = config_cls(n_layers)
    # setting up structures
    spStorage = SpacepointStorage(hits, config)
    doubletsStorage = DoubletStorage()
    doublet_making(config, spStorage, det, doubletsStorage)

    # returning the results
    return hits, spStorage, doubletsStorage


def structures_to_doublets(hits: pd.DataFrame = None, sps: SpacepointStorage = None, ds: DoubletStorage = None):
    doublets = []
    for i, sp in enumerate(ds.spmIdx):
        inner_indexes = ds.inner[ds.innerStart[i]:ds.innerStart[i + 1 if i + 1 < len(ds.spmIdx) else -1]]
        doublets += [(sps.idsp[i], sps.idsp[sp]) for i in inner_indexes]
        outer_indexes = ds.outer[ds.outerStart[i]:ds.outerStart[i + 1 if i + 1 < len(ds.spmIdx) else -1]]
        doublets += [(sps.idsp[sp], sps.idsp[i]) for i in outer_indexes]

    return np.unique(np.array(doublets), axis=0)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-o', '--out', default=None)
@click.option('--score/--no-score', is_flag=True, default=True)
@click.argument('hits_path', default='/tmp/barrel_100/event000001000')
def cli(out=None, score=True, hits_path=None):
    '''
    Generate initial doublets.
    '''
    path = hits_path.replace('-hits.csv', '')
    event_id = re.search('(event[0-9]+)', hits_path)[0]
    if out is None: out = os.path.dirname(hits_path)

    print(f'Loading file {hits_path}')
    hits = pd.read_csv(path + '-hits.csv').set_index('hit_id', drop=False)

    doublets_df = generate_doublets(hits=hits)
    print(f'found {doublets_df.shape[0]} doublets.')

    if score:
        from hepqpr.qallse.data_wrapper import DataWrapper
        dw = DataWrapper.from_path(path)
        p, r, ms = dw.compute_score(doublets_df.values)
        print(f'DBLETS SCORE -- precision {p * 100}%, recall: {r * 100}% (missing doublets: {len(ms)})')

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{event_id}-doublets.csv'), 'w') as f:
        doublets_df.to_csv(f, index=False)
        print(f'doublets written to {f.name}')

    print('done')
