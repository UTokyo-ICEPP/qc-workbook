import click
from hepqpr.qallse import Volayer
import pandas as pd


def filter_doublets(hits, doublets, max_holes):
    doublets = doublets.copy()

    # compute doublet spans
    hits['volayer'] = hits[['volume_id', 'layer_id']].apply(lambda serie: Volayer.get_index(serie.tolist()), axis=1)
    doublets['span'] = hits.volayer.get(doublets.end).values - hits.volayer.get(doublets.start).values

    # filter
    return doublets[doublets.span <= max_holes + 1]


@click.command()
@click.option('-h', '--max-holes', type=int, default=1,
              help='Maximum number of holes (i.e. missing layers) allowed in doublets.')
@click.option('-i', 'hits_path', required=True, help='path to the hits files')
def cli(max_holes, hits_path):
    # load data
    doublets_path = hits_path.replace('-hits.csv', '-doublets.csv')
    hits = pd.read_csv(hits_path, index_col=0)
    doublets = pd.read_csv(doublets_path)
    # filter
    filtered_doublets = filter_doublets(hits, doublets, max_holes)
    n_discarded = len(doublets) - len(filtered_doublets)
    print(f'Discarded {n_discarded} doublets out of {len(doublets)} ({(n_discarded/len(doublets))*100:.3f}%).')
    # save
    doublets.to_csv(doublets_path + '.orig', index=False)
    filtered_doublets.to_csv(doublets_path, columns=['start', 'end'], index=False)

if __name__ == '__main__':
    cli()
