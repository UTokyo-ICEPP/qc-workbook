#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create smaller datasets from a TrackML event.

During creation, the following simplifications are performed:

1. remove all hits from the endcaps;
2. keep only one instance of _double hits_ (i.e. duplicate signals on one volume from the same particle).

Then, a given percentage of particles and noisy hits a selected to be included in the new dataset.
The `weight` used to compute the TrackML score are [potentially] modified ignore low pt and/or short particles
(default cuts: <1 GeV, <5 hits).

Usage
-----

From the terminal:

.. code-block:: bash

    # create a dataset with 10% of a full TrackML event (default event=1000)
    # the new dataset is written in /tmp/mini10
    create_dataset -n 0.1 -o /tmp --prefix mini10

From a script:

.. code::

    from hepqpr.qallse.dsmaker import create_dataset
    metadata, path = create_dataset(
        density=0.1,
        output_path='/tmp',
        prefix='mini10'
    )
    # => path='/tmp/mini10/event000001000'

"""

import json
import os
import random
import re
from datetime import datetime
from typing import Dict, Tuple

import click
import numpy as np
import pandas as pd

BARREL_VOLUME_IDS = [8, 13, 17]

import logging

logger = logging.getLogger(__name__)


def _get_default_input_path():
    from os import path
    return path.join(path.dirname(path.realpath(__file__)), 'data', 'event000001000')


def create_dataset(
        input_path=_get_default_input_path(),
        output_path='.',
        density=.1,
        min_hits_per_track=5,
        high_pt_cut=1.,
        double_hits_ok=False,
        gen_doublets=False,
        prefix=None, random_seed=None, phi_bounds=None) -> Tuple[Dict, str]:
    input_path = input_path.replace('-hits.csv', '')  # just in case

    # capture all parameters, so we can dump them to a file later
    input_params = locals()

    # initialise random
    if random_seed is None:
        random_seed = random.randint(0, 1<<30)
    random.seed(random_seed)

    event_id = re.search('(event[0-9]+)', input_path)[0]

    # compute the prefix
    if prefix is None:
        prefix = f'ez-{density}'
        if high_pt_cut > 0:
            prefix += f'_hpt-{high_pt_cut}'
        else:
            prefix += '_baby'
        if double_hits_ok:
            prefix += '_dbl'

    # ---------- prepare data

    # load the data
    hits = pd.read_csv(input_path + '-hits.csv')
    particles = pd.read_csv(input_path + '-particles.csv')
    truth = pd.read_csv(input_path + '-truth.csv')

    # add indexes
    particles.set_index('particle_id', drop=False, inplace=True)
    truth.set_index('hit_id', drop=False, inplace=True)
    hits.set_index('hit_id', drop=False, inplace=True)

    # create a merged dataset with hits and truth
    df = hits.join(truth, rsuffix='_', how='inner')

    logger.debug(f'Loaded {len(df)} hits from {input_path}.')

    # ---------- filter hits

    # keep only hits in the barrel region
    df = df[hits.volume_id.isin(BARREL_VOLUME_IDS)]
    logger.debug(f'Filtered hits from barrel. Remaining hits: {len(df)}.')

    if phi_bounds is not None:
        df['phi'] = np.arctan2(df.y, df.x)
        df = df[(df.phi >= phi_bounds[0]) & (df.phi <= phi_bounds[1])]
        logger.debug(f'Filtered using phi bounds {phi_bounds}. Remaining hits: {len(df)}.')

    # store the noise for later, then remove them from the main dataframe
    # do this before filtering double hits, as noise will be thrown away as duplicates
    noise_df = df.loc[df.particle_id == 0]
    df = df[df.particle_id != 0]

    if not double_hits_ok:
        df.drop_duplicates(['particle_id', 'volume_id', 'layer_id'], keep='first', inplace=True)
        logger.debug(f'Dropped double hits. Remaining hits: {len(df) + len(noise_df)}.')

    # ---------- sample tracks

    num_tracks = int(df.particle_id.nunique() * density)
    sampled_particle_ids = random.sample(df.particle_id.unique().tolist(), num_tracks)
    df = df[df.particle_id.isin(sampled_particle_ids)]

    # ---------- sample noise

    num_noise = int(len(noise_df) * density)
    sampled_noise = random.sample(noise_df.hit_id.values.tolist(), num_noise)
    noise_df = noise_df.loc[sampled_noise]

    # ---------- recreate hits, particle and truth

    new_hit_ids = df.hit_id.values.tolist() + noise_df.hit_id.values.tolist()
    new_hits = hits.loc[new_hit_ids]
    new_truth = truth.loc[new_hit_ids]
    new_particles = particles.loc[sampled_particle_ids]

    # ---------- fix truth

    if high_pt_cut > 0:
        # set low pt weights to 0
        hpt_mask = np.sqrt(truth.tpx ** 2 + truth.tpy ** 2) >= high_pt_cut
        new_truth.loc[~hpt_mask, 'weight'] = 0
        logger.debug(f'High Pt hits: {sum(hpt_mask)}/{len(new_truth)}')

    if min_hits_per_track > 0:
        short_tracks = new_truth.groupby('particle_id').filter(lambda g: len(g) < min_hits_per_track)
        new_truth.loc[short_tracks.index, 'weight'] = 0

    new_truth.weight = new_truth.weight / new_truth.weight.sum()

    # ---------- write data

    # write the dataset to disk
    output_path = os.path.join(output_path, prefix)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, event_id)

    new_hits.to_csv(output_path + '-hits.csv', index=False)
    new_truth.to_csv(output_path + '-truth.csv', index=False)
    new_particles.to_csv(output_path + '-particles.csv', index=False)

    # ---------- write metadata

    metadata = dict(
        num_hits=new_hits.shape[0],
        num_tracks=num_tracks,
        num_important_tracks=new_truth[new_truth.weight != 0].particle_id.nunique(),
        num_noise=num_noise,
        random_seed=random_seed,
        time=datetime.now().isoformat(),
    )
    for k, v in metadata.items():
        logger.debug(f'  {k}={v}')

    metadata['params'] = input_params

    with open(output_path + '-meta.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    # ------------ gen doublets

    if gen_doublets:

        from hepqpr.qallse.seeding import generate_doublets
        doublets_df = generate_doublets(hits=new_hits)
        with open(output_path + '-doublets.csv', 'w') as f:
            doublets_df.to_csv(f, index=False)
            logger.info(f'Doublets (len={len(doublets_df)}) generated in f{output_path}.')

    return metadata, output_path


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-n', '--density', type=click.FloatRange(0, 1), default=.1,
              help='The sampling to apply, in percent.')
@click.option('--hpt', type=float, default=1.,
              help='Only select tracks with a transverse momentum '
                   'higher or equal than FLOAT (in GeV, inclusive)')
@click.option('--double-hits/--no-double-hits', is_flag=True, default=False,
              help='Keep only one instance of double hits.')
@click.option('-m', '--min-hits', type=int, default=5,
              help='The minimum number of hits per tracks (inclusive)')
@click.option('-p', '--prefix', type=str, default=None,
              help='Name of the dataset output directory')
@click.option('-s', '--seed', type=int, default=None,
              help='Seed to use when initializing the random module')
@click.option('--no-doublets', is_flag=True, default=False,
              help='Don\'t generate initial doublets')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Be verbose.')
@click.option('-o', '--output-path', default='.',
              help='Where to create the dataset directoy')
@click.option('-i', 'input_path', default=_get_default_input_path(),
              help='Path to the original event hits file')
def cli(density, hpt, double_hits, min_hits, prefix, seed,
        no_doublets, verbose, output_path, input_path):
    '''
    Create datasets from TrackML events suitable for HEPQPR.Qallse.

    Main simplifications: no hits from the end-caps, no double hits
    (use the <double-hits> flag to force the inclusion of double hits).
    Optional simplifications: if <hpt> and/or <min-hits> is set, the
    weights used by the TrackML scores are set to 0 for hits with a
    low pt and/or belonging to short tracks.

    If <density> is set, particles and noise are random-sampled using
    the given percentage. This shouldn't alter the dataset characteristics,
    except for the noise-to-hit ratio (a bit lower).
    '''
    if verbose:
        import sys
        logging.basicConfig(
            stream=sys.stderr,
            format="%(asctime)s [dsmaker] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S',
            level=logging.DEBUG)

    meta, path = create_dataset(
        input_path, output_path,
        density, min_hits,
        hpt, double_hits,
        not no_doublets, prefix, seed)

    seed, density = meta['random_seed'], meta['num_tracks']
    print(f'Dataset written in {path}* (seed={seed}, num. tracks={density})')

if __name__ == "__main__":
    cli()
