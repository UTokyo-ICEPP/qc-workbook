import sys

import click
import pickle
from os import path as op

from hepqpr.qallse.cli.func import *
from hepqpr.qallse.cli.utils import *


class GlobalOptions:
    def __init__(self, hits_path, opath=None, prefix=''):
        self.hits_path = hits_path
        self.output_path = opath
        self.prefix = prefix
        self._dw = None  # lazy creation (not created with --help)

    @property
    def path(self):
        if self.hits_path is None:
            # simulate the required=True click option, because if used directly,
            # one cannot display a subcommand help without passing the hit path ...
            click.echo("Error: Missing option '-i' / '--hits-path'.", err=True)
            sys.exit(1)
        return self.hits_path.replace('-hits.csv', '')

    @property
    def dw(self):
        if self._dw is None: self._dw = DataWrapper.from_path(self.path)
        return self._dw

    def get_output_path(self, filename):
        return op.join(self.output_path, self.prefix + filename)


# ------

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--debug/--no-debug', is_flag=True, default=False)
@click.option('-i', '--hits-path',
              help='[required] Path to the hits file.')
@click.option('-o', '--output-path', default=None, metavar='directory',
              help='Where to save the output files.')
@click.option('-p', '--prefix', default='', metavar='text',
              help='Prefix prepended to all output files.')
@click.pass_context
def cli(ctx, debug, hits_path, output_path, prefix):
    '''
      Solve the pattern recognition problem using QA.

      The <hits-path> is the path to a hit file generated using the
      `create_dataset` method. The directory should also contain a truth file
      and the initial doublets file (created either during `create_dataset` or
      using the `run_seeding` script).

      Output files will be saved to the given <output-path>, if any, using default names.
      If set, the <prefix> will be prepended to all output files.
    '''
    # configure logging
    init_logging(logging.DEBUG if debug else logging.INFO)

    # load input data
    ctx.obj = GlobalOptions(hits_path, output_path, prefix)


@cli.command('build')
@click.option('--add-missing', is_flag=True, default=False,
              help='If set, ensure 100% input recall.')
@click.option('-c', '--cls', default='qallse_d0', metavar='module_name',
              help='Model to use.')
@click.option('-e', '--extra', type=str, multiple=True, metavar='key=value',
              help='Override default model configuration.')
@click.pass_obj
def cli_build(ctx, add_missing, cls, extra):
    '''
    Generate the QUBO.

    The QUBO and the xplets used by it are saved as pickle files in the current directory
    (use --output-path and --prefix options to change it).

    <add-missing> will add any true missing doublet to the input, ensuring an input recall of 100%.
    <cls> lets you choose which model to use: qallse_d0 (default), qallse, qallse_mp, etc.
    <extra> are key=values corresponding to configuration options of the model, (e.g. -e qubo_conflict_strength=0.5).
    '''
    from hepqpr.qallse import dumper
    extra_config = extra_to_dict(extra)
    ModelClass = qallse_class_from_string('.' + cls)
    model = ModelClass(ctx.dw, **extra_config)

    build_model(ctx.path, model, add_missing)
    dumper.dump_model(model, ctx.output_path, ctx.prefix, qubo_kwargs=dict(w_marker=None, c_marker=None))
    print('Wrote qubo to', ctx.get_output_path("qubo.pickle"))

@cli.command('qbsolv')
@click.option('-q', '--qubo', default=None, metavar='filepath',
              help='Path a the pickled QUBO.')
@click.option('-dw', '--dwave-conf', default=None, type=str, metavar='filepath',
              help='Path to a dwave.conf. If set, use a D-Wave as the sub-QUBO solver.')
@click.option('-v', '--verbosity', type=click.IntRange(-1, 6), default=-1, metavar='int',
              help='qbsolv verbosity.')
@click.option('-l', '--logfile', type=str, default=None, metavar='filepath',
              help='Where to redirect the qbsolv output. Does only make sense for verbosity > 0.')
@click.option('-e', '--extra', type=str, multiple=True, metavar='key=<value:int>',
              help='Additional options to qbsolv. '
                   'Allowed keys: seed, num_repeats, (+If D-Wave: num_reads).')
@click.pass_obj
def cli_qbsolv(ctx, qubo, dwave_conf, verbosity, logfile, extra):
    '''
    Sample a QUBO using qbsolv (!slower!) and a D-Wave (optional).

    By default, this will run qbsolv (https://github.com/dwavesystems/qbsolv)
    in simulation. To use a D-Wave, set the <dw> option to
    a valid dwave configuration file (see https://cloud.dwavesys.com/leap/).

    <qubo> is the path to the pickled qubo (default to <output-path>/<prefix>qubo.pickle).
    <verbosity> and <extra> are passed to qbsolv. <logfile> will redirect all qbsolv output to
    a file (see also the parse_qbsolv script).
    '''
    try:
        if qubo is None: qubo = ctx.get_output_path('qubo.pickle')
        with open(qubo, 'rb') as f:
            Q = pickle.load(f)
    except:
        print(f'Failed to load QUBO. Are you sure {qubo} is a pickled qubo file ?')
        sys.exit(-1)

    qbsolv_kwargs = extra_to_dict(extra, typ=int)
    qbsolv_kwargs['logfile'] = logfile
    qbsolv_kwargs['verbosity'] = verbosity

    if dwave_conf is not None:
        response = solve_dwave(Q, dwave_conf, **qbsolv_kwargs)
    else:
        response = solve_qbsolv(Q, **qbsolv_kwargs)

    print_stats(ctx.dw, response, Q)
    if ctx.output_path is not None:
        oname = ctx.get_output_path('qbsolv_response.pickle')
        with open(oname, 'wb') as f: pickle.dump(response, f)
        print(f'Wrote response to {oname}')


@cli.command('neal',
             help='Sample a QUBO using neal.')
@click.option('-q', '--qubo', default=None, metavar='filepath',
              help='Path to the pickled QUBO. Default to <output_path>/<prefix>qubo.pickle')
@click.option('-s', '--seed', default=None, type=int, metavar='int',
              help='Seed to use.')
@click.pass_obj
def cli_neal(ctx, qubo, seed):
    '''
    Solve a QUBO using neal (!fast!)

    neal (https://github.com/dwavesystems/dwave-neal) is a simulated annealing sampler.
    It is faster than qbsolv by two order of magnitude with similar (if not better) results.
    '''
    try:
        if qubo is None: qubo = ctx.get_output_path('qubo.pickle')
        with open(qubo, 'rb') as f:
            Q = pickle.load(f)
    except:
        print(f'Failed to load QUBO. Are you sure {qubo} is a pickled qubo file ?')
        sys.exit(-1)

    response = solve_neal(Q, seed=seed)
    print_stats(ctx.dw, response, Q)
    if ctx.output_path is not None:
        oname = ctx.get_output_path('neal_response.pickle')
        with open(oname, 'wb') as f: pickle.dump(response, f)
        print(f'Wrote response to {oname}')


@cli.command('quickstart',
             context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def cli_quickstart(ctx):
    '''
    Run the whole algorithm (build+neal).

    This accepts the same options as the build command. If no <output-path> is set,
    a temporary directory is created for the time of the run and deleted on exit.

    Minimal example using a very small dataset:

    \b
        create_dataset -n 0.01 -p mini
        qallse -i mini/event000001000-hits.csv quickstart

    '''

    def _chain():
        ctx.forward(cli_build)
        ctx.invoke(cli_neal)

    if ctx.obj.output_path is None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx.obj.output_path = tmpdir
            _chain()
    else:
        _chain()


@cli.command('plot')
@click.option('-r', '--response', metavar='filepath', required=True,
              help='Path to the response file.')
@click.option('-d', '--dims', default='xy', type=click.Choice(['xy', 'zr', 'zxy']),
              help='Dimensions of the plot.')
@click.option('-m', '--mode', default='d', type=click.Choice(['d', 't', 'dt']),
              help='Plot the doublets only (d), the triplets only (t), or both (dt).')
@click.pass_obj
def cli_plot(ctx, response, dims, mode):
    '''
    Plot the final doublets and final tracks.

    This uses (https://plot.ly) and the hepqpr.qallse.plotting module to
    show the final tracks and doublets.
    The plots are saved as html files either in <output-path> or in the current directory.

    WARNING: don't try to plot results from large datasets, especially 3D plots !!
    '''
    from hepqpr.qallse.plotting import iplot_results, iplot_results_tracks

    dims = list(dims)

    with open(response, 'rb') as f:
        r = pickle.load(f)
    final_doublets, final_tracks = process_response(r)
    _, missings, _ = diff_rows(final_doublets, ctx.dw.get_real_doublets())


    if ctx.output_path is None:
        ctx.output_path = '.'
    dout = ctx.get_output_path('plot-doublets.html')
    tout = ctx.get_output_path('plot-triplets.html')

    if 'd' in mode:
        iplot_results(ctx.dw, final_doublets, missings, dims=dims, filename=dout)
    if 't' in mode:
        iplot_results_tracks(ctx.dw, final_tracks, dims=dims, filename=tout)
